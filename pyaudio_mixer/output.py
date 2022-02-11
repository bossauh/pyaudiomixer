import asyncio
from typing import Union
import os
import queue
import threading
import time
from pathlib import Path
from typing import List

import ffmpy
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from .exceptions import *
from .utils import BasicFX
from .input import InputTrack

sd.default.channels = 2
sd.default.samplerate = 44100
sd.default.dtype = "float32"


class OutputTrack:

    """
    Parameters
    ----------
    `name`: str
        The name of this track.
    `callback` : Callable
        A user supplied function that looks and functions like this. Defaults to None. This callback can be used to modify the data before playing it back to the user.

        >>> def callback(track: OutputTrack, data: np.ndarray) -> np.ndarray:
        >>>     # Modify `data` to your likings if needed then you must return it back.
        >>>     return data
    `sounddevice_parameters` : dict
        Key, Value pair that will be passed as parameter to sd.OutputStream. Defaults to None.
    `conversion_path` : str
        Directory to store ffmpeg conversions. FFMpeg conversions are done when the provided file format is not supported. Defaults to None. When this is None, a UnsupportedFormat is instead raised everytime PyAudioMixer encounters a unsupported audio format.
    `apply_basic_fx` : bool
        Whether to apply the basic effects such as the volume changer. Defaults to True. This uses the BasicFX class.
    `volume` : int
        The volume of this track. Defaults to 1.0 (100%). Volume changing is controlled by the BasicFX class.
    `effect_parameters` :
        Dictionary that tells the BasicFX class what effects to use. Key being the function effect and the value (which is a another dictionary) being the parameters. Defaults to None. (Do not set a set_volume effect here. That is controlled by the `volume` parameter)
    `queue_maxsize` : int
        The maxsize parameter passed onto the queue.Queue of this track. Defaults to 50. You usually don't need to touch this.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.callback = kwargs.get("callback")
        self.sounddevice_parameters = kwargs.get("sounddevice_parameters", {})
        self.conversion_path = kwargs.get("conversion_path")
        self.apply_basic_fx = kwargs.get("apply_basic_fx", True)
        self.__kwargs = kwargs

        # Main queue, this is where all data that
        # then gets outputted to the user is stored.
        # Based on my testing, there is usually no reason to
        # change the max size, but it can be changed by passing
        # `queue_maxsize` as a parameter of this class.
        self.q = queue.Queue(maxsize=kwargs.get("queue_maxsize", 50))

        # Signal Variables
        self._clear_signal = False
        self._stop_signal = False
        self._stop_cast_signal = False
        self._stopped = True
        self._playing = False
        self._playing_details = {}

        # Effect variables
        self.basicfx = BasicFX(
            dtype=self.sounddevice_parameters.get("dtype", sd.default.dtype[1]),
            samplerate=self.sounddevice_parameters.get("samplerate", sd.default.samplerate)
        )
        __effect_params = kwargs.get("effect_parameters", {})
        self.effect_parameters = {}
        self.update_effects(__effect_params)

        # Start the track on initialization
        self.stream = None
        self.start()

    @property
    def volume(self) -> float:
        return self.effect_parameters["set_volume"]["factor"]

    @volume.setter
    def volume(self, value: float) -> None:
        self.effect_parameters["set_volume"]["factor"] = value

    @property
    def playing_details(self) -> Union[None, dict]:
        """
        Get details about the currently playing file (played via play_file coroutine). Returns None if there is no playing file.
        """

        if self._playing_details:
            return self._playing_details
        return None
    
    def update_effects(self, effect_parameters: dict) -> dict:
        """
        Update the effect_parameters attribute of this class. Check docstring of the class itself for more information about that attribute. (Also to know how this parameter is formatted)

        Returns
        -------
        `dict` :
            A dictionary containing what effects are enabled (and their parameters)
        """
        
        set_volume = self.effect_parameters.get("set_volume", {})
        effect_parameters.pop("set_volume", None)
        self.effect_parameters = {
            "set_volume": {
                "factor": set_volume.get("factor", self.__kwargs.get("volume", 1.0))
            },
            **effect_parameters
        }
        return self.effect_parameters

    def start(self) -> None:
        threading.Thread(target=self.__start__, daemon=True).start()

        # Wait for it to start before returning
        while self._stopped:
            time.sleep(0.001)

    async def stop(self) -> None:
        await self.abort()
        self.stop_cast()
        self._stop_signal = True

        # Wait for it to stop before returning
        while not self._stopped:
            await asyncio.sleep(0.001)

    async def abort(self) -> None:
        """
        Clears the queue which in turn causes all audio to stop playing. This does not actually stop the stream.
        """

        if self._playing:
            self._clear_signal = True
            self._playing_details = {}

        while self._playing:
            await asyncio.sleep(0.001)
        
    def stop_cast(self) -> None:
        """Stop the currently casted input if there is any."""
        self._stop_cast_signal = True
        
    def cast_input(self, inp: InputTrack) -> None:
        """
        Direct all data of the provided input track to this output track.
        This is a non blocking function that runs in the background. 
        Casting can be stopped by calling `cast_stop()`

        Notes
        -----
        - This does not wait for the cast to start therefore this function returns immediately.

        Parameters
        ----------
        `inp` : InputTrack
            The input track to use.

        Raises
        ------
        `RuntimeError` :
            Raised when the provided track isn't even running to begin with.
        """

        if inp._stopped:
            raise RuntimeError("input track is not running")

        threading.Thread(target=self.__cast_input__, args=(inp,), daemon=True).start()
    
    def __cast_input__(self, inp: InputTrack) -> None:
        while not self._stop_cast_signal:
            frame = inp.read()
            if inp._stopped:
                break

            if frame is not None:
                self.write(frame)
        
        self._stop_cast_signal = False

    def write(
        self,
        data: np.ndarray,
        wait: bool = True,
        resample: bool = False,
        resampling_method: str = "soxr_vhq",
        original_samplerate: int = None
    ) -> bool:
        """
        Write the provided data into the buffer (i.e., play it on the speakers).

        Parameters
        ----------
        `data` : np.ndarray 
            The data to write.
        `wait` : bool
            Wait for there to be a space in the queue. Defaults to True. If this is False, this function returns instantly.
        `resample` : bool
            Whether to resample the given data to match this track's samplerate. Defaults to False. If this is true, you have to provide the original samplerate via the `original_samplerate` parameter.
        `resampling_method` : str
            Method to use for resampling. Read more about it on `resample()`'s docstring. Defaults to "soxr_vhq".
        `original_samplerate` : int
            The samplerate of the provided data. You only need this parameter if `resample` is True.

        Returns
        -------
        `bool` :
            Whether putting it in the queue was successfull. If wait is True, this is usually always True. This will be False if wait is False and the queue is somehow full at the of calling this write() method.

        Raises
        ------
        `InterruptedError` : 
            Raised when abort() get's called. How abort() basically works is that it first sends the clear signal, now once this function is called, we check if the clear signal has been sent and if it has been sent then it raises a InterruptedError, telling the caller that it's time to stop writing frames. (oh and it also clears the queue)
        `ValueError` :
            Raised when `resample` is True but `original_samplerate` was not provided.
        """

        if self._clear_signal:
            self.q.queue.clear()
            self._clear_signal = False
            raise InterruptedError

        if resample:
            if original_samplerate is None:
                raise ValueError("original_samplerate must be provided")
            data = self.resample(data, original_samplerate, resampling_method)

        try:
            self.q.put(data, block=wait)
            return True
        except queue.Full:
            return False

    def resample(self, data: np.ndarray, original: int, type_: str = "soxr_vhq") -> np.ndarray:
        """
        Resample audio data to match the track's samplerate.

        Parameters
        ----------
        `data` : np.ndarray
            Audio ndarray with shape of (frames, channels)
        `original` : int
            The original samplerate.
        `type_` : str
            Resampling method. Refer to [libora.resample's](https://librosa.org/doc/main/generated/librosa.resample.html) documentation.

        Returns
        -------
        `np.ndarray` :
            The resample audio data.
        """

        data = np.swapaxes(data, -1, 0)
        data = librosa.resample(data, original, self.stream.samplerate, type_)
        data = np.swapaxes(data, 0, -1)
        return data

    def chunk_split(self, data: np.ndarray, size: int = 512) -> List[np.ndarray]:
        """
        Split the provided ndarray by chunks.

        Parameters
        ----------
        `data` : np.ndarray
            Audio data with the shape of (frames, channels). The channels doesn't really matter.
        `size` : int
            What the size of each chunk should be.

        Returns
        -------
        `List[np.ndarray]` :
            The list of ndarrays.
        """

        n = len(data) / size
        if n < 1:
            n = 1

        return np.array_split(data, n)

    async def play_file(
        self,
        path: str,
        blocking: bool = False,
        resample: bool = True,
        chunk_size: int = 512,
        load_in_memory: bool = True,
        **kwargs
    ) -> None:

        """
        Play the provided audio file.

        Notes
        -----
        - Keep in mind, this method loads the entire audio file into memory. It is only released once the audio is done playing.
        - FFmpeg will be used for converting files into .wav files if the provided format is not supported.

        Parameters
        ----------
        `path` : str
            Path.
        `blocking` : bool
            Whether to block until the audio file is done playing. Defaults to False.
        `resample` : bool
            Whether to call self.resample to match this track's samplerate. Defaults to True.
        `chunk_size` : int
            The entire audio data is split into chunks. This defines the length of each chunk. Defaults to 512.
        `load_in_memory` : bool
            Whether to load the entire file to memory. Defaults to True.
        **kwargs :
            Other parameters to pass to soundfile.read or soundfile.blocks if load_in_memory is False.
        """

        # Stop whatever is playing (if there is any)
        await self.abort()

        if "always_2d" not in kwargs.keys():
            kwargs["always_2d"] = True

        if "dtype" not in kwargs.keys():
            kwargs["dtype"] = "float32"

        if not load_in_memory:

            target_sr = self.stream.samplerate

            # Automatically figure out the best blocksize and resampling method
            if "blocksize" not in kwargs.keys():
                if target_sr >= 88200:
                    kwargs["blocksize"] = 6192
                else:
                    kwargs["blocksize"] = 512

            if (target_sr >= 8000) and (target_sr <= 22050):
                resampling_type = "soxr_qq"
            elif ((target_sr >= 44100) and (target_sr <= 48000)) or target_sr >= 176400:
                resampling_type = "fft"
            elif (target_sr >= 88200) and (target_sr <= 96000):
                resampling_type = "linear"
            else:
                resampling_type = "soxr_vhq"

        load_method = {
            True: sf.read,
            False: sf.blocks
        }

        try:
            data = load_method[load_in_memory](path, **kwargs)
            if load_in_memory:
                data, samplerate = data
            else:
                _, samplerate = sf.read(path, frames=1)
        except RuntimeError as e:
            if not self.conversion_path:
                raise UnsupportedFormat(e)

            # Create if the directory to the conversion path does not exist
            Path(self.conversion_path).mkdir(parents=True, exist_ok=True)
            out = os.path.basename(path).split(".")[0] + ".wav"
            out = os.path.join(self.conversion_path, out)

            ff = ffmpy.FFmpeg(
                inputs={path: None},
                outputs={out: None},
                global_options=["-loglevel", "quiet", "-y"]
            )

            ff.run()
            return await self.play_file(out, blocking, resample, chunk_size, load_in_memory, **kwargs)

        def match_channels(d):
            # Match the number of channels of this track.
            try:
                channel_count = d.shape[1]
            except IndexError:
                channel_count = 1

            if channel_count != self.stream.channels:
                if (channel_count == 1) and (self.stream.channels == 2):
                    channel_count = 2
                d = np.repeat(d, channel_count, axis=-1)
            return d

        if load_in_memory:
            data = match_channels(data)

            if resample:
                # Match the samplerate of this track
                data = self.resample(data, samplerate)
            data = self.chunk_split(data, chunk_size)

        def _write():
            for d in data:
                try:

                    if not load_in_memory:
                        d = match_channels(d)
                        if resample:
                            d = self.resample(d, samplerate, resampling_type)

                    self.write(d)
                except (KeyboardInterrupt, InterruptedError):
                    break

        # Assign playing details
        __detail_sr = self.stream.samplerate if resample else samplerate
        self._playing_details = {
            "file": path,
            "duration": librosa.get_duration(filename=path, sr=__detail_sr),
            "samplerate": __detail_sr,
            "read": 0  # How many seconds were already read
        }

        if blocking:
            _write()
            while self._playing:
                await asyncio.sleep(0.001)
        else:
            threading.Thread(target=_write, daemon=True).start()
            while not self._playing:
                await asyncio.sleep(0.001)

    def _apply_basic_fx(self, data: np.ndarray) -> np.ndarray:
        for f in self.basicfx.effects:
            params = self.effect_parameters.get(f.__name__)
            if params is not None:
                data = f(data, **params)
        return data

    def __start__(self) -> None:
        with sd.OutputStream(**self.sounddevice_parameters) as f:
            self._stopped = False
            self.stream = f
            while not self._stop_signal:
                try:
                    data = self.q.get(block=False)
                except queue.Empty:
                    data = None

                # Call the callback (yes even if it's None)
                if self.callback:
                    data = self.callback(self, data)

                if data is not None:
                    self._playing = True

                    if self.apply_basic_fx:
                        data = self._apply_basic_fx(data)

                    f.write(data)
                else:
                    self._playing = False

                time.sleep(0.001)

        """This code is only reached once the stop signal is True. (i.e., track has been stopped)"""
        self._stopped = True
        self.stream = None
        self._stop_signal = False

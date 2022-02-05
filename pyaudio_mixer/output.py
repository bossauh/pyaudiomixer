import asyncio
import math
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
        Whether to apply the basic effects such as the volume changer. Defaults to True.
    `volume` : int
        The volume of this track. Defaults to 1.0 (100%).
    `queue_maxsize` : int
        The maxsize parameter passed onto the queue.Queue of this track. Defaults to 50. You usually don't need to touch this.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.callback = kwargs.get("callback")
        self.sounddevice_parameters = kwargs.get("sounddevice_parameters", {})
        self.conversion_path = kwargs.get("conversion_path")
        self.apply_basic_fx = kwargs.get("apply_basic_fx", True)

        # Main queue, this is where all data that
        # then gets outputted to the user is stored.
        # Based on my testing, there is usually no reason to
        # change the max size, but it can be changed by passing
        # `queue_maxsize` as a parameter of this class.
        self.q = queue.Queue(maxsize=kwargs.get("queue_maxsize", 50))

        # Signal Variables
        self._clear_signal = False
        self._stop_signal = False
        self._vol = kwargs.get("volume", 1.0)
        self._stopped = True
        self._playing = False

        # Start the track on initialization
        self.stream = None
        self.start()

    @property
    def volume(self) -> float:
        return self._vol

    @volume.setter
    def volume(self, value: float) -> None:
        self._vol = value

    def start(self) -> None:
        threading.Thread(target=self.__start__, daemon=True).start()

        # Wait for it to start before returning
        while self._stopped:
            time.sleep(0.001)

    async def stop(self) -> None:
        await self.abort()
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

        while self._playing:
            await asyncio.sleep(0.001)

    def write(self, data: np.ndarray, wait: bool = True) -> bool:
        """
        Write the provided data into the buffer (i.e., play it on the speakers).

        Parameters
        ----------
        `data` : np.ndarray 
            The data to write.
        `wait` : bool
            Wait for there to be a space in the queue. Defaults to True. If this is False, this function returns instantly.

        Returns
        -------
        `bool` :
            Whether putting it in the queue was successfull. If wait is True, this is usually always True. This will be False if wait is False and the queue is somehow full at the of calling this write() method.

        Raises
        ------
        `InterruptedError` : 
            Raised when abort() get's called. How abort() basically works is that it first sends the clear signal, now once this function is called, we check if the clear signal has been sent and if it has been sent then it raises a InterruptedError, telling the caller that it's time to stop writing frames. (oh and it also clears the queue)
        """

        if self._clear_signal:
            self.q.queue.clear()
            self._clear_signal = False
            raise InterruptedError

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
        **kwargs :
            Other parameters to pass to soundfile.read
        """

        # Stop whatever is playing (if there is any)
        await self.abort()

        if "always_2d" not in kwargs.keys():
            kwargs["always_2d"] = False

        if "dtype" not in kwargs.keys():
            kwargs["dtype"] = "float32"

        try:
            data, samplerate = sf.read(path, **kwargs)
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
            return await self.play_file(out, blocking, resample, chunk_size, **kwargs)

        # Match the number of channels of this track.
        try:
            channel_count = data.shape[1]
        except IndexError:
            channel_count = 1

        if channel_count != self.stream.channels:
            if (channel_count == 1) and (self.stream.channels == 2):
                channel_count = 2
            data = np.repeat(data, channel_count, axis=-1)

        if resample:
            # Match the samplerate of this track
            data = self.resample(data, samplerate)
        data = self.chunk_split(data, chunk_size)

        def _write():
            for d in data:
                try:
                    self.write(d)
                except (KeyboardInterrupt, InterruptedError):
                    break

        if blocking:
            _write()
            while self._playing:
                await asyncio.sleep(0.001)
        else:
            threading.Thread(target=_write, daemon=True).start()
            while not self._playing:
                await asyncio.sleep(0.001)

    def _apply_basic_fx(self, data: np.ndarray) -> np.ndarray:
        return np.multiply(data, pow(
            2, (math.sqrt(math.sqrt(math.sqrt(self._vol))) * 192 - 192) / 6), casting="unsafe")

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

        """This code is only reached once the stop signal is True. (i.e., track has been stopped)"""
        self._stopped = True
        self.stream = None
        self._stop_signal = False

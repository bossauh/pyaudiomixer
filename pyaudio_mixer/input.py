from typing import Union
import sounddevice as sd
import threading
import time
import numpy as np
from .utils import BasicFX


class InputTrack:

    """
    Parameters
    ----------
    `name` : str
        The name of this track.
    `sounddevice_parameters` : dict
        Key, Value pair that will be passed as parameters to sd.InputStream. Defaults to None.
    `callback` : Callable
        A user supplied function that looks and functions like the one provided below. Defaults to None. This callback can be used to modify the data before being returned by the .read() method.

        >>> def callback(track: InputTrack, data: np.ndarray, overflow: bool) -> np.ndarray:
        >>>     # Modify `data` to your likings if needed. Then return it back either as a ndarray again or "None"
        >>>     return data
    `apply_basic_fx` : bool
        Whether to apply the basic effects such as the volume changer. Defaults to True. This uses the BasicFX class.
    `volume` : float
        The volume of this track (how loud the input is). Defaults to 1.0 (100%). Volume changing is controlled by the BasicFX class.
    `effect_parameters` :
        Dictionary that tells the BasicFX class what effects to use. Key being the function effect and the value (which is a another dictionary) being the parameters. Defaults to None. (Do not set a set_volume effect here. That is controlled by the `volume` parameter)
    `chunk_size` : The size of each chunk returned from .read(). Defaults to 512.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.sounddevice_parameters = kwargs.get("sounddevice_parameters", {})
        self.chunk_size = kwargs.get("chunk_size", 512)
        self.callback = kwargs.get("callback")
        self.apply_basic_fx = kwargs.get("apply_basic_fx", True)
        self.__kwargs = kwargs

        # Signal Variables
        self._stop_signal = False
        self._stopped = True

        # Data Variable
        self.__data = None
        self.__buffer = []
        self.overflow = False

        # BasicFX
        self.basicfx = BasicFX(
            dtype=self.sounddevice_parameters.get("dtype", sd.default.dtype[0]),
            samplerate=self.sounddevice_parameters.get("samplerate", sd.default.samplerate)
        )
        __effect_params = kwargs.get("effect_parameters", {})
        self.effect_parameters = {}
        self.update_effects(__effect_params)

        self.stream = None
        self.start()
    
    @property
    def volume(self) -> float:
        return self.effect_parameters["set_volume"]["factor"]
    
    @volume.setter
    def volume(self, value: float) -> None:
        self.effect_parameters["set_volume"]["factor"] = value
    
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
    
    def _apply_basic_fx(self, data: np.ndarray) -> np.ndarray:

        if data is None:
            return

        for effect in self.basicfx.effects:
            params = self.effect_parameters.get(effect.__name__)
            if params is not None:
                data = effect(data, **params)
        return data
    
    def read(self) -> Union[np.ndarray, None]:

        """
        Read the data coming from the InputStream.

        Returns
        -------
        `np.ndarray` :
            Audio data with shape of (frames (or size of chunks), channels).
        `None` :
            If the data that's supposed to be returned is the same as the last few returned data. When calling .read() constantly (which you most likely would), you should always check if the value is None.
        """

        data = self.__data
        if self.callback:
            data = self.callback(self, data, self.overflow)
        
        if self.apply_basic_fx:
            data = self._apply_basic_fx(data)
        
        data = np.resize(
            data,
            (
                self.chunk_size,
                self.sounddevice_parameters.get(
                    "channels",
                    sd.default.channels[0]
                )
            )
        )

        if any((data == x).all() for x in self.__buffer):
            return

        self.__buffer.append(data)

        if len(self.__buffer) > 1024:
            self.__buffer.pop(0)

        return data

    def start(self) -> None:
        threading.Thread(target=self.__start__, daemon=True).start()

        while self._stopped:
            time.sleep(0.001)
        
    def stop(self) -> None:
        self._stop_signal = True
        while not self._stopped:
            time.sleep(0.001)
    
    def __start__(self) -> None:
        with sd.InputStream(**self.sounddevice_parameters) as f:
            self._stopped = False
            self.stream = f
            while not self._stop_signal:
                data, overflow = f.read(self.chunk_size)
                self.__data = data
                self.overflow = overflow

                time.sleep(0.001)
        
        """This code is only reached once the track has been stopped."""
        self._stopped = True
        self.stream = None
        self.__data = None
        self._stop_signal = False

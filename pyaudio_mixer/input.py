from typing import Union
import sounddevice as sd
import threading
import time
import numpy as np


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

    `chunk_size` : The size of each chunk returned from .read(). Defaults to 512.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.sounddevice_parameters = kwargs.get("sounddevice_parameters", {})
        self.chunk_size = kwargs.get("chunk_size", 512)
        self.callback = kwargs.get("callback")

        # Signal Variables
        self._stop_signal = False
        self._stopped = True

        # Data Variable
        self.__data = None
        self.overflow = False

        self.stream = None
        self.start()
    
    def read(self) -> Union[np.ndarray, None]:

        """
        Read the data coming from the InputStream.

        Returns
        -------
        `np.ndarray` :
            Audio data with shape of (frames (or size of chunks), channels).
        """
        
        return self.__data

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

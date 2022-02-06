import sounddevice as sd
import threading
import time


class InputTrack:

    """
    Parameters
    ----------
    `name` : str
        The name of this track.
    `sounddevice_parameters` : dict
        Key, Value pair that will be passed as parameters to sd.InputStream. Defaults to None.
    `chunk_size` : The size of each chunk returned from .read(). Defaults to 512.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.sounddevice_parameters = kwargs.get("sounddevice_parameters", {})
        self.chunk_size = kwargs.get("chunk_size", 512)

        # Signal Variables
        self._stop_signal = False
        self._stopped = True

        # Data Variable
        self.__data = None
        self.overflow = False

        self.stream = None
        self.start()

    def start(self) -> None:
        threading.Thread(target=self.__start__, daemon=True).start()

        while self._stopped:
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

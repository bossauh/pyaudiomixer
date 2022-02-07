import numpy as np
import math


class BasicFX:
    
    """
    This class is responsible for handling basic effects such as volume.
    """

    def __init__(self) -> None:
        self.effects = [self.set_volume]
    
    def set_volume(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Increase the volume of the provided audio data by a factor of `float`"""
        return np.multiply(data, pow(
            2, (math.sqrt(math.sqrt(math.sqrt(factor))) * 192 - 192) / 6), casting="unsafe")

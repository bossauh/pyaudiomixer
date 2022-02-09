import numpy as np
import math


class BasicFX:
    
    """
    This class is responsible for handling basic effects such as volume changer.
    """

    def __init__(self, dtype: str = "float32") -> None:
        self.effects = [self.set_volume]
        self.dtype = dtype
    
    def set_volume(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Increase the volume of the provided audio data by a factor of `float`"""
        return np.multiply(data, pow(
            2, (math.sqrt(math.sqrt(math.sqrt(factor))) * 192 - 192) / 6), casting="unsafe", dtype=self.dtype)

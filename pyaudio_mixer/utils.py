import math
import noisereduce

import numpy as np
from pedalboard import (Chorus, Compressor, Pedalboard, Phaser, PitchShift,
                        Reverb, NoiseGate, HighpassFilter, LowpassFilter)


class BasicFX:
    
    """
    This class is responsible for handling basic effects such as volume changer.
    The pedalboard is also utilized for more "advanced" effects such as compression, pitch shifting, etc.

    Notes
    -----
    - The more effects you have, the longer it will take to process those. So a general rule of thumb is to increase your chunk size so that more data gets processed at once.
    - The way the pedalboard library is handled here is dare I say inefficient due to how a new board is always created per effect function call. There is a future plan to optimize this. # TODO

    Parameters
    ----------
    `dtype` : str
        Data type must match that of the track holding this BasicFX instance.
    `samplerate` : int
        Same rule for `dtype`
    """

    def __init__(self, dtype: str = "float32", samplerate: int = 44100) -> None:

        self.effects = [
            self.set_volume,
            self.chorus,
            self.reverb,
            self.phaser,
            self.pitch_shift,
            self.compressor,
            self.noise_gate,
            self.noise_reduction
        ]

        self.dtype = dtype
        self.samplerate = samplerate
    
    def set_volume(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Increase the volume of the provided audio data by a factor of `float`"""
        return np.multiply(data, pow(
            2, (math.sqrt(math.sqrt(math.sqrt(factor))) * 192 - 192) / 6), casting="unsafe", dtype=self.dtype)
    
    def chorus(self, data: np.ndarray, **params) -> np.ndarray:

        board = Pedalboard([Chorus(**params)])
        data = board(data, self.samplerate)
        return data
    
    def reverb(self, data: np.ndarray, **params) -> np.ndarray:
        board = Pedalboard([Reverb(**params)])
        return board(data, self.samplerate)
    
    def phaser(self, data: np.ndarray, **params) -> np.ndarray:
        board = Pedalboard([Phaser(**params)])
        return board(data, self.samplerate)
    
    def pitch_shift(self, data: np.ndarray, **params) -> np.ndarray:
        board = Pedalboard([PitchShift(**params)])
        return board(data, self.samplerate)
    
    def compressor(self, data: np.ndarray, **params) -> np.ndarray:
        board = Pedalboard([Compressor(**params)])
        return board(data, self.samplerate)
    
    def noise_gate(self, data: np.ndarray, **params) -> np.ndarray:
        board = Pedalboard([NoiseGate(**params)])
        return board(data, self.samplerate)
    
    def noise_reduction(self, data: np.ndarray, **params) -> np.ndarray:
        data = data.transpose()
        return noisereduce.reduce_noise(data, sr=self.samplerate, **params)
    
    def highpass_filter(self, data: np.ndarray, **params) -> np.ndarray:
        board = Pedalboard([HighpassFilter(**params)])
        return board(data, self.samplerate)
    
    def lowpass_filter(self, data: np.ndarray, **params) -> np.ndarray:
        board = Pedalboard([LowpassFilter(**params)])
        return board(data, self.samplerate)

import asyncio
import numpy as np
from maglevapi.testing import Testing
from pyaudio_mixer import InputTrack


class TestInput(Testing):
    def __init__(self) -> None:
        super().__init__(save_path="./tests/results/TestInput.tresult")
    
    async def test_input(self) -> None:
        t = InputTrack("track")
        await asyncio.sleep(0.5)

        assert t.name == "track"
        assert t.read() is not None
        await asyncio.sleep(.2)
        assert t.read().shape[0] == 512
        assert not t._stopped
        assert t.stream

        await asyncio.sleep(.5)
        t.stop()
        assert t.read() is None
        assert t._stopped
        assert t.stream is None
    
    async def test_input_parameters(self) -> None:

        sounddevice_parameters = {
            "samplerate": 16000,
            "channels": 1,
            "dtype": "int16"
        }
        
        called = False
        def callback(track: InputTrack, data: np.ndarray, overflow: bool) -> np.ndarray:
            nonlocal called
            if data is None:
                return

            assert data.shape[0] == track.chunk_size
            assert track.stream.samplerate == sounddevice_parameters["samplerate"]
            assert track.stream.channels == sounddevice_parameters["channels"]
            assert track.stream.dtype == sounddevice_parameters["dtype"]
            assert track.volume == 0.5

            called = True
            return data

        t = InputTrack("track", callback=callback, chunk_size=1024, sounddevice_parameters=sounddevice_parameters, volume=0.5)
        await asyncio.sleep(0.5)

        for _ in range(50):
            assert t.read() is not None
            await asyncio.sleep(.1)

        assert called
        await asyncio.sleep(.5)
        t.stop()
        assert t._stopped
        assert t.read() is None

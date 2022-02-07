"""
tests.output
----------
This file is meant to be imported from within run_tests.py.
This test will test the output tracks.
"""

import asyncio
import numpy as np
from maglevapi.testing import Testing
from pyaudio_mixer import OutputTrack


class TestOutput(Testing):
    def __init__(self) -> None:
        super().__init__(save_path="./tests/results/TestOutput.tresult")
        self.conversion_path = "./tests/data/converted"
        self.test_files = [
            "./tests/data/48000khz.wav",
            "./tests/data/m4a-file-1.m4a",
            "./tests/data/surround.m4a",
            "./tests/data/long.m4a",
            "./tests/data/8000khz.wav"
        ]

    async def test_output_track_basic(self) -> None:

        """Test the basic functionalities"""
        
        t = OutputTrack("track")
        assert not t._stopped
        assert not t._stop_signal
        assert t.stream is not None
        await t.stop()
        assert t._stopped
        assert not t._stop_signal
    
    async def test_output_track_parameters(self) -> None:
        """Test the parameters of the output track."""

        params = {
            "name": "Test",
            "sounddevice_parameters": {
                "dtype": "int16",
                "samplerate": 16000
            },
            "conversion_path": self.conversion_path,
            "apply_basic_fx": False,
            "volume": 0.5
        }
        t = OutputTrack(**params)
        
        assert t.name == params["name"]
        assert t.conversion_path == params["conversion_path"]
        assert t.apply_basic_fx == params["apply_basic_fx"]
        assert t.volume == params["volume"]
        assert t.stream.dtype == params["sounddevice_parameters"]["dtype"]
        assert t.stream.samplerate == params["sounddevice_parameters"]["samplerate"]
        assert not t.playing_details

        await t.stop()
        assert t._stopped
    
    async def test_output_track_on_off(self) -> None:
        t = OutputTrack("tack")
        assert not t._stopped
        assert not t._stop_signal

        for _ in range(8):
            await t.stop()
            assert t._stopped
            assert not t._stop_signal
            t.start()
            assert not t._stopped
            assert not t._stop_signal
        
        await t.stop()
    
    async def test_output_play_file(self) -> None:
        volume = 0.6
        t = OutputTrack("track", conversion_path=self.conversion_path, volume=volume)
        assert t.volume == volume

        # Test in memory, This will play all tests files for 3 seconds, one after the other.
        for f in self.test_files:
            await t.play_file(f, blocking=False, resample=True, load_in_memory=True)
            assert t._playing
            assert t.playing_details["samplerate"] == t.stream.samplerate
            await asyncio.sleep(3)
        await t.abort()
        assert not t.playing_details
        assert not t._playing

        t.volume = 0.8
        assert t.volume == 0.8

        # Test no resampling
        await t.play_file(self.test_files[0], blocking=False, resample=False, load_in_memory=False)
        assert t.playing_details["samplerate"] == 48000
        await asyncio.sleep(3)
        await t.abort()

        # Test not in memory
        for f in self.test_files:
            await t.play_file(f, blocking=False, resample=True, load_in_memory=False)
            assert t._playing
            assert t.playing_details["samplerate"] == t.stream.samplerate
            await asyncio.sleep(3)
        await t.stop()
        assert not t.playing_details
        assert not t._playing
    
    async def test_output_spam(self) -> None:
        t = OutputTrack("track")

        for _ in range(12):
            await t.play_file(self.test_files[4])
            assert t._playing
            assert t.playing_details

        await t.play_file(self.test_files[0], load_in_memory=False)
        assert t._playing
        assert t.playing_details

        await asyncio.sleep(3)
        for _ in range(8):
            await t.abort()
            assert not t._playing
            assert not t.playing_details
        
        for _ in range(12):
            await t.stop()
            assert t._stopped
    
    async def test_output_callback(self) -> None:
        
        called = False
        def callback(track: OutputTrack, data: np.ndarray):
            nonlocal called
            assert data is None and track.name == "track"
            called = True

        t = OutputTrack("track", callback=callback)
        await asyncio.sleep(1)
        await t.stop()
        assert t._stopped
        assert called

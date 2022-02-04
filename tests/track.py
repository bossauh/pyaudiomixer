"""
tests.tack
----------
This file is meant to be imported from within run_tests.py
"""

import numpy as np
import asyncio
import soundfile as sf
from maglevapi.testing import Testing
from maglevapi.profiling import timeit
from pyaudio_mixer import OutputTrack


class TestTrack(Testing):
    def __init__(self) -> None:
        super().__init__(save_path="./tests/results/TestTrack.tresult")
        self.conversion_path = "./tests/data/converted"
        self.test_files = [
            "./tests/data/48000khz.wav",
            "./tests/data/m4a-file-1.m4a",
            "./tests/data/surround.m4a",
            "./tests/data/long.m4a"
        ]

    async def test_output_track(self) -> None:
        track = OutputTrack("Track 0")
        await asyncio.sleep(1)

        assert track._stopped == False
        await track.stop()
        assert track._stopped == True

    async def test_output_track_callback(self) -> None:

        called = False
        def callback(track: OutputTrack, data: np.ndarray) -> np.ndarray:
            nonlocal called
            called = True
            return data

        track = OutputTrack("Track 0", callback=callback)
        await asyncio.sleep(1)
        assert called == True
        await track.stop()

    async def test_output_track_stop_start(self) -> None:
        track = OutputTrack("Track 0")

        for _ in range(4):
            await track.stop()
            assert track._stopped == True
            await asyncio.sleep(0.5)
            track.start()
            assert track._stopped == False
            await asyncio.sleep(0.5)

        await track.stop()

    async def test_output_track_sounddevice_parameters(self) -> None:
        sounddevice_parameters = {
            "samplerate": 16000,
            "dtype": "int16",
        }
        track = OutputTrack(
            "Track 0", sounddevice_parameters=sounddevice_parameters)

        assert track.stream.samplerate == sounddevice_parameters["samplerate"]
        assert track.stream.dtype == sounddevice_parameters["dtype"]
        await track.stop()

    async def test_output_play_abort(self) -> None:
        
        track = OutputTrack("Track", conversion_path=self.conversion_path)

        await track.play_file(self.test_files[3], blocking=False)
        assert track._playing == True

        await asyncio.sleep(5)
        await track.abort()
        assert track._playing == False
    
    async def test_output_play_multiple(self) -> None:
        track = OutputTrack("Track", conversion_path=self.conversion_path)

        await track.play_file(self.test_files[0])
        assert track._playing == True

        await asyncio.sleep(5)
        await track.play_file(self.test_files[1])
        assert track._playing == True

        await asyncio.sleep(5)
        await track.play_file(self.test_files[3])
        assert track._playing == True

        await asyncio.sleep(5)
        await track.abort()
        assert track._playing == False

    async def test_output_play_spam(self) -> None:
        track = OutputTrack("Track", conversion_path=self.conversion_path)

        await track.play_file(self.test_files[0])
        assert track._playing == True
        print("1")

        await track.play_file(self.test_files[0])
        assert track._playing == True
        print("2")

        await track.play_file(self.test_files[0])
        assert track._playing == True
        print("3")

        await track.play_file(self.test_files[0])
        assert track._playing == True
        print("4")

        await asyncio.sleep(5)

        await track.play_file(self.test_files[1])
        assert track._playing == True
        print("1")

        await track.play_file(self.test_files[1])
        assert track._playing == True
        print("2")

        await track.play_file(self.test_files[1])
        assert track._playing == True
        print("3")

        await track.play_file(self.test_files[1])
        assert track._playing == True
        print("4")

        await asyncio.sleep(5)
        await track.abort()
    
    async def test_output_volume(self) -> None:
        track = OutputTrack("Track", volume=0.5, conversion_path=self.conversion_path)

        await track.play_file(self.test_files[3], blocking=False)
        assert track.volume == 0.5

        await asyncio.sleep(5)
        track.volume = 0.35
        await asyncio.sleep(5)
        track.volume = 1.0
        await asyncio.sleep(5)
        await track.stop()
        await asyncio.sleep(10)

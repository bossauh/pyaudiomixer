import asyncio
from maglevapi.testing import Testing
from pyaudio_mixer import Mixer, OutputTrack, InputTrack


class TestMixer(Testing):
    def __init__(self) -> None:
        super().__init__(save_path="./tests/results/TestMixer.tresult")
        self.conversion_path = "./tests/data/converted"
        self.test_files = [
            "./tests/data/48000khz.wav",
            "./tests/data/m4a-file-1.m4a",
            "./tests/data/surround.m4a",
            "./tests/data/long.m4a",
            "./tests/data/8000khz.wav"
        ]

    async def test_basic(self) -> None:

        o1 = OutputTrack("o1", conversion_path=self.conversion_path)
        o2 = OutputTrack("o2", conversion_path=self.conversion_path)
        i1 = InputTrack("i1")
        i2 = InputTrack("i2")

        mixer = Mixer([o1, o2, i1, i2])
        assert len(mixer.tracks) == 4
        assert len(mixer.input_tracks) == 2
        assert len(mixer.output_tracks) == 2
        assert len(mixer.available_output_tracks) == 2
        
        for i, f in enumerate(self.test_files):
            t = await mixer.play_file(f)
            if i > 1:
                assert t is None
            else:
                assert t.name in ["o1", "o2"]
                assert t._playing
        
        await asyncio.sleep(5)

        await mixer.abort_outputs()
        for t in mixer.tracks:
            assert not t._stopped
        
        await mixer.stop_inputs()
        await mixer.stop_outputs()
        for t in mixer.tracks:
            assert t._stopped

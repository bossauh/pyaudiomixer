from typing import Union, List
from .output import OutputTrack
from .input import InputTrack


class Mixer:
    """
    This class will be responsible for containing all your tracks.
    You usually don't need to use this Mixer class unless you want to keep things organized.

    Notes
    -----
    - The name format of the generated tracks (if there is any) is "track {idx}" with idx being their index on the list of either input tracks or output tracks.

    Parameters
    ----------
    `tracks` : Union[List[OutputTrack], List[InputTrack]]
        List of tracks. You can set this to None and pass a `> 0` number to either the `generate_out` or `generate_in` to generate the tracks instead.
    `generate_out` : int
        Number of output tracks to generate if wanted. Defaults to 0.
    `generate_in` : int
        Number of input tracks to generate if wanted. Defaults to 0. 
    """

    def __init__(
        self,
        tracks: Union[List[OutputTrack], List[InputTrack]] = None,
        generate_out: int = 0,
        generate_in: int = 0
    ) -> None:
        self.tracks = tracks
        
        # Generate tracks
        self.generate_tracks(generate_out, type_="out")
        self.generate_tracks(generate_in, type_="in")

    @property
    def input_tracks(self) -> List[InputTrack]:
        """Returns a list of all input tracks (List will be empty if there isn't any)"""
        return [x for x in self.tracks if isinstance(x, InputTrack)]

    @property
    def output_tracks(self) -> List[OutputTrack]:
        """Returns a list of all output tracks (List will be empty if there isn't any)"""
        return [x for x in self.tracks if isinstance(x, OutputTrack)]

    @property
    def available_output_tracks(self) -> List[OutputTrack]:
        """
        Returns a list of output tracks wherein their playing status is False (i.e, nothing is being played in them therefore they are free to occupy).

        There isn't a input track version of this because the input tracks are always available.
        """
        return [x for x in self.tracks if isinstance(x, OutputTrack) and not x._playing]

    def generate_tracks(self, n: int, type_: str = "out") -> Union[List[OutputTrack], List[InputTrack]]:

        """
        Generate a "n" amount of input ("in") or output ("out", default) tracks.

        Notes
        -----
        - The name format of the generated tracks (if there is any) is "track {idx}" with idx being their index on the list of either input tracks or output tracks.
        - This modifies the self.tracks list.

        Parameters
        ----------
        `n` : int
            Number of tracks to generate.
        `type_` : str
            Type of tracks to generate. Defaults to "out". "out" = OutputTrack, "in" = InputTrack

        Returns
        -------
        `Union[List[OutputTrack], List[InputTrack]]` :
            List of the generated tracks.
        """

        if n < 1:
            return []

        mappings = {
            "out": OutputTrack,
            "in": InputTrack
        }

        for _ in range(n):
            
            if type_ == "out":
                idx = len(self.output_tracks)
            elif type_ == "in":
                idx = len(self.input_tracks)
            
            idx = (idx - 1) if idx > 0 else 0
            track = mappings[type_](f"track {idx}")
            self.tracks.append(track)
    
    def get_output_track(self, name: str) -> Union[OutputTrack, None]:
        track = [x for x in self.output_tracks if x.name == name]
        if track:
            return track[0]
    
    def get_input_track(self, name: str) -> Union[InputTrack, None]:
        track = [x for x in self.input_tracks if x.name == name]
        if track:
            return track[0]
    
    async def abort_outputs(self) -> None:
        for track in self.output_tracks:
            await track.abort()
    
    async def stop_outputs(self) -> None:
        for track in self.output_tracks:
            await track.stop()
        
    async def stop_inputs(self) -> None:
        for track in self.input_tracks:
            track.stop()

    async def play_file(self, *args, **kwargs) -> Union[OutputTrack, None]:
        """
        Play the provided audio file using one of the free output tracks.

        Returns
        -------
        `OutputTrack` :
            The track the provided file is being played on.
        `None` :
            If no free track was found therefore the provided audio file wasn't played.
        """

        track = self.available_output_tracks
        if not track:
            return
        track = track[0]
        await track.play_file(*args, **kwargs)
        return track

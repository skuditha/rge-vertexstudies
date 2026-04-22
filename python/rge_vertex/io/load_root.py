from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import awkward as ak
import uproot

DEFAULT_TRACK_BRANCHES = [
    "run", "file_index", "event_index", "global_event_id", "particle_index",
    "pid", "charge", "status", "sector", "rec_track_detector", "chi2pid",
    "px", "py", "pz", "p", "theta", "phi",
    "vx_particle", "vy_particle", "vz_particle",
    "has_ftrack", "vx_ftrack", "vy_ftrack", "vz_ftrack",
    "ftrack_sector", "ftrack_chi2", "ftrack_ndf", "ftrack_chi2_ndf",
    "detector_region",
]


def open_tracks_tree(root_path: str | Path):
    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")
    root_file = uproot.open(root_path)
    if "tracks" not in root_file:
        raise KeyError(f"'tracks' tree not found in {root_path}")
    return root_file["tracks"]


def iterate_tracks(root_path: str | Path, branches: Optional[Iterable[str]] = None, step_size: str | int = "100 MB"):
    tree = open_tracks_tree(root_path)
    if branches is None:
        branches = DEFAULT_TRACK_BRANCHES
    for arrays in tree.iterate(branches, step_size=step_size, library="ak"):
        yield arrays


def get_available_branches(root_path: str | Path) -> list[str]:
    return list(open_tracks_tree(root_path).keys())


def to_numpy(array):
    return ak.to_numpy(array)

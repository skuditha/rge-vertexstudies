from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rge_vertex.io.load_root import iterate_tracks
from rge_vertex.selections.tracks import combined_track_mask


@dataclass
class HistogramResult:
    counts: np.ndarray
    edges: np.ndarray
    entries: int

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])


def collect_vz_histogram(root_path: str | Path, *, vertex_source: str, charge, detector_region, sector, pid=None, chi2pid_abs_max=None, bins=240, vz_range=(-12.0, 4.0), step_size="100 MB") -> HistogramResult:
    branch = "vz_particle" if vertex_source == "particle" else "vz_ftrack"
    needed = [branch, "has_ftrack", "charge", "detector_region", "sector", "pid", "chi2pid"]
    total_counts = np.zeros(bins, dtype=np.float64)
    edges = np.linspace(vz_range[0], vz_range[1], bins + 1)
    entries = 0
    for arrays in iterate_tracks(root_path, branches=needed, step_size=step_size):
        mask = combined_track_mask(arrays, vertex_source=vertex_source, charge=charge, detector_region=detector_region, sector=sector, pid=pid, chi2pid_abs_max=chi2pid_abs_max)
        values = np.asarray(arrays[branch])[mask]
        values = values[np.isfinite(values)]
        values = values[(values >= vz_range[0]) & (values <= vz_range[1])]
        counts, _ = np.histogram(values, bins=edges)
        total_counts += counts
        entries += int(values.size)
    return HistogramResult(counts=total_counts, edges=edges, entries=entries)


def save_histogram_csv(result: HistogramResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"bin_low": result.edges[:-1], "bin_high": result.edges[1:], "bin_center": result.centers, "count": result.counts})
    df.to_csv(output_path, index=False)


def plot_particle_vs_ftrack_overlay(particle_hist: HistogramResult, ftrack_hist: HistogramResult, *, title: str, output_path: str | Path, normalize: bool = False) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    particle_counts = particle_hist.counts.astype(float)
    ftrack_counts = ftrack_hist.counts.astype(float)
    ylabel = "counts"
    if normalize:
        if particle_counts.sum() > 0:
            particle_counts = particle_counts / particle_counts.sum()
        if ftrack_counts.sum() > 0:
            ftrack_counts = ftrack_counts / ftrack_counts.sum()
        ylabel = "normalized counts"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.step(particle_hist.centers, particle_counts, where="mid", label=f"REC::Particle.vz, N={particle_hist.entries}")
    ax.step(ftrack_hist.centers, ftrack_counts, where="mid", label=f"REC::FTrack.vz, N={ftrack_hist.entries}")
    ax.set_title(title)
    ax.set_xlabel("vz [cm]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_histogram_summary(rows: Iterable[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rge_vertex.io.load_root import iterate_tracks
from rge_vertex.selections.tracks import combined_track_mask, get_vz_values


@dataclass
class HistogramResult:
    counts: np.ndarray
    edges: np.ndarray
    entries: int

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])


def collect_vz_histogram(
    root_path: str | Path,
    *,
    vertex_source: str,
    charge: str | int | None,
    detector_region: str | int | None,
    sector: int | str | None,
    pid: int | None = None,
    chi2pid_abs_max: float | None = None,
    bins: int = 240,
    vz_range: tuple[float, float] = (-12.0, 4.0),
    step_size: str | int = "100 MB",
) -> HistogramResult:
    """Collect a vz histogram for particle, ftrack, or hybrid vertex source.

    hybrid means:
      use REC::FTrack.vz when available and finite,
      otherwise use REC::Particle.vz.
    """
    needed_branches = [
        "vz_particle",
        "vz_ftrack",
        "has_ftrack",
        "charge",
        "detector_region",
        "sector",
        "pid",
        "chi2pid",
    ]

    total_counts = np.zeros(bins, dtype=np.float64)
    edges = np.linspace(vz_range[0], vz_range[1], bins + 1)
    entries = 0

    for arrays in iterate_tracks(root_path, branches=needed_branches, step_size=step_size):
        mask = combined_track_mask(
            arrays,
            vertex_source=vertex_source,
            charge=charge,
            detector_region=detector_region,
            sector=sector,
            pid=pid,
            chi2pid_abs_max=chi2pid_abs_max,
        )

        values = get_vz_values(arrays, vertex_source)[mask]
        values = values[np.isfinite(values)]
        values = values[(values >= vz_range[0]) & (values <= vz_range[1])]

        counts, _ = np.histogram(values, bins=edges)
        total_counts += counts
        entries += int(values.size)

    return HistogramResult(counts=total_counts, edges=edges, entries=entries)


def save_histogram_csv(result: HistogramResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "bin_low": result.edges[:-1],
            "bin_high": result.edges[1:],
            "bin_center": result.centers,
            "count": result.counts,
        }
    )
    df.to_csv(output_path, index=False)


def plot_vertex_source_overlay(
    histograms: dict[str, HistogramResult],
    *,
    title: str,
    output_path: str | Path,
    xlabel: str = "vz [cm]",
    ylabel: str = "counts",
    normalize: bool = False,
) -> None:
    """Plot one or more vertex-source histograms on the same axes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for source, hist in histograms.items():
        counts = hist.counts.astype(float)
        if normalize:
            if counts.sum() > 0:
                counts = counts / counts.sum()
            ylabel = "normalized counts"

        label = {
            "particle": "REC::Particle.vz",
            "ftrack": "REC::FTrack.vz only",
            "hybrid": "Hybrid: FTrack if available, else Particle",
        }.get(source, source)

        ax.step(
            hist.centers,
            counts,
            where="mid",
            label=f"{label}, N={hist.entries}",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_particle_vs_ftrack_overlay(
    particle_hist: HistogramResult,
    ftrack_hist: HistogramResult,
    *,
    title: str,
    output_path: str | Path,
    xlabel: str = "vz [cm]",
    ylabel: str = "counts",
    normalize: bool = False,
) -> None:
    """Backward-compatible two-source overlay."""
    plot_vertex_source_overlay(
        {"particle": particle_hist, "ftrack": ftrack_hist},
        title=title,
        output_path=output_path,
        xlabel=xlabel,
        ylabel=ylabel,
        normalize=normalize,
    )


def write_histogram_summary(rows: Iterable[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False)

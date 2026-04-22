from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from rge_vertex.io.configs import load_runs, repo_path
from rge_vertex.plotting.histograms import (
    HistogramResult,
    collect_vz_histogram,
    plot_vertex_source_overlay,
    save_histogram_csv,
)


def normalize_run_number(run: str | int) -> str:
    return str(run).zfill(6)


def _value_allowed(value: Any, allowed: list[Any] | None) -> bool:
    if allowed is None:
        return True
    return value in allowed


def iter_filtered_runs(
    runs_config: str | Path,
    *,
    run_class: str = "ld2_solid",
    polarities: list[str] | None = None,
    solid_targets: list[str] | None = None,
    enabled_only: bool = True,
    run: str | None = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Iterate over runs matching the production-QA filters."""
    runs = load_runs(runs_config)

    if run is not None:
        run = normalize_run_number(run)
        if run not in runs:
            raise KeyError(f"Run {run} not found in {runs_config}")
        meta = runs[run]
        if enabled_only and not bool(meta.get("enabled", True)):
            return
        yield run, meta
        return

    for run_number, meta in sorted(runs.items()):
        run_number = normalize_run_number(run_number)

        if enabled_only and not bool(meta.get("enabled", True)):
            continue

        if run_class is not None and meta.get("run_class") != run_class:
            continue

        if not _value_allowed(meta.get("polarity"), polarities):
            continue

        if not _value_allowed(meta.get("solid_target"), solid_targets):
            continue

        yield run_number, meta


def sectors_for_detector_region(detector_region: str, forward_sectors: list[Any], central_sectors: list[Any]) -> list[Any]:
    if detector_region == "central":
        return central_sectors
    if detector_region == "forward":
        return forward_sectors
    return ["all"]


def sector_label_for_path(detector_region: str, sector: int | str) -> str:
    if detector_region == "central":
        return "central_all"
    return f"sector_{sector}"


def sector_label_for_title(detector_region: str, sector: int | str) -> str:
    if detector_region == "central":
        return "central detector, no sector split"
    return f"sector={sector}"


def counts_in_window(hist: HistogramResult, window: tuple[float, float]) -> float:
    lo, hi = window
    mask = (hist.centers >= lo) & (hist.centers <= hi)
    return float(np.sum(hist.counts[mask]))


def peak_in_window(hist: HistogramResult, window: tuple[float, float]) -> tuple[float, float]:
    """Return peak center/count within a rough QA window."""
    lo, hi = window
    mask = (hist.centers >= lo) & (hist.centers <= hi)
    if not np.any(mask):
        return float("nan"), float("nan")

    centers = hist.centers[mask]
    counts = hist.counts[mask]
    if counts.size == 0 or np.max(counts) <= 0:
        return float("nan"), float("nan")

    idx = int(np.argmax(counts))
    return float(centers[idx]), float(counts[idx])


def histogram_qa_metrics(
    hist: HistogramResult,
    *,
    ld2_window: tuple[float, float],
    solid_window: tuple[float, float],
    reference_region: tuple[float, float],
) -> dict[str, float]:
    ld2_peak_center, ld2_peak_count = peak_in_window(hist, ld2_window)
    solid_peak_center, solid_peak_count = peak_in_window(hist, solid_window)
    ref_peak_center, ref_peak_count = peak_in_window(hist, reference_region)

    return {
        "entries": hist.entries,
        "ld2_window_counts": counts_in_window(hist, ld2_window),
        "solid_window_counts": counts_in_window(hist, solid_window),
        "reference_region_counts": counts_in_window(hist, reference_region),
        "ld2_peak_center": ld2_peak_center,
        "ld2_peak_count": ld2_peak_count,
        "solid_peak_center": solid_peak_center,
        "solid_peak_count": solid_peak_count,
        "reference_peak_center": ref_peak_center,
        "reference_peak_count": ref_peak_count,
    }


def make_production_qa_for_run(
    *,
    run: str,
    meta: dict[str, Any],
    repo_root: str | Path,
    qa_config: dict[str, Any],
) -> list[dict[str, Any]]:
    repo_root = Path(repo_root)

    hist_cfg = qa_config["histogram"]
    category_cfg = qa_config["categories"]
    quality_cfg = qa_config.get("quality", {})
    window_cfg = qa_config.get("qa_windows", {})
    output_cfg = qa_config.get("outputs", {})

    root_path = repo_path(meta.get("output_root", f"outputs/ntuples/{run}.root"), repo_root)
    if not root_path.exists():
        raise FileNotFoundError(f"ROOT ntuple for run {run} not found: {root_path}")

    plots_dir = repo_path(output_cfg.get("plots_dir", "outputs/plots"), repo_root)

    vertex_sources = category_cfg.get("vertex_sources", ["particle", "ftrack", "hybrid"])
    charges = category_cfg.get("charges", ["negative", "positive"])
    detector_regions = category_cfg.get("detector_regions", ["forward", "central"])
    forward_sectors = category_cfg.get("forward_sectors", ["all", 1, 2, 3, 4, 5, 6])
    central_sectors = category_cfg.get("central_sectors", ["all"])

    bins = int(hist_cfg.get("bins", 240))
    vz_range = (float(hist_cfg.get("vz_min_cm", -12.0)), float(hist_cfg.get("vz_max_cm", 4.0)))
    step_size = hist_cfg.get("step_size", "100 MB")
    normalize_overlay = bool(hist_cfg.get("normalize_overlay", False))

    chi2pid_abs_max = quality_cfg.get("chi2pid_abs_max")

    ld2_window = tuple(window_cfg.get("ld2_window_cm", [-7.5, -4.5]))
    solid_window = tuple(window_cfg.get("solid_window_cm", [-1.6, -0.4]))
    reference_region = tuple(window_cfg.get("reference_region_cm", [-4.3, -2.7]))

    rows: list[dict[str, Any]] = []

    for charge in charges:
        for detector_region in detector_regions:
            sectors = sectors_for_detector_region(detector_region, forward_sectors, central_sectors)

            for sector in sectors:
                histograms: dict[str, HistogramResult] = {}

                for vertex_source in vertex_sources:
                    histograms[vertex_source] = collect_vz_histogram(
                        root_path,
                        vertex_source=vertex_source,
                        charge=charge,
                        detector_region=detector_region,
                        sector=sector,
                        chi2pid_abs_max=chi2pid_abs_max,
                        bins=bins,
                        vz_range=vz_range,
                        step_size=step_size,
                    )

                sector_path = sector_label_for_path(detector_region, sector)
                sector_title = sector_label_for_title(detector_region, sector)

                rel_dir = (
                    Path(str(run))
                    / "production_qa"
                    / str(meta.get("polarity", "unknown_polarity"))
                    / str(meta.get("solid_target", "unknown_target"))
                    / detector_region
                    / charge
                )

                plot_path = plots_dir / rel_dir / f"{sector_path}_vertex_sources.png"

                title = (
                    f"Run {run}: {meta.get('polarity', '')}, LD2+{meta.get('solid_target', '')}\n"
                    f"{charge}, {detector_region}, {sector_title}"
                )

                plot_vertex_source_overlay(
                    histograms,
                    title=title,
                    output_path=plot_path,
                    normalize=normalize_overlay,
                )

                hist_csv_paths: dict[str, Path] = {}
                for vertex_source, hist in histograms.items():
                    csv_path = plots_dir / rel_dir / f"{sector_path}_{vertex_source}_hist.csv"
                    save_histogram_csv(hist, csv_path)
                    hist_csv_paths[vertex_source] = csv_path

                    metrics = histogram_qa_metrics(
                        hist,
                        ld2_window=ld2_window,
                        solid_window=solid_window,
                        reference_region=reference_region,
                    )

                    rows.append(
                        {
                            "run": run,
                            "label": meta.get("label", ""),
                            "run_class": meta.get("run_class", ""),
                            "polarity": meta.get("polarity", ""),
                            "solid_target": meta.get("solid_target", ""),
                            "target_config": meta.get("target_config", ""),
                            "vertex_source": vertex_source,
                            "charge": charge,
                            "detector_region": detector_region,
                            "sector": sector,
                            "sector_is_applicable": detector_region == "forward",
                            "hist_csv": str(hist_csv_paths[vertex_source]),
                            "overlay_plot": str(plot_path),
                            **metrics,
                        }
                    )

    return rows


def add_cross_source_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add particle/ftrack/hybrid statistics ratios within each category."""
    df = pd.DataFrame(rows)
    if df.empty:
        return rows

    keys = ["run", "charge", "detector_region", "sector"]

    for _, group in df.groupby(keys, dropna=False):
        entries_by_source = {
            row["vertex_source"]: row["entries"]
            for _, row in group.iterrows()
        }
        particle_n = entries_by_source.get("particle", 0)
        ftrack_n = entries_by_source.get("ftrack", 0)
        hybrid_n = entries_by_source.get("hybrid", 0)

        idxs = group.index
        df.loc[idxs, "particle_entries_category"] = particle_n
        df.loc[idxs, "ftrack_entries_category"] = ftrack_n
        df.loc[idxs, "hybrid_entries_category"] = hybrid_n
        df.loc[idxs, "ftrack_fraction_vs_particle"] = ftrack_n / particle_n if particle_n else np.nan
        df.loc[idxs, "hybrid_fraction_vs_particle"] = hybrid_n / particle_n if particle_n else np.nan

    return df.to_dict(orient="records")


def save_production_qa_summary(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = add_cross_source_ratios(rows)
    pd.DataFrame(rows).to_csv(output_path, index=False)

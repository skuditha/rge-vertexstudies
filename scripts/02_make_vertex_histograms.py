#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

from rge_vertex.io.configs import iter_runs, repo_path
from rge_vertex.plotting.histograms import (
    collect_vz_histogram,
    plot_vertex_source_overlay,
    save_histogram_csv,
    write_histogram_summary,
)


VERTEX_SOURCES = ["particle", "ftrack", "hybrid"]


def sectors_for_detector_region(detector_region: str, include_forward_sector_plots: bool) -> list[int | str]:
    """Return sector categories for a detector region.

    The forward detector is split into sectors 1-6.
    The central detector is treated as one sectorless entity.
    """
    if detector_region == "central":
        return ["all"]

    if detector_region == "forward" and include_forward_sector_plots:
        return ["all", 1, 2, 3, 4, 5, 6]

    return ["all"]


def sector_label_for_title(detector_region: str, sector: int | str) -> str:
    if detector_region == "central":
        return "central detector, no sector split"
    return f"sector={sector}"


def sector_label_for_path(detector_region: str, sector: int | str) -> str:
    if detector_region == "central":
        return "central_all"
    return f"sector_{sector}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make vertex-source comparison histograms: Particle, FTrack-only, and Hybrid."
    )
    parser.add_argument("--runs-config", default="configs/runs.yaml")
    parser.add_argument("--run", default=None)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--summary-csv", default="outputs/tables/vertex_histogram_summary.csv")
    parser.add_argument("--bins", type=int, default=240)
    parser.add_argument("--vz-min", type=float, default=-12.0)
    parser.add_argument("--vz-max", type=float, default=4.0)
    parser.add_argument("--step-size", default="100 MB")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--no-sector-plots",
        action="store_true",
        help="Disable sector 1-6 plots for the forward detector. Central is never sector-split.",
    )
    parser.add_argument("--chi2pid-abs-max", type=float, default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_base = repo_path(args.output_dir, repo_root)
    summary_rows = []

    charges = ["negative", "positive"]
    detector_regions = ["forward", "central"]
    include_forward_sector_plots = not args.no_sector_plots

    for run, meta in iter_runs(args.runs_config, run=args.run, enabled_only=True):
        root_path = repo_path(meta.get("output_root", f"outputs/ntuples/{run}.root"), repo_root)
        if not root_path.exists():
            raise FileNotFoundError(f"ROOT ntuple for run {run} not found: {root_path}")

        for charge in charges:
            for detector_region in detector_regions:
                for sector in sectors_for_detector_region(detector_region, include_forward_sector_plots):
                    histograms = {}

                    for vertex_source in VERTEX_SOURCES:
                        histograms[vertex_source] = collect_vz_histogram(
                            root_path,
                            vertex_source=vertex_source,
                            charge=charge,
                            detector_region=detector_region,
                            sector=sector,
                            chi2pid_abs_max=args.chi2pid_abs_max,
                            bins=args.bins,
                            vz_range=(args.vz_min, args.vz_max),
                            step_size=args.step_size,
                        )

                    sector_path_label = sector_label_for_path(detector_region, sector)
                    sector_title_label = sector_label_for_title(detector_region, sector)

                    rel_dir = Path(str(run)) / "vertex_histograms" / detector_region / charge
                    plot_path = output_base / rel_dir / f"{sector_path_label}_vertex_sources.png"

                    title = (
                        f"Run {run}: {charge}, {detector_region}, {sector_title_label}\n"
                        "Particle vs FTrack-only vs Hybrid vertex source"
                    )

                    plot_vertex_source_overlay(
                        histograms,
                        title=title,
                        output_path=plot_path,
                        normalize=args.normalize,
                    )

                    hist_csv_paths = {}
                    for vertex_source, hist in histograms.items():
                        csv_path = output_base / rel_dir / f"{sector_path_label}_{vertex_source}_hist.csv"
                        save_histogram_csv(hist, csv_path)
                        hist_csv_paths[vertex_source] = csv_path

                    particle_n = histograms["particle"].entries
                    ftrack_n = histograms["ftrack"].entries
                    hybrid_n = histograms["hybrid"].entries

                    summary_rows.append(
                        {
                            "run": run,
                            "label": meta.get("label", ""),
                            "run_class": meta.get("run_class", ""),
                            "solid_target": meta.get("solid_target", ""),
                            "charge": charge,
                            "detector_region": detector_region,
                            "sector": sector,
                            "sector_is_applicable": detector_region == "forward",
                            "particle_entries": particle_n,
                            "ftrack_entries": ftrack_n,
                            "hybrid_entries": hybrid_n,
                            "ftrack_fraction_vs_particle": ftrack_n / particle_n if particle_n > 0 else None,
                            "hybrid_fraction_vs_particle": hybrid_n / particle_n if particle_n > 0 else None,
                            "plot_path": str(plot_path),
                            "particle_hist_csv": str(hist_csv_paths["particle"]),
                            "ftrack_hist_csv": str(hist_csv_paths["ftrack"]),
                            "hybrid_hist_csv": str(hist_csv_paths["hybrid"]),
                        }
                    )

                    print(
                        f"{run} {charge:8s} {detector_region:7s} {sector_title_label}: "
                        f"Particle N={particle_n}, FTrack N={ftrack_n}, Hybrid N={hybrid_n}"
                    )

    write_histogram_summary(summary_rows, repo_path(args.summary_csv, repo_root))
    print(f"Wrote summary: {repo_path(args.summary_csv, repo_root)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from rge_vertex.io.configs import iter_runs, repo_path
from rge_vertex.plotting.histograms import collect_vz_histogram, plot_particle_vs_ftrack_overlay, save_histogram_csv, write_histogram_summary


def sector_values(include_sectors: bool) -> list[int | str]:
    return ["all", 1, 2, 3, 4, 5, 6] if include_sectors else ["all"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Make REC::Particle.vz vs REC::FTrack.vz comparison histograms.")
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
    parser.add_argument("--no-sector-plots", action="store_true")
    parser.add_argument("--chi2pid-abs-max", type=float, default=None)
    args = parser.parse_args()
    repo_root = Path(args.repo_root)
    output_base = repo_path(args.output_dir, repo_root)
    summary_rows = []
    charges = ["negative", "positive"]
    detector_regions = ["forward", "central"]
    sectors = sector_values(include_sectors=not args.no_sector_plots)
    for run, meta in iter_runs(args.runs_config, run=args.run, enabled_only=True):
        root_path = repo_path(meta.get("output_root", f"outputs/ntuples/{run}.root"), repo_root)
        if not root_path.exists():
            raise FileNotFoundError(f"ROOT ntuple for run {run} not found: {root_path}")
        for charge in charges:
            for detector_region in detector_regions:
                for sector in sectors:
                    particle_hist = collect_vz_histogram(root_path, vertex_source="particle", charge=charge, detector_region=detector_region, sector=sector, chi2pid_abs_max=args.chi2pid_abs_max, bins=args.bins, vz_range=(args.vz_min, args.vz_max), step_size=args.step_size)
                    ftrack_hist = collect_vz_histogram(root_path, vertex_source="ftrack", charge=charge, detector_region=detector_region, sector=sector, chi2pid_abs_max=args.chi2pid_abs_max, bins=args.bins, vz_range=(args.vz_min, args.vz_max), step_size=args.step_size)
                    sector_label = f"sector_{sector}"
                    rel_dir = Path(str(run)) / "vertex_histograms" / detector_region / charge
                    plot_path = output_base / rel_dir / f"{sector_label}_particle_vs_ftrack.png"
                    csv_particle = output_base / rel_dir / f"{sector_label}_particle_hist.csv"
                    csv_ftrack = output_base / rel_dir / f"{sector_label}_ftrack_hist.csv"
                    title = f"Run {run}: {charge}, {detector_region}, sector={sector}\nREC::Particle.vz vs REC::FTrack.vz"
                    plot_particle_vs_ftrack_overlay(particle_hist, ftrack_hist, title=title, output_path=plot_path, normalize=args.normalize)
                    save_histogram_csv(particle_hist, csv_particle)
                    save_histogram_csv(ftrack_hist, csv_ftrack)
                    summary_rows.append({
                        "run": run,
                        "label": meta.get("label", ""),
                        "run_class": meta.get("run_class", ""),
                        "solid_target": meta.get("solid_target", ""),
                        "charge": charge,
                        "detector_region": detector_region,
                        "sector": sector,
                        "particle_entries": particle_hist.entries,
                        "ftrack_entries": ftrack_hist.entries,
                        "ftrack_fraction_vs_particle": ftrack_hist.entries / particle_hist.entries if particle_hist.entries > 0 else None,
                        "plot_path": str(plot_path),
                        "particle_hist_csv": str(csv_particle),
                        "ftrack_hist_csv": str(csv_ftrack),
                    })
                    print(f"{run} {charge:8s} {detector_region:7s} sector={sector}: Particle N={particle_hist.entries}, FTrack N={ftrack_hist.entries}")
    write_histogram_summary(summary_rows, repo_path(args.summary_csv, repo_root))
    print(f"Wrote summary: {repo_path(args.summary_csv, repo_root)}")


if __name__ == "__main__":
    main()

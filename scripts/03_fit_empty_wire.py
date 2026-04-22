#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rge_vertex.fitting.fit_empty_wire import (
    EmptyWireFitOutputs,
    fit_empty_wire_category,
    save_empty_wire_outputs,
)
from rge_vertex.io.configs import load_yaml, load_runs, repo_path


def sectors_for_detector(detector_region: str, fit_config: dict) -> list[int | str]:
    categories = fit_config["categories"]
    if detector_region == "central":
        return categories.get("central_sectors", ["all"])
    if detector_region == "forward":
        return categories.get("forward_sectors", ["all", 1, 2, 3, 4, 5, 6])
    return ["all"]


def sector_label(detector_region: str, sector: int | str) -> str:
    if detector_region == "central":
        return "central_all"
    return f"sector_{sector}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit empty+wire run vertex distributions with four Gaussians.")
    parser.add_argument("--runs-config", default="configs/runs.yaml")
    parser.add_argument("--fit-config", default="configs/empty_wire_fit.yaml")
    parser.add_argument("--run", default=None, help="Run number. Defaults to empty_wire_fit.run.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--plots-dir", default="outputs/plots")
    parser.add_argument("--fit-results-csv", default="outputs/fits/empty_wire_fit_results.csv")
    parser.add_argument("--reference-csv", default="outputs/tables/reference_foil_positions.csv")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    fit_config_all = load_yaml(args.fit_config)
    fit_config = fit_config_all["empty_wire_fit"]

    run = str(args.run or fit_config["run"]).zfill(6)
    runs = load_runs(args.runs_config)

    if run not in runs:
        raise KeyError(f"Run {run} not found in {args.runs_config}")

    run_meta = runs[run]
    root_path = repo_path(run_meta.get("output_root", f"outputs/ntuples/{run}.root"), repo_root)

    if not root_path.exists():
        raise FileNotFoundError(f"Ntuple not found for run {run}: {root_path}")

    categories = fit_config["categories"]
    vertex_sources = categories.get("vertex_sources", ["particle", "ftrack"])
    charges = categories.get("charges", ["negative", "positive"])
    detector_regions = categories.get("detector_regions", ["forward", "central"])

    outputs = EmptyWireFitOutputs(fit_rows=[], reference_rows=[])

    for vertex_source in vertex_sources:
        for charge in charges:
            for detector_region in detector_regions:
                for sector in sectors_for_detector(detector_region, fit_config):
                    rel_plot = (
                        Path(str(run))
                        / "empty_wire_fits"
                        / vertex_source
                        / detector_region
                        / charge
                        / f"{sector_label(detector_region, sector)}.png"
                    )
                    output_plot = repo_path(args.plots_dir, repo_root) / rel_plot

                    fit_rows, reference_row = fit_empty_wire_category(
                        run=run,
                        root_path=root_path,
                        vertex_source=vertex_source,
                        charge=charge,
                        detector_region=detector_region,
                        sector=sector,
                        config=fit_config,
                        output_plot=output_plot,
                    )

                    outputs.fit_rows.extend(fit_rows)
                    outputs.reference_rows.append(reference_row)

                    status = "OK" if reference_row["fit_success"] else "FAIL"
                    print(
                        f"{status} {run} {vertex_source:8s} {charge:8s} "
                        f"{detector_region:7s} sector={sector}: "
                        f"ref={reference_row['reference_mean']:.4g}, "
                        f"window=[{reference_row['reference_low']:.4g}, {reference_row['reference_high']:.4g}]"
                    )

    save_empty_wire_outputs(
        outputs,
        fit_results_csv=repo_path(args.fit_results_csv, repo_root),
        reference_csv=repo_path(args.reference_csv, repo_root),
    )

    print(f"Wrote fit results: {repo_path(args.fit_results_csv, repo_root)}")
    print(f"Wrote reference foil table: {repo_path(args.reference_csv, repo_root)}")


if __name__ == "__main__":
    main()

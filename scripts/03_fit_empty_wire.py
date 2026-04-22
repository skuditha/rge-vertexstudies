#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rge_vertex.fitting.fit_empty_wire_local import (
    fit_empty_wire_local_category,
    save_empty_wire_local_outputs,
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


def should_skip_category(vertex_source: str, detector_region: str, fit_config: dict) -> bool:
    categories = fit_config["categories"]
    return (
        bool(categories.get("skip_central_ftrack", True))
        and vertex_source == "ftrack"
        and detector_region == "central"
    )


def make_skip_rows(
    *,
    run: str,
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
    fit_config: dict,
) -> list[dict]:
    rows = []
    for component, comp_cfg in fit_config["components"].items():
        rows.append(
            {
                "run": run,
                "vertex_source": vertex_source,
                "charge": charge,
                "detector_region": detector_region,
                "sector": sector,
                "component": component,
                "fit_status": "skipped_central_ftrack",
                "fit_success": False,
                "fit_valid": False,
                "fmin_valid": False,
                "entries_category": 0,
                "entries_window": 0,
                "peak_found": False,
                "peak_center": None,
                "peak_height": None,
                "peak_prominence": None,
                "nll": None,
                "deviance": None,
                "ndof": None,
                "deviance_ndof": None,
                "background_enabled": bool(fit_config.get("local_background", {}).get("enabled", True)),
                "expected_mean_cm": comp_cfg.get("expected_mean_cm"),
                "search_window_low": comp_cfg.get("search_window_cm", [None, None])[0],
                "search_window_high": comp_cfg.get("search_window_cm", [None, None])[1],
                "fit_window_low": comp_cfg.get("fit_window_cm", [None, None])[0],
                "fit_window_high": comp_cfg.get("fit_window_cm", [None, None])[1],
                "mean": None,
                "mean_error": None,
                "sigma": None,
                "sigma_error": None,
                "yield_signal": None,
                "yield_signal_error": None,
                "bkg_c0": None,
                "bkg_c0_error": None,
                "bkg_c1": None,
                "bkg_c1_error": None,
                "bkg_c2": None,
                "bkg_c2_error": None,
                "message": "Central detector is treated as sectorless and FTrack is skipped by config.",
            }
        )
    return rows


def make_reference_row_from_skip(
    *,
    run: str,
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
    fit_config: dict,
) -> dict:
    return {
        "run": run,
        "vertex_source": vertex_source,
        "charge": charge,
        "detector_region": detector_region,
        "sector": sector,
        "reference_component": fit_config["reference_foil"]["component_name"],
        "n_sigma": fit_config["reference_foil"]["n_sigma_window"],
        "fit_status": "skipped_central_ftrack",
        "reference_mean": None,
        "reference_mean_error": None,
        "reference_sigma": None,
        "reference_sigma_error": None,
        "reference_low": None,
        "reference_high": None,
        "entries_category": 0,
        "entries_window": 0,
        "message": "Central FTrack category skipped.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit empty+wire run with local Poisson peak fits."
    )
    parser.add_argument("--runs-config", default="configs/runs.yaml")
    parser.add_argument("--fit-config", default="configs/empty_wire_local_fit.yaml")
    parser.add_argument("--run", default=None, help="Run number. Defaults to empty_wire_local_fit.run.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--plots-dir", default="outputs/plots")
    parser.add_argument("--fit-results-csv", default="outputs/fits/empty_wire_local_peak_fit_results.csv")
    parser.add_argument("--reference-csv", default="outputs/tables/reference_foil_positions.csv")
    parser.add_argument("--unresolved-csv", default="outputs/tables/unresolved_empty_wire_categories.csv")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    fit_config_all = load_yaml(args.fit_config)
    fit_config = fit_config_all["empty_wire_local_fit"]

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

    all_fit_rows = []
    all_reference_rows = []
    all_unresolved_rows = []

    for vertex_source in vertex_sources:
        for charge in charges:
            for detector_region in detector_regions:
                for sector in sectors_for_detector(detector_region, fit_config):
                    if should_skip_category(vertex_source, detector_region, fit_config):
                        skip_rows = make_skip_rows(
                            run=run,
                            vertex_source=vertex_source,
                            charge=charge,
                            detector_region=detector_region,
                            sector=sector,
                            fit_config=fit_config,
                        )
                        all_fit_rows.extend(skip_rows)
                        all_unresolved_rows.extend(skip_rows)
                        all_reference_rows.append(
                            make_reference_row_from_skip(
                                run=run,
                                vertex_source=vertex_source,
                                charge=charge,
                                detector_region=detector_region,
                                sector=sector,
                                fit_config=fit_config,
                            )
                        )
                        print(f"SKIP {run} {vertex_source:8s} {charge:8s} {detector_region:7s} sector={sector}")
                        continue

                    rel_plot = (
                        Path(str(run))
                        / "empty_wire_local_fits"
                        / vertex_source
                        / detector_region
                        / charge
                        / f"{sector_label(detector_region, sector)}.png"
                    )
                    output_plot = repo_path(args.plots_dir, repo_root) / rel_plot

                    fit_rows, reference_row, unresolved_rows = fit_empty_wire_local_category(
                        run=run,
                        root_path=root_path,
                        vertex_source=vertex_source,
                        charge=charge,
                        detector_region=detector_region,
                        sector=sector,
                        config=fit_config,
                        output_plot=output_plot,
                    )

                    all_fit_rows.extend(fit_rows)
                    all_reference_rows.append(reference_row)
                    all_unresolved_rows.extend(unresolved_rows)

                    ref_status = reference_row["fit_status"]
                    ref_mean = reference_row["reference_mean"]
                    ref_low = reference_row["reference_low"]
                    ref_high = reference_row["reference_high"]

                    if pd.notna(ref_mean):
                        ref_text = f"ref={ref_mean:.4g}, window=[{ref_low:.4g}, {ref_high:.4g}]"
                    else:
                        ref_text = "ref unresolved"

                    print(
                        f"{ref_status.upper():22s} {run} {vertex_source:8s} {charge:8s} "
                        f"{detector_region:7s} sector={sector}: {ref_text}"
                    )

    save_empty_wire_local_outputs(
        fit_rows=all_fit_rows,
        reference_rows=all_reference_rows,
        unresolved_rows=all_unresolved_rows,
        fit_results_csv=repo_path(args.fit_results_csv, repo_root),
        reference_csv=repo_path(args.reference_csv, repo_root),
        unresolved_csv=repo_path(args.unresolved_csv, repo_root),
    )

    print(f"Wrote fit results: {repo_path(args.fit_results_csv, repo_root)}")
    print(f"Wrote reference foil table: {repo_path(args.reference_csv, repo_root)}")
    print(f"Wrote unresolved categories: {repo_path(args.unresolved_csv, repo_root)}")


if __name__ == "__main__":
    main()

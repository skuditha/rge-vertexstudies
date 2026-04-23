#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

from rge_vertex.fitting.ld2_solid_local import (
    fit_ld2_solid_local_category,
    save_ld2_solid_local_outputs,
)
from rge_vertex.io.configs import load_yaml, repo_path
from rge_vertex.studies.production_qa import (
    iter_filtered_runs,
    sector_label_for_path,
    sectors_for_detector_region,
)

def _parse_list_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None

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
    meta: dict,
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
                "label": meta.get("label", ""),
                "run_class": meta.get("run_class", ""),
                "polarity": meta.get("polarity", ""),
                "solid_target": meta.get("solid_target", ""),
                "target_config": meta.get("target_config", ""),
                "vertex_source": vertex_source,
                "charge": charge,
                "detector_region": detector_region,
                "sector": sector,
                "component": component,
                "signal_model": comp_cfg.get("signal_model", "gaussian"),
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
                "message": "Central detector is sectorless and FTrack-only is skipped by config.",
            }
        )
    return rows

def make_skip_category_summary(
    *,
    run: str,
    meta: dict,
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
) -> dict:
    return {
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
        "entries_category": 0,
        "ld2_fit_status": "skipped_central_ftrack",
        "solid_fit_status": "skipped_central_ftrack",
        "ld2_mean": None,
        "ld2_mean_error": None,
        "ld2_sigma": None,
        "ld2_sigma_error": None,
        "solid_mean": None,
        "solid_mean_error": None,
        "solid_sigma": None,
        "solid_sigma_error": None,
        "ld2_deviance_ndof": None,
        "solid_deviance_ndof": None,
        "mean_gap_solid_minus_ld2": None,
        "both_good": False,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit local LD2 + solid peaks for ld2_solid runs. "
                    "Parallel-safe default: writes one set of CSVs per run."
    )
    parser.add_argument("--runs-config", default="configs/runs.yaml")
    parser.add_argument("--fit-config", default="configs/ld2_solid_local_fit.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run", default=None)
    parser.add_argument("--polarity", default=None)
    parser.add_argument("--polarities", default=None)
    parser.add_argument("--solid-targets", default=None)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--per-run-fit-dir", default=None)
    parser.add_argument("--per-run-table-dir", default=None)
    parser.add_argument("--write-merged-outputs", action="store_true")
    parser.add_argument("--merged-fit-results-csv", default=None)
    parser.add_argument("--merged-unresolved-csv", default=None)
    parser.add_argument("--merged-category-summary-csv", default=None)
    parser.add_argument("--skip-if-done", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    fit_all = load_yaml(args.fit_config)
    fit_config = fit_all["ld2_solid_local_fit"]
    run_filter = fit_config.get("run_filter", {})
    run_class = run_filter.get("run_class", "ld2_solid")

    polarities = _parse_list_arg(args.polarities)
    if args.polarity is not None:
        polarities = [args.polarity]
    if polarities is None:
        polarities = run_filter.get("polarities")

    solid_targets = _parse_list_arg(args.solid_targets)
    if solid_targets is None:
        solid_targets = run_filter.get("solid_targets")

    enabled_only = bool(run_filter.get("enabled_only", True)) and not args.include_disabled

    selected_runs = list(
        iter_filtered_runs(
            args.runs_config,
            run_class=run_class,
            polarities=polarities,
            solid_targets=solid_targets,
            enabled_only=enabled_only,
            run=args.run,
        )
    )
    if not selected_runs:
        raise RuntimeError("No ld2_solid runs matched the fit filters.")

    output_cfg = fit_config.get("outputs", {})
    plots_dir = repo_path(output_cfg.get("plots_dir", "outputs/plots"), repo_root)
    per_run_fit_dir = repo_path(args.per_run_fit_dir or output_cfg.get("per_run_fit_dir", "outputs/fits/ld2_solid_per_run"), repo_root)
    per_run_table_dir = repo_path(args.per_run_table_dir or output_cfg.get("per_run_table_dir", "outputs/tables/ld2_solid_per_run"), repo_root)
    merged_fit_results_csv = repo_path(args.merged_fit_results_csv or output_cfg.get("merged_fit_results_csv", "outputs/fits/ld2_solid_local_fit_results.csv"), repo_root)
    merged_unresolved_csv = repo_path(args.merged_unresolved_csv or output_cfg.get("merged_unresolved_csv", "outputs/tables/unresolved_ld2_solid_categories.csv"), repo_root)
    merged_category_summary_csv = repo_path(args.merged_category_summary_csv or output_cfg.get("merged_category_summary_csv", "outputs/tables/ld2_solid_category_summary.csv"), repo_root)

    categories = fit_config["categories"]
    vertex_sources = categories.get("vertex_sources", ["particle", "ftrack", "hybrid"])
    charges = categories.get("charges", ["negative", "positive"])
    detector_regions = categories.get("detector_regions", ["forward", "central"])
    forward_sectors = categories.get("forward_sectors", ["all", 1, 2, 3, 4, 5, 6])
    central_sectors = categories.get("central_sectors", ["all"])

    print(f"LD2+solid local fits: processing {len(selected_runs)} run(s)")

    merged_fit_rows = []
    merged_unresolved_rows = []
    merged_category_summary_rows = []

    for run, meta in selected_runs:
        root_path = repo_path(meta.get("output_root", f"outputs/ntuples/{run}.root"), repo_root)
        if not root_path.exists():
            raise FileNotFoundError(f"Ntuple not found for run {run}: {root_path}")

        per_run_fit_results_csv = per_run_fit_dir / f"{run}_ld2_solid_local_fit_results.csv"
        per_run_unresolved_csv = per_run_table_dir / f"{run}_unresolved_ld2_solid_categories.csv"
        per_run_category_summary_csv = per_run_table_dir / f"{run}_ld2_solid_category_summary.csv"

        if args.skip_if_done and per_run_fit_results_csv.exists() and per_run_unresolved_csv.exists() and per_run_category_summary_csv.exists():
            print(f"SKIP {run}: per-run fit outputs already exist")
            continue

        print(f"\\nRun {run}: polarity={meta.get('polarity')}, solid_target={meta.get('solid_target')}, output={root_path}")

        all_fit_rows = []
        all_unresolved_rows = []
        all_category_summary_rows = []

        for vertex_source in vertex_sources:
            for charge in charges:
                for detector_region in detector_regions:
                    sectors = sectors_for_detector_region(detector_region, forward_sectors, central_sectors)

                    for sector in sectors:
                        if should_skip_category(vertex_source, detector_region, fit_config):
                            skip_rows = make_skip_rows(
                                run=run,
                                meta=meta,
                                vertex_source=vertex_source,
                                charge=charge,
                                detector_region=detector_region,
                                sector=sector,
                                fit_config=fit_config,
                            )
                            all_fit_rows.extend(skip_rows)
                            all_unresolved_rows.extend(skip_rows)
                            all_category_summary_rows.append(
                                make_skip_category_summary(
                                    run=run, meta=meta, vertex_source=vertex_source,
                                    charge=charge, detector_region=detector_region, sector=sector
                                )
                            )
                            print(f"SKIP {run} {vertex_source:8s} {charge:8s} {detector_region:7s} sector={sector}")
                            continue

                        rel_plot = (
                            Path(str(run))
                            / "ld2_solid_local_fits"
                            / str(meta.get("polarity", "unknown_polarity"))
                            / str(meta.get("solid_target", "unknown_target"))
                            / vertex_source
                            / detector_region
                            / charge
                            / f"{sector_label_for_path(detector_region, sector)}.png"
                        )
                        output_plot = plots_dir / rel_plot

                        fit_rows, unresolved_rows, category_summary = fit_ld2_solid_local_category(
                            run=run,
                            meta=meta,
                            root_path=root_path,
                            vertex_source=vertex_source,
                            charge=charge,
                            detector_region=detector_region,
                            sector=sector,
                            config=fit_config,
                            output_plot=output_plot,
                        )

                        all_fit_rows.extend(fit_rows)
                        all_unresolved_rows.extend(unresolved_rows)
                        all_category_summary_rows.append(category_summary)

                        status = "BOTH_GOOD" if category_summary["both_good"] else "CHECK"
                        gap = category_summary["mean_gap_solid_minus_ld2"]
                        gap_txt = f" gap={gap:.3f}" if gap is not None and gap == gap else ""
                        print(
                            f"{status:10s} {run} {vertex_source:8s} {charge:8s} "
                            f"{detector_region:7s} sector={sector}"
                            f" ld2={category_summary['ld2_fit_status']} solid={category_summary['solid_fit_status']}{gap_txt}"
                        )

        save_ld2_solid_local_outputs(
            fit_rows=all_fit_rows,
            unresolved_rows=all_unresolved_rows,
            category_summary_rows=all_category_summary_rows,
            fit_results_csv=per_run_fit_results_csv,
            unresolved_csv=per_run_unresolved_csv,
            category_summary_csv=per_run_category_summary_csv,
        )

        print(f"  wrote per-run fit results: {per_run_fit_results_csv}")
        print(f"  wrote per-run unresolved table: {per_run_unresolved_csv}")
        print(f"  wrote per-run category summary: {per_run_category_summary_csv}")

        if args.write_merged_outputs:
            merged_fit_rows.extend(all_fit_rows)
            merged_unresolved_rows.extend(all_unresolved_rows)
            merged_category_summary_rows.extend(all_category_summary_rows)

    if args.write_merged_outputs and (merged_fit_rows or merged_unresolved_rows or merged_category_summary_rows):
        save_ld2_solid_local_outputs(
            fit_rows=merged_fit_rows,
            unresolved_rows=merged_unresolved_rows,
            category_summary_rows=merged_category_summary_rows,
            fit_results_csv=merged_fit_results_csv,
            unresolved_csv=merged_unresolved_csv,
            category_summary_csv=merged_category_summary_csv,
        )
        print(f"\\nWrote merged fit results: {merged_fit_results_csv}")
        print(f"Wrote merged unresolved categories: {merged_unresolved_csv}")
        print(f"Wrote merged category summary: {merged_category_summary_csv}")
    elif args.write_merged_outputs:
        print("\\nMerged outputs not written because no new rows were produced.")

if __name__ == "__main__":
    main()

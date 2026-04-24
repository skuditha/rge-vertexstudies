#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rge_vertex.io.configs import load_yaml, repo_path
from rge_vertex.studies.production_qa import (
    iter_filtered_runs,
    sector_label_for_path,
    sectors_for_detector_region,
)
from rge_vertex.studies.charge_vs_pid import (
    should_skip_category,
    build_comparison_rows,
    save_charge_vs_pid_outputs,
)
from rge_vertex.fitting.ld2_solid_local import fit_ld2_solid_local_category


def _parse_list_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def make_skip_rows(
    *,
    run: str,
    meta: dict,
    vertex_source: str,
    charge: str,
    pid: int | None,
    pid_category: str,
    detector_region: str,
    sector: int | str,
    config: dict,
) -> tuple[list[dict], dict]:
    fit_rows = []
    for component, comp_cfg in config["components"].items():
        fit_rows.append(
            {
                "run": run,
                "label": meta.get("label", ""),
                "run_class": meta.get("run_class", ""),
                "polarity": meta.get("polarity", ""),
                "solid_target": meta.get("solid_target", ""),
                "target_config": meta.get("target_config", ""),
                "vertex_source": vertex_source,
                "charge": charge,
                "pid": pid,
                "pid_category": pid_category,
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
                "background_enabled": bool(config.get("local_background", {}).get("enabled", True)),
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
                "box_width": None,
                "box_width_error": None,
                "bkg_c0": None,
                "bkg_c0_error": None,
                "bkg_c1": None,
                "bkg_c1_error": None,
                "bkg_c2": None,
                "bkg_c2_error": None,
                "message": "Central detector is sectorless and FTrack-only is skipped by config.",
            }
        )

    category_summary = {
        "run": run,
        "label": meta.get("label", ""),
        "run_class": meta.get("run_class", ""),
        "polarity": meta.get("polarity", ""),
        "solid_target": meta.get("solid_target", ""),
        "target_config": meta.get("target_config", ""),
        "vertex_source": vertex_source,
        "charge": charge,
        "pid": pid,
        "pid_category": pid_category,
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
    return fit_rows, category_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare charge-only vs PID-split LD2+solid fits.")
    parser.add_argument("--runs-config", default="configs/runs.yaml")
    parser.add_argument("--config", default="configs/pid_study.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run", default=None)
    parser.add_argument("--polarity", default=None)
    parser.add_argument("--polarities", default=None)
    parser.add_argument("--solid-targets", default=None)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--per-run-fit-dir", default=None)
    parser.add_argument("--per-run-table-dir", default=None)
    parser.add_argument("--write-merged-outputs", action="store_true")
    parser.add_argument("--skip-if-done", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    cfg_all = load_yaml(args.config)
    cfg = cfg_all["pid_study"]

    run_filter = cfg.get("run_filter", {})
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
        raise RuntimeError("No ld2_solid runs matched the PID-study filters.")

    output_cfg = cfg.get("outputs", {})
    plots_dir = repo_path(output_cfg.get("plots_dir", "outputs/plots"), repo_root)
    per_run_fit_dir = repo_path(
        args.per_run_fit_dir or output_cfg.get("per_run_fit_dir", "outputs/fits/pid_study_per_run"),
        repo_root,
    )
    per_run_table_dir = repo_path(
        args.per_run_table_dir or output_cfg.get("per_run_table_dir", "outputs/tables/pid_study_per_run"),
        repo_root,
    )

    categories_cfg = cfg["categories"]
    vertex_sources = categories_cfg.get("vertex_sources", ["particle", "ftrack", "hybrid"])
    detector_regions = categories_cfg.get("detector_regions", ["forward", "central"])
    forward_sectors = categories_cfg.get("forward_sectors", ["all", 1, 2, 3, 4, 5, 6])
    central_sectors = categories_cfg.get("central_sectors", ["all"])

    comparison_groups = cfg.get("comparison_groups", {})

    merged_fit_rows = []
    merged_unresolved_rows = []
    merged_category_summary_rows = []
    merged_comparison_rows = []

    print(f"Charge-vs-PID study: processing {len(selected_runs)} run(s)")

    for run, meta in selected_runs:
        root_path = repo_path(meta.get("output_root", f"outputs/ntuples/{run}.root"), repo_root)
        if not root_path.exists():
            raise FileNotFoundError(f"Ntuple not found for run {run}: {root_path}")

        per_run_fit_results_csv = per_run_fit_dir / f"{run}_pid_study_fit_results.csv"
        per_run_unresolved_csv = per_run_table_dir / f"{run}_pid_study_unresolved.csv"
        per_run_category_summary_csv = per_run_table_dir / f"{run}_pid_study_category_summary.csv"
        per_run_comparison_csv = per_run_table_dir / f"{run}_charge_vs_pid_comparison_summary.csv"

        if (
            args.skip_if_done
            and per_run_fit_results_csv.exists()
            and per_run_unresolved_csv.exists()
            and per_run_category_summary_csv.exists()
            and per_run_comparison_csv.exists()
        ):
            print(f"SKIP {run}: per-run PID-study outputs already exist")
            continue

        print(f"\nRun {run}: polarity={meta.get('polarity')}, solid_target={meta.get('solid_target')}, output={root_path}")

        fit_rows = []
        unresolved_rows = []
        category_summary_rows = []

        for group_name, group_cfg in comparison_groups.items():
            for cat in group_cfg.get("categories", []):
                pid_category = cat["name"]
                charge = cat["charge"]
                pid = cat.get("pid")

                for vertex_source in vertex_sources:
                    for detector_region in detector_regions:
                        sectors = sectors_for_detector_region(detector_region, forward_sectors, central_sectors)

                        for sector in sectors:
                            if should_skip_category(vertex_source, detector_region, cfg):
                                skip_fit_rows, skip_summary = make_skip_rows(
                                    run=run, meta=meta, vertex_source=vertex_source, charge=charge, pid=pid,
                                    pid_category=pid_category, detector_region=detector_region, sector=sector, config=cfg,
                                )
                                fit_rows.extend(skip_fit_rows)
                                unresolved_rows.extend(skip_fit_rows)
                                category_summary_rows.append(skip_summary)
                                print(f"SKIP {run} {group_name:8s} {pid_category:16s} {vertex_source:8s} {detector_region:7s} sector={sector}")
                                continue

                            rel_plot = (
                                Path(str(run))
                                / "charge_vs_pid"
                                / str(meta.get("polarity", "unknown_polarity"))
                                / str(meta.get("solid_target", "unknown_target"))
                                / group_name
                                / pid_category
                                / vertex_source
                                / detector_region
                                / f"{sector_label_for_path(detector_region, sector)}.png"
                            )
                            output_plot = plots_dir / rel_plot

                            cat_fit_rows, cat_unresolved_rows, cat_summary = fit_ld2_solid_local_category(
                                run=run,
                                meta=meta,
                                root_path=root_path,
                                vertex_source=vertex_source,
                                charge=charge,
                                detector_region=detector_region,
                                sector=sector,
                                config=cfg,
                                output_plot=output_plot,
                                pid=pid,
                                pid_category=pid_category,
                            )

                            fit_rows.extend(cat_fit_rows)
                            unresolved_rows.extend(cat_unresolved_rows)
                            category_summary_rows.append(cat_summary)

                            print(
                                f"{run} {group_name:8s} {pid_category:16s} "
                                f"{vertex_source:8s} {detector_region:7s} sector={sector} "
                                f"ld2={cat_summary['ld2_fit_status']} solid={cat_summary['solid_fit_status']}"
                            )

        comparison_rows = build_comparison_rows(category_summary_rows, comparison_groups)

        save_charge_vs_pid_outputs(
            fit_rows=fit_rows,
            unresolved_rows=unresolved_rows,
            category_summary_rows=category_summary_rows,
            comparison_rows=comparison_rows,
            fit_results_csv=per_run_fit_results_csv,
            unresolved_csv=per_run_unresolved_csv,
            category_summary_csv=per_run_category_summary_csv,
            comparison_csv=per_run_comparison_csv,
        )

        print(f"  wrote per-run fit results: {per_run_fit_results_csv}")
        print(f"  wrote per-run unresolved table: {per_run_unresolved_csv}")
        print(f"  wrote per-run category summary: {per_run_category_summary_csv}")
        print(f"  wrote per-run comparison summary: {per_run_comparison_csv}")

        if args.write_merged_outputs:
            merged_fit_rows.extend(fit_rows)
            merged_unresolved_rows.extend(unresolved_rows)
            merged_category_summary_rows.extend(category_summary_rows)
            merged_comparison_rows.extend(comparison_rows)

    if args.write_merged_outputs and (
        merged_fit_rows or merged_unresolved_rows or merged_category_summary_rows or merged_comparison_rows
    ):
        output_cfg = cfg.get("outputs", {})
        save_charge_vs_pid_outputs(
            fit_rows=merged_fit_rows,
            unresolved_rows=merged_unresolved_rows,
            category_summary_rows=merged_category_summary_rows,
            comparison_rows=merged_comparison_rows,
            fit_results_csv=repo_path(output_cfg.get("merged_fit_results_csv", "outputs/fits/pid_study_fit_results.csv"), repo_root),
            unresolved_csv=repo_path(output_cfg.get("merged_unresolved_csv", "outputs/tables/pid_study_unresolved_categories.csv"), repo_root),
            category_summary_csv=repo_path(output_cfg.get("merged_category_summary_csv", "outputs/tables/pid_study_category_summary.csv"), repo_root),
            comparison_csv=repo_path(output_cfg.get("merged_comparison_csv", "outputs/tables/charge_vs_pid_comparison_summary.csv"), repo_root),
        )


if __name__ == "__main__":
    main()

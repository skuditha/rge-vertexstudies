#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rge_vertex.io.configs import load_yaml, repo_path
from rge_vertex.studies.run_dependence import (
    filter_dataframe,
    load_csv,
    make_entry_fraction_plots,
    make_fit_run_dependence_plots,
    save_run_dependence_summary,
)


def _parse_list_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Make run-dependence plots from merged LD2+solid fit summaries.")
    parser.add_argument("--config", default="configs/run_dependence.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--category-summary-csv", default=None)
    parser.add_argument("--production-qa-summary-csv", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--summary-csv", default=None)

    parser.add_argument("--polarities", default=None)
    parser.add_argument("--solid-targets", default=None)
    parser.add_argument("--vertex-sources", default=None)
    parser.add_argument("--charges", default=None)
    parser.add_argument("--detector-regions", default=None)
    parser.add_argument("--sectors", default=None)

    parser.add_argument("--no-entry-fractions", action="store_true")
    parser.add_argument("--allow-bad-fits", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    cfg_all = load_yaml(args.config)
    cfg = cfg_all["run_dependence"]

    input_cfg = cfg.get("inputs", {})
    filter_cfg = cfg.get("filters", {})
    plotting_cfg = cfg.get("plotting", {})
    output_cfg = cfg.get("outputs", {})

    category_summary_csv = repo_path(
        args.category_summary_csv or input_cfg.get("category_summary_csv", "outputs/tables/ld2_solid_category_summary.csv"),
        repo_root,
    )
    production_qa_summary_csv = repo_path(
        args.production_qa_summary_csv or input_cfg.get("production_qa_summary_csv", "outputs/tables/production_qa_histogram_summary.csv"),
        repo_root,
    )
    plots_dir = repo_path(
        args.plots_dir or output_cfg.get("plots_dir", "outputs/plots/run_dependence"),
        repo_root,
    )
    summary_csv = repo_path(
        args.summary_csv or output_cfg.get("summary_csv", "outputs/tables/run_dependence_plot_groups.csv"),
        repo_root,
    )

    polarities = _parse_list_arg(args.polarities) or filter_cfg.get("polarities")
    solid_targets = _parse_list_arg(args.solid_targets) or filter_cfg.get("solid_targets")
    vertex_sources = _parse_list_arg(args.vertex_sources) or filter_cfg.get("vertex_sources")
    charges = _parse_list_arg(args.charges) or filter_cfg.get("charges")
    detector_regions = _parse_list_arg(args.detector_regions) or filter_cfg.get("detector_regions")
    sectors = _parse_list_arg(args.sectors) or filter_cfg.get("sectors")

    metrics = plotting_cfg.get("metrics", [])
    require_good_fits = not args.allow_bad_fits and bool(plotting_cfg.get("require_good_fits", True))
    make_entry_fractions = (not args.no_entry_fractions) and bool(plotting_cfg.get("make_entry_fraction_plots", True))

    category_df = load_csv(category_summary_csv)
    category_df = filter_dataframe(
        category_df,
        polarities=polarities,
        solid_targets=solid_targets,
        vertex_sources=vertex_sources,
        charges=charges,
        detector_regions=detector_regions,
        sectors=sectors,
    )

    summary_rows = make_fit_run_dependence_plots(
        category_df,
        metrics=metrics,
        plots_dir=plots_dir,
        require_good_fits=require_good_fits,
    )

    if make_entry_fractions and production_qa_summary_csv.exists():
        qa_df = load_csv(production_qa_summary_csv)
        qa_df = filter_dataframe(
            qa_df,
            polarities=polarities,
            solid_targets=solid_targets,
            charges=charges,
            detector_regions=detector_regions,
            sectors=sectors,
        )
        summary_rows.extend(
            make_entry_fraction_plots(
                qa_df,
                plots_dir=plots_dir,
            )
        )

    save_run_dependence_summary(summary_rows, summary_csv)

    print(f"Wrote run-dependence plot summary: {summary_csv}")
    print(f"Plots directory: {plots_dir}")
    print(f"Number of plots: {len(summary_rows)}")


if __name__ == "__main__":
    main()

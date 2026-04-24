#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rge_vertex.io.configs import load_yaml, repo_path
from rge_vertex.studies.cut_validation import (
    build_validation_plots,
    filter_recommended_cuts,
    load_recommended_cuts,
    load_runs_lookup,
    save_pdf_bundle,
    save_plot_index,
)


def _parse_list_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Make publication-style validation plots for recommended LD2+solid vertex cuts.")
    parser.add_argument("--config", default="configs/cut_validation.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--runs-config", default=None)
    parser.add_argument("--recommended-cuts-csv", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--plot-index-csv", default=None)
    parser.add_argument("--pdf-bundle", default=None)

    parser.add_argument("--polarities", default=None)
    parser.add_argument("--solid-targets", default=None)
    parser.add_argument("--charges", default=None)
    parser.add_argument("--detector-regions", default=None)
    parser.add_argument("--sectors", default=None)
    parser.add_argument("--max-plots", default=None, type=int)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    cfg_all = load_yaml(args.config)
    cfg = cfg_all["cut_validation"]

    input_cfg = cfg.get("inputs", {})
    selection_cfg = cfg.get("selection", {})
    run_cfg = cfg.get("representative_run", {})
    hist_cfg = cfg.get("histogram", {})
    plot_cfg = cfg.get("plotting", {})
    output_cfg = cfg.get("outputs", {})

    runs_config = repo_path(
        args.runs_config or input_cfg.get("runs_config", "configs/runs.yaml"),
        repo_root,
    )
    recommended_cuts_csv = repo_path(
        args.recommended_cuts_csv or input_cfg.get("recommended_cuts_csv", "outputs/tables/ld2_solid_vertex_cuts_recommended.csv"),
        repo_root,
    )
    plots_dir = repo_path(
        args.plots_dir or output_cfg.get("plots_dir", "outputs/plots/cut_validation"),
        repo_root,
    )
    plot_index_csv = repo_path(
        args.plot_index_csv or output_cfg.get("plot_index_csv", "outputs/tables/cut_validation_plot_index.csv"),
        repo_root,
    )
    pdf_bundle = repo_path(
        args.pdf_bundle or output_cfg.get("pdf_bundle", "outputs/plots/cut_validation/cut_validation_bundle.pdf"),
        repo_root,
    )

    polarities = _parse_list_arg(args.polarities) or selection_cfg.get("polarities")
    solid_targets = _parse_list_arg(args.solid_targets) or selection_cfg.get("solid_targets")
    charges = _parse_list_arg(args.charges) or selection_cfg.get("charges")
    detector_regions = _parse_list_arg(args.detector_regions) or selection_cfg.get("detector_regions")
    sectors = _parse_list_arg(args.sectors) or selection_cfg.get("sectors")
    max_plots = args.max_plots if args.max_plots is not None else selection_cfg.get("max_plots")

    reco_df = load_recommended_cuts(recommended_cuts_csv)
    reco_df = filter_recommended_cuts(
        reco_df,
        polarities=polarities,
        solid_targets=solid_targets,
        charges=charges,
        detector_regions=detector_regions,
        sectors=sectors,
        max_plots=max_plots,
    )

    runs_lookup = load_runs_lookup(
        runs_config,
        repo_root=repo_root,
        enabled_only=bool(run_cfg.get("enabled_only", True)),
    )

    rows = build_validation_plots(
        reco_df,
        runs_lookup=runs_lookup,
        repo_root=repo_root,
        plots_dir=plots_dir,
        representative_run_strategy=run_cfg.get("strategy", "median_source_run"),
        bins=int(hist_cfg.get("bins", 240)),
        vz_range=(float(hist_cfg.get("vz_min_cm", -12.0)), float(hist_cfg.get("vz_max_cm", 4.0))),
        chi2pid_abs_max=hist_cfg.get("chi2pid_abs_max"),
        figure_width=float(plot_cfg.get("figure_width", 10.0)),
        figure_height=float(plot_cfg.get("figure_height", 6.0)),
        dpi=int(plot_cfg.get("dpi", 180)),
        logy=bool(plot_cfg.get("logy", False)),
        show_text_box=bool(plot_cfg.get("show_text_box", True)),
        legend_fontsize=float(plot_cfg.get("legend_fontsize", 9)),
        title_fontsize=float(plot_cfg.get("title_fontsize", 12)),
    )

    plot_index_df = save_plot_index(rows, plot_index_csv)

    if not args.no_pdf:
        save_pdf_bundle(plot_index_df, pdf_bundle)

    print(f"Wrote plot index: {plot_index_csv}")
    print(f"Plots directory: {plots_dir}")
    if not args.no_pdf:
        print(f"Wrote PDF bundle: {pdf_bundle}")
    print(f"Requested plots: {len(reco_df)}")
    print(f"Plot-index rows: {len(plot_index_df)}")


if __name__ == "__main__":
    main()

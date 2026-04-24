#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from rge_vertex.io.configs import load_yaml, repo_path


def merge_csvs(input_dir: Path, pattern: str, output_csv: Path, dedupe_subset: list[str] | None) -> None:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"No CSVs found in {input_dir} matching {pattern}")

    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)

    if dedupe_subset:
        existing = [c for c in dedupe_subset if c in merged.columns]
        if existing:
            merged = merged.drop_duplicates(subset=existing, keep="last")

    sort_cols = [c for c in [
        "run", "polarity", "solid_target", "vertex_source", "charge",
        "pid_category", "detector_region", "sector", "component"
    ] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Merged {len(files)} files into {output_csv}")
    print(f"Rows: {len(merged)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-run charge-vs-PID study CSVs.")
    parser.add_argument("--config", default="configs/pid_study.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--per-run-fit-dir", default=None)
    parser.add_argument("--per-run-table-dir", default=None)
    parser.add_argument("--fit-pattern", default="*_pid_study_fit_results.csv")
    parser.add_argument("--unresolved-pattern", default="*_pid_study_unresolved.csv")
    parser.add_argument("--category-pattern", default="*_pid_study_category_summary.csv")
    parser.add_argument("--comparison-pattern", default="*_charge_vs_pid_comparison_summary.csv")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    cfg_all = load_yaml(args.config)
    cfg = cfg_all["pid_study"]
    output_cfg = cfg.get("outputs", {})

    per_run_fit_dir = repo_path(
        args.per_run_fit_dir or output_cfg.get("per_run_fit_dir", "outputs/fits/pid_study_per_run"),
        repo_root,
    )
    per_run_table_dir = repo_path(
        args.per_run_table_dir or output_cfg.get("per_run_table_dir", "outputs/tables/pid_study_per_run"),
        repo_root,
    )

    merge_csvs(
        per_run_fit_dir,
        args.fit_pattern,
        repo_path(output_cfg.get("merged_fit_results_csv", "outputs/fits/pid_study_fit_results.csv"), repo_root),
        dedupe_subset=["run", "vertex_source", "charge", "pid_category", "detector_region", "sector", "component"],
    )
    merge_csvs(
        per_run_table_dir,
        args.unresolved_pattern,
        repo_path(output_cfg.get("merged_unresolved_csv", "outputs/tables/pid_study_unresolved_categories.csv"), repo_root),
        dedupe_subset=["run", "vertex_source", "charge", "pid_category", "detector_region", "sector", "component", "fit_status"],
    )
    merge_csvs(
        per_run_table_dir,
        args.category_pattern,
        repo_path(output_cfg.get("merged_category_summary_csv", "outputs/tables/pid_study_category_summary.csv"), repo_root),
        dedupe_subset=["run", "vertex_source", "charge", "pid_category", "detector_region", "sector"],
    )
    merge_csvs(
        per_run_table_dir,
        args.comparison_pattern,
        repo_path(output_cfg.get("merged_comparison_csv", "outputs/tables/charge_vs_pid_comparison_summary.csv"), repo_root),
        dedupe_subset=["run", "vertex_source", "charge", "test_pid_category", "detector_region", "sector"],
    )


if __name__ == "__main__":
    main()

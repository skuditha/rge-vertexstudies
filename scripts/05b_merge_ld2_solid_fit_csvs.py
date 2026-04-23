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

    sort_cols = [c for c in ["run", "polarity", "solid_target", "vertex_source", "charge", "detector_region", "sector", "component"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Merged {len(files)} files into {output_csv}")
    print(f"Rows: {len(merged)}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-run LD2+solid fit CSVs into combined outputs.")
    parser.add_argument("--fit-config", default="configs/ld2_solid_local_fit.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--per-run-fit-dir", default=None)
    parser.add_argument("--per-run-table-dir", default=None)
    parser.add_argument("--fit-pattern", default="*_ld2_solid_local_fit_results.csv")
    parser.add_argument("--unresolved-pattern", default="*_unresolved_ld2_solid_categories.csv")
    parser.add_argument("--category-pattern", default="*_ld2_solid_category_summary.csv")
    parser.add_argument("--merged-fit-results-csv", default=None)
    parser.add_argument("--merged-unresolved-csv", default=None)
    parser.add_argument("--merged-category-summary-csv", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    fit_all = load_yaml(args.fit_config)
    fit_config = fit_all["ld2_solid_local_fit"]
    output_cfg = fit_config.get("outputs", {})

    per_run_fit_dir = repo_path(args.per_run_fit_dir or output_cfg.get("per_run_fit_dir", "outputs/fits/ld2_solid_per_run"), repo_root)
    per_run_table_dir = repo_path(args.per_run_table_dir or output_cfg.get("per_run_table_dir", "outputs/tables/ld2_solid_per_run"), repo_root)
    merged_fit_results_csv = repo_path(args.merged_fit_results_csv or output_cfg.get("merged_fit_results_csv", "outputs/fits/ld2_solid_local_fit_results.csv"), repo_root)
    merged_unresolved_csv = repo_path(args.merged_unresolved_csv or output_cfg.get("merged_unresolved_csv", "outputs/tables/unresolved_ld2_solid_categories.csv"), repo_root)
    merged_category_summary_csv = repo_path(args.merged_category_summary_csv or output_cfg.get("merged_category_summary_csv", "outputs/tables/ld2_solid_category_summary.csv"), repo_root)

    merge_csvs(
        per_run_fit_dir,
        args.fit_pattern,
        merged_fit_results_csv,
        dedupe_subset=["run", "vertex_source", "charge", "detector_region", "sector", "component"],
    )
    merge_csvs(
        per_run_table_dir,
        args.unresolved_pattern,
        merged_unresolved_csv,
        dedupe_subset=["run", "vertex_source", "charge", "detector_region", "sector", "component", "fit_status"],
    )
    merge_csvs(
        per_run_table_dir,
        args.category_pattern,
        merged_category_summary_csv,
        dedupe_subset=["run", "vertex_source", "charge", "detector_region", "sector"],
    )

if __name__ == "__main__":
    main()

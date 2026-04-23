#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from rge_vertex.io.configs import load_yaml, repo_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-run production QA CSVs into one combined summary.")
    parser.add_argument("--qa-config", default="configs/production_qa.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--pattern", default="*_production_qa_histogram_summary.csv")
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    qa_all = load_yaml(args.qa_config)
    qa_config = qa_all["production_qa"]
    output_cfg = qa_config.get("outputs", {})

    input_dir = repo_path(
        args.input_dir or output_cfg.get("per_run_summary_dir", "outputs/tables/production_qa_per_run"),
        repo_root,
    )
    output_csv = repo_path(
        args.output_csv or output_cfg.get("merged_summary_csv", "outputs/tables/production_qa_histogram_summary.csv"),
        repo_root,
    )

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No per-run QA CSVs found in {input_dir} matching {args.pattern}")

    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)

    if not args.no_dedupe:
        subset = ["run", "vertex_source", "charge", "detector_region", "sector", "hist_csv"]
        existing = [c for c in subset if c in merged.columns]
        if existing:
            merged = merged.drop_duplicates(subset=existing, keep="last")

    sort_cols = [c for c in ["run", "polarity", "solid_target", "vertex_source", "charge", "detector_region", "sector"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"Merged {len(files)} per-run QA CSVs into {output_csv}")
    print(f"Rows: {len(merged)}")

if __name__ == "__main__":
    main()

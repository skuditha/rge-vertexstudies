#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

from rge_vertex.io.configs import load_yaml, repo_path
from rge_vertex.studies.production_qa import (
    iter_filtered_runs,
    make_production_qa_for_run,
    save_production_qa_summary,
)

def _parse_list_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make QA vertex histograms for ld2_solid production runs. "
                    "Parallel-safe default: writes one summary CSV per run."
    )
    parser.add_argument("--runs-config", default="configs/runs.yaml")
    parser.add_argument("--qa-config", default="configs/production_qa.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run", default=None, help="Optional single run number.")
    parser.add_argument("--polarity", default=None, help="Optional polarity filter.")
    parser.add_argument("--polarities", default=None, help="Optional comma-separated polarity filter.")
    parser.add_argument("--solid-targets", default=None, help="Optional comma-separated target filter.")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--per-run-summary-dir", default=None,
                        help="Directory for one per-run QA summary CSV per job.")
    parser.add_argument("--write-merged-summary", action="store_true",
                        help="Also write a merged summary CSV for all runs processed in this invocation.")
    parser.add_argument("--merged-summary-csv", default=None,
                        help="Merged summary CSV path.")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="Skip a run if its per-run summary CSV already exists.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    qa_all = load_yaml(args.qa_config)
    qa_config = qa_all["production_qa"]
    run_filter = qa_config.get("run_filter", {})
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

    output_cfg = qa_config.get("outputs", {})
    per_run_summary_dir = repo_path(
        args.per_run_summary_dir or output_cfg.get("per_run_summary_dir", "outputs/tables/production_qa_per_run"),
        repo_root,
    )
    merged_summary_csv = repo_path(
        args.merged_summary_csv or output_cfg.get("merged_summary_csv", "outputs/tables/production_qa_histogram_summary.csv"),
        repo_root,
    )

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
        raise RuntimeError("No runs matched the production QA filters.")

    print(f"Production QA: processing {len(selected_runs)} run(s)")
    all_rows = []

    for run, meta in selected_runs:
        per_run_csv = per_run_summary_dir / f"{run}_production_qa_histogram_summary.csv"
        if args.skip_if_done and per_run_csv.exists():
            print(f"SKIP {run}: per-run QA summary already exists at {per_run_csv}")
            continue

        print(f"\\nRun {run}: polarity={meta.get('polarity')}, solid_target={meta.get('solid_target')}, output={meta.get('output_root')}")
        rows = make_production_qa_for_run(
            run=run,
            meta=meta,
            repo_root=repo_root,
            qa_config=qa_config,
        )
        all_rows.extend(rows)
        save_production_qa_summary(rows, per_run_csv)
        print(f"  wrote per-run QA summary: {per_run_csv}")
        print(f"  wrote QA rows: {len(rows)}")

    if args.write_merged_summary and all_rows:
        save_production_qa_summary(all_rows, merged_summary_csv)
        print(f"\\nWrote merged production QA summary: {merged_summary_csv}")
    elif args.write_merged_summary:
        print("\\nMerged summary not written because no new rows were produced.")

if __name__ == "__main__":
    main()

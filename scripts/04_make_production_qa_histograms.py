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
        description="Make QA vertex histograms for all enabled ld2_solid production runs."
    )
    parser.add_argument("--runs-config", default="configs/runs.yaml")
    parser.add_argument("--qa-config", default="configs/production_qa.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run", default=None, help="Optional single run number.")
    parser.add_argument("--polarity", default=None, help="Optional polarity filter, e.g. inbending or outbending.")
    parser.add_argument(
        "--polarities",
        default=None,
        help="Optional comma-separated polarity filter, e.g. inbending,outbending.",
    )
    parser.add_argument(
        "--solid-targets",
        default=None,
        help="Optional comma-separated target filter, e.g. C,Al,Cu,Sn,Pb.",
    )
    parser.add_argument("--include-disabled", action="store_true")
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

    all_rows = []

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
        raise RuntimeError(
            "No runs matched the production QA filters. Check runs.yaml and configs/production_qa.yaml."
        )

    print(f"Production QA: processing {len(selected_runs)} run(s)")

    for run, meta in selected_runs:
        print(
            f"\nRun {run}: polarity={meta.get('polarity')}, "
            f"solid_target={meta.get('solid_target')}, output={meta.get('output_root')}"
        )

        rows = make_production_qa_for_run(
            run=run,
            meta=meta,
            repo_root=repo_root,
            qa_config=qa_config,
        )
        all_rows.extend(rows)

        print(f"  wrote QA rows: {len(rows)}")

    output_cfg = qa_config.get("outputs", {})
    summary_csv = repo_path(
        output_cfg.get("summary_csv", "outputs/tables/production_qa_histogram_summary.csv"),
        repo_root,
    )

    save_production_qa_summary(all_rows, summary_csv)
    print(f"\nWrote production QA summary: {summary_csv}")


if __name__ == "__main__":
    main()

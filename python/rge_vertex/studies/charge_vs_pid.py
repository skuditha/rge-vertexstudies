from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def should_skip_category(vertex_source: str, detector_region: str, fit_config: dict) -> bool:
    categories = fit_config["categories"]
    return (
        bool(categories.get("skip_central_ftrack", True))
        and vertex_source == "ftrack"
        and detector_region == "central"
    )


def build_comparison_rows(category_summary_rows: list[dict[str, Any]], comparison_groups: dict[str, Any]) -> list[dict[str, Any]]:
    df = pd.DataFrame(category_summary_rows)
    if df.empty:
        return []

    compare_rows: list[dict[str, Any]] = []
    key_cols = ["run", "polarity", "solid_target", "vertex_source", "detector_region", "sector"]

    for group_name, group_cfg in comparison_groups.items():
        baseline_name = group_cfg["baseline"]
        group_categories = [c["name"] for c in group_cfg.get("categories", [])]

        group_df = df[df["pid_category"].isin(group_categories)].copy()
        if group_df.empty:
            continue

        for _, subset in group_df.groupby(key_cols, dropna=False):
            baseline_match = subset[subset["pid_category"] == baseline_name]
            if baseline_match.empty:
                continue
            baseline = baseline_match.iloc[0]

            for _, row in subset.iterrows():
                if row["pid_category"] == baseline_name:
                    continue

                out = {
                    "comparison_group": group_name,
                    "baseline_pid_category": baseline_name,
                    "test_pid_category": row["pid_category"],
                    "run": row["run"],
                    "polarity": row["polarity"],
                    "solid_target": row["solid_target"],
                    "vertex_source": row["vertex_source"],
                    "detector_region": row["detector_region"],
                    "sector": row["sector"],
                    "charge": row["charge"],
                    "baseline_entries": baseline["entries_category"],
                    "test_entries": row["entries_category"],
                    "test_fraction_vs_baseline": (
                        row["entries_category"] / baseline["entries_category"]
                        if baseline["entries_category"] and baseline["entries_category"] > 0
                        else np.nan
                    ),
                    "baseline_both_good": baseline["both_good"],
                    "test_both_good": row["both_good"],
                }

                for metric in [
                    "ld2_mean",
                    "solid_mean",
                    "ld2_sigma",
                    "solid_sigma",
                    "mean_gap_solid_minus_ld2",
                ]:
                    out[f"baseline_{metric}"] = baseline.get(metric, np.nan)
                    out[f"test_{metric}"] = row.get(metric, np.nan)
                    if np.isfinite(baseline.get(metric, np.nan)) and np.isfinite(row.get(metric, np.nan)):
                        out[f"delta_{metric}"] = row[metric] - baseline[metric]
                    else:
                        out[f"delta_{metric}"] = np.nan

                compare_rows.append(out)

    return compare_rows


def save_charge_vs_pid_outputs(
    *,
    fit_rows: list[dict[str, Any]],
    unresolved_rows: list[dict[str, Any]],
    category_summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    fit_results_csv: str | Path,
    unresolved_csv: str | Path,
    category_summary_csv: str | Path,
    comparison_csv: str | Path,
) -> None:
    fit_results_csv = Path(fit_results_csv)
    unresolved_csv = Path(unresolved_csv)
    category_summary_csv = Path(category_summary_csv)
    comparison_csv = Path(comparison_csv)

    fit_results_csv.parent.mkdir(parents=True, exist_ok=True)
    unresolved_csv.parent.mkdir(parents=True, exist_ok=True)
    category_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fit_rows).to_csv(fit_results_csv, index=False)
    pd.DataFrame(unresolved_rows).to_csv(unresolved_csv, index=False)
    pd.DataFrame(category_summary_rows).to_csv(category_summary_csv, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_csv, index=False)

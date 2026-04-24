from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIT_METRIC_SPECS = {
    "ld2_mean": {
        "ylabel": "LD2 mean vz [cm]",
        "status_mode": "ld2",
    },
    "solid_mean": {
        "ylabel": "Solid mean vz [cm]",
        "status_mode": "solid",
    },
    "ld2_sigma": {
        "ylabel": "LD2 sigma [cm]",
        "status_mode": "ld2",
    },
    "solid_sigma": {
        "ylabel": "Solid sigma [cm]",
        "status_mode": "solid",
    },
    "mean_gap_solid_minus_ld2": {
        "ylabel": "solid mean - LD2 mean [cm]",
        "status_mode": "both",
    },
}


def sanitize_piece(value: Any) -> str:
    text = str(value)
    return text.replace("/", "_").replace(" ", "_")


def sector_label(detector_region: str, sector: Any) -> str:
    if detector_region == "central":
        return "central_all"
    return f"sector_{sanitize_piece(sector)}"


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def filter_dataframe(
    df: pd.DataFrame,
    *,
    polarities: list[str] | None = None,
    solid_targets: list[str] | None = None,
    vertex_sources: list[str] | None = None,
    charges: list[str] | None = None,
    detector_regions: list[str] | None = None,
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()

    if polarities is not None:
        out = out[out["polarity"].isin(polarities)]
    if solid_targets is not None:
        out = out[out["solid_target"].isin(solid_targets)]
    if vertex_sources is not None and "vertex_source" in out.columns:
        out = out[out["vertex_source"].isin(vertex_sources)]
    if charges is not None:
        out = out[out["charge"].isin(charges)]
    if detector_regions is not None:
        out = out[out["detector_region"].isin(detector_regions)]
    if sectors is not None:
        out = out[out["sector"].astype(str).isin([str(s) for s in sectors])]

    return out


def _fit_metric_mask(df: pd.DataFrame, metric: str, require_good_fits: bool) -> pd.Series:
    mask = np.isfinite(df[metric].to_numpy(dtype=float))

    if not require_good_fits:
        return pd.Series(mask, index=df.index)

    mode = FIT_METRIC_SPECS[metric]["status_mode"]
    if mode == "ld2":
        mask &= (df["ld2_fit_status"] == "good").to_numpy()
    elif mode == "solid":
        mask &= (df["solid_fit_status"] == "good").to_numpy()
    elif mode == "both":
        mask &= df["both_good"].fillna(False).to_numpy(dtype=bool)
    else:
        raise ValueError(f"Unknown status mode: {mode}")

    return pd.Series(mask, index=df.index)


def _sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_run_int"] = out["run"].astype(str).str.zfill(6).astype(int)
    out = out.sort_values("_run_int").drop(columns="_run_int")
    return out


def make_fit_run_dependence_plots(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    plots_dir: str | Path,
    require_good_fits: bool = True,
) -> list[dict[str, Any]]:
    plots_dir = Path(plots_dir)
    summary_rows: list[dict[str, Any]] = []

    group_keys = ["polarity", "solid_target", "charge", "detector_region", "sector"]

    for group_values, group_df in df.groupby(group_keys, dropna=False):
        polarity, solid_target, charge, detector_region, sector = group_values
        group_df = _sort_runs(group_df)

        for metric in metrics:
            if metric not in FIT_METRIC_SPECS:
                raise ValueError(f"Unknown fit metric: {metric}")

            plot_df = group_df[_fit_metric_mask(group_df, metric, require_good_fits)].copy()
            if plot_df.empty:
                continue

            output_path = (
                plots_dir
                / "fit_metrics"
                / metric
                / sanitize_piece(polarity)
                / sanitize_piece(solid_target)
                / sanitize_piece(detector_region)
                / sanitize_piece(charge)
                / f"{sector_label(detector_region, sector)}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(9, 6))
            counts_by_source: dict[str, int] = {}

            for vertex_source, source_df in plot_df.groupby("vertex_source", dropna=False):
                source_df = _sort_runs(source_df)
                x = source_df["run"].astype(str).str.zfill(6).astype(int).to_numpy()
                y = source_df[metric].to_numpy(dtype=float)
                counts_by_source[str(vertex_source)] = len(source_df)

                ax.plot(x, y, marker="o", label=str(vertex_source))

            title = (
                f"{metric} vs run\n"
                f"{polarity}, LD2+{solid_target}, {charge}, {detector_region}, {sector_label(detector_region, sector)}"
            )
            ax.set_title(title)
            ax.set_xlabel("run")
            ax.set_ylabel(FIT_METRIC_SPECS[metric]["ylabel"])
            ax.grid(True, alpha=0.25)
            ax.legend()

            fig.tight_layout()
            fig.savefig(output_path, dpi=160)
            plt.close(fig)

            summary_rows.append(
                {
                    "plot_type": "fit_metric",
                    "metric": metric,
                    "polarity": polarity,
                    "solid_target": solid_target,
                    "charge": charge,
                    "detector_region": detector_region,
                    "sector": sector,
                    "plot_path": str(output_path),
                    "n_particle": counts_by_source.get("particle", 0),
                    "n_ftrack": counts_by_source.get("ftrack", 0),
                    "n_hybrid": counts_by_source.get("hybrid", 0),
                }
            )

    return summary_rows


def _dedupe_qa_summary(df: pd.DataFrame) -> pd.DataFrame:
    subset = ["run", "polarity", "solid_target", "charge", "detector_region", "sector"]
    existing = [c for c in subset if c in df.columns]
    out = df.drop_duplicates(subset=existing, keep="first").copy()
    return _sort_runs(out)


def make_entry_fraction_plots(
    df: pd.DataFrame,
    *,
    plots_dir: str | Path,
) -> list[dict[str, Any]]:
    plots_dir = Path(plots_dir)
    df = _dedupe_qa_summary(df)

    summary_rows: list[dict[str, Any]] = []
    group_keys = ["polarity", "solid_target", "charge", "detector_region", "sector"]

    for group_values, group_df in df.groupby(group_keys, dropna=False):
        polarity, solid_target, charge, detector_region, sector = group_values
        group_df = _sort_runs(group_df)

        output_path = (
            plots_dir
            / "entry_fractions"
            / sanitize_piece(polarity)
            / sanitize_piece(solid_target)
            / sanitize_piece(detector_region)
            / sanitize_piece(charge)
            / f"{sector_label(detector_region, sector)}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        x = group_df["run"].astype(str).str.zfill(6).astype(int).to_numpy()
        ftrack_fraction = group_df["ftrack_fraction_vs_particle"].to_numpy(dtype=float)
        hybrid_fraction = group_df["hybrid_fraction_vs_particle"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, ftrack_fraction, marker="o", label="ftrack / particle")
        ax.plot(x, hybrid_fraction, marker="o", label="hybrid / particle")

        title = (
            "Entry fractions vs run\n"
            f"{polarity}, LD2+{solid_target}, {charge}, {detector_region}, {sector_label(detector_region, sector)}"
        )
        ax.set_title(title)
        ax.set_xlabel("run")
        ax.set_ylabel("fraction of particle entries")
        ax.grid(True, alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

        summary_rows.append(
            {
                "plot_type": "entry_fraction",
                "metric": "entry_fractions",
                "polarity": polarity,
                "solid_target": solid_target,
                "charge": charge,
                "detector_region": detector_region,
                "sector": sector,
                "plot_path": str(output_path),
                "n_runs": len(group_df),
            }
        )

    return summary_rows


def save_run_dependence_summary(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

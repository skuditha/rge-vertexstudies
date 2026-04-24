from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from rge_vertex.io.configs import repo_path
from rge_vertex.plotting.histograms import collect_vz_histogram


def _norm_run(value: Any) -> str:
    return str(value).zfill(6)


def _norm_sector(detector_region: str, sector: Any) -> str:
    if detector_region == "central":
        return "all"
    if pd.isna(sector):
        return "all"
    text = str(sector).strip()
    if text in ("all", "central_all"):
        return "all"
    try:
        return str(int(float(text)))
    except Exception:
        return text


def load_recommended_cuts(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recommended cuts CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["charge"] = df["charge"].astype(str)
    df["polarity"] = df["polarity"].astype(str)
    df["solid_target"] = df["solid_target"].astype(str)
    df["detector_region"] = df["detector_region"].astype(str)
    df["sector_norm"] = [
        _norm_sector(det, sec)
        for det, sec in zip(df["detector_region"], df["sector"])
    ]
    if "vertex_source" not in df.columns:
        df["vertex_source"] = "hybrid"
    else:
        df["vertex_source"] = df["vertex_source"].astype(str)

    if "recommendation_status" not in df.columns:
        df["recommendation_status"] = "recommended"

    return df


def filter_recommended_cuts(
    df: pd.DataFrame,
    *,
    polarities: list[str] | None = None,
    solid_targets: list[str] | None = None,
    charges: list[str] | None = None,
    detector_regions: list[str] | None = None,
    sectors: list[str] | None = None,
    max_plots: int | None = None,
) -> pd.DataFrame:
    out = df.copy()

    out = out[out["recommendation_status"] == "recommended"]

    if polarities is not None:
        out = out[out["polarity"].isin(polarities)]
    if solid_targets is not None:
        out = out[out["solid_target"].isin(solid_targets)]
    if charges is not None:
        out = out[out["charge"].isin(charges)]
    if detector_regions is not None:
        out = out[out["detector_region"].isin(detector_regions)]
    if sectors is not None:
        wanted = {str(s) for s in sectors}
        out = out[out["sector_norm"].astype(str).isin(wanted)]

    sort_cols = [c for c in ["polarity", "solid_target", "charge", "detector_region", "sector_norm"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    if max_plots is not None:
        out = out.head(int(max_plots)).reset_index(drop=True)

    return out


def load_runs_lookup(runs_config: str | Path, *, repo_root: str | Path = ".", enabled_only: bool = True) -> dict[str, dict[str, Any]]:
    runs_config = Path(runs_config)
    import yaml

    with open(runs_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    lookup: dict[str, dict[str, Any]] = {}
    for run, meta in (cfg.get("runs", {}) or {}).items():
        run_str = _norm_run(run)
        meta = dict(meta or {})
        if enabled_only and not bool(meta.get("enabled", True)):
            continue
        lookup[run_str] = meta

    return lookup


def _parse_source_runs(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    runs = [_norm_run(x.strip()) for x in text.split(",") if x.strip()]
    return runs


def _matching_runs_from_lookup(
    row: pd.Series,
    runs_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    matches = []
    for run, meta in runs_lookup.items():
        if str(meta.get("run_class", "")) != "ld2_solid":
            continue
        if str(meta.get("polarity", "")) != str(row["polarity"]):
            continue
        if str(meta.get("solid_target", "")) != str(row["solid_target"]):
            continue
        matches.append(run)
    return sorted(matches)


def choose_representative_run(
    row: pd.Series,
    *,
    runs_lookup: dict[str, dict[str, Any]],
    strategy: str = "median_source_run",
) -> str | None:
    source_runs = [r for r in _parse_source_runs(row.get("source_runs")) if r in runs_lookup]

    if not source_runs:
        source_runs = _matching_runs_from_lookup(row, runs_lookup)

    if not source_runs:
        return None

    if strategy == "first_source_run":
        return source_runs[0]
    if strategy == "last_source_run":
        return source_runs[-1]
    if strategy == "median_source_run":
        return source_runs[len(source_runs) // 2]

    raise ValueError(f"Unknown representative-run strategy: {strategy}")


def _run_root_path(run: str, meta: dict[str, Any], repo_root: str | Path) -> Path:
    return repo_path(meta.get("output_root", f"outputs/ntuples/{run}.root"), repo_root)


def sector_label_for_path(detector_region: str, sector_norm: str) -> str:
    if detector_region == "central":
        return "central_all"
    return f"sector_{sector_norm}"


def make_validation_plot(
    *,
    root_path: str | Path,
    row: pd.Series,
    plot_path: str | Path,
    bins: int,
    vz_range: tuple[float, float],
    chi2pid_abs_max: float | None,
    figure_width: float,
    figure_height: float,
    dpi: int,
    logy: bool,
    show_text_box: bool,
    legend_fontsize: float,
    title_fontsize: float,
) -> dict[str, Any]:
    root_path = Path(root_path)
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    detector_region = str(row["detector_region"])
    sector_norm = str(row["sector_norm"])
    sector_value: Any = "all" if sector_norm == "all" else int(sector_norm)

    hist = collect_vz_histogram(
        root_path,
        vertex_source=str(row["vertex_source"]),
        charge=str(row["charge"]),
        detector_region=detector_region,
        sector=sector_value,
        pid=None,
        chi2pid_abs_max=chi2pid_abs_max,
        bins=int(bins),
        vz_range=vz_range,
    )

    fig, ax = plt.subplots(figsize=(float(figure_width), float(figure_height)))
    ax.step(hist.centers, hist.counts, where="mid", linewidth=1.5, label=f"data, N={hist.entries}")

    # Shaded physics regions
    if np.isfinite(row.get("ld2_cut_low", np.nan)) and np.isfinite(row.get("ld2_cut_high", np.nan)):
        ax.axvspan(
            float(row["ld2_cut_low"]),
            float(row["ld2_cut_high"]),
            alpha=0.18,
            label="LD2 selected",
        )
        ax.axvline(float(row["ld2_cut_low"]), linestyle="--", linewidth=1.0)
        ax.axvline(float(row["ld2_cut_high"]), linestyle="--", linewidth=1.0)

    if np.isfinite(row.get("reference_low", np.nan)) and np.isfinite(row.get("reference_high", np.nan)):
        ax.axvspan(
            float(row["reference_low"]),
            float(row["reference_high"]),
            alpha=0.20,
            label="reference foil forbidden",
        )
        ax.axvline(float(row["reference_low"]), linestyle=":", linewidth=1.0)
        ax.axvline(float(row["reference_high"]), linestyle=":", linewidth=1.0)

    if np.isfinite(row.get("solid_cut_low", np.nan)) and np.isfinite(row.get("solid_cut_high", np.nan)):
        ax.axvspan(
            float(row["solid_cut_low"]),
            float(row["solid_cut_high"]),
            alpha=0.18,
            label="solid selected",
        )
        ax.axvline(float(row["solid_cut_low"]), linestyle="--", linewidth=1.0)
        ax.axvline(float(row["solid_cut_high"]), linestyle="--", linewidth=1.0)

    title = (
        f"Vertex-cut validation: run {row['representative_run']}\n"
        f"{row['polarity']}, LD2+{row['solid_target']}, {row['charge']}, "
        f"{row['detector_region']}, {sector_label_for_path(row['detector_region'], row['sector_norm'])}, "
        f"{row['vertex_source']}.vz"
    )
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("vz [cm]")
    ax.set_ylabel("counts")
    ax.grid(True, alpha=0.25)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=legend_fontsize)

    if show_text_box:
        textbox_lines = [
            f"LD2: [{row['ld2_cut_low']:.3f}, {row['ld2_cut_high']:.3f}] cm",
            f"Ref: [{row['reference_low']:.3f}, {row['reference_high']:.3f}] cm",
            f"Solid: [{row['solid_cut_low']:.3f}, {row['solid_cut_high']:.3f}] cm",
            f"fallback: {row.get('fallback_level', '')}",
            f"n runs: {int(row.get('n_unique_runs', 0))}",
        ]
        ax.text(
            0.98,
            0.98,
            "\n".join(textbox_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.12),
        )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=int(dpi))
    plt.close(fig)

    return {
        "plot_path": str(plot_path),
        "entries_histogram": hist.entries,
        "bins": bins,
        "vz_min_cm": vz_range[0],
        "vz_max_cm": vz_range[1],
    }


def build_validation_plots(
    reco_df: pd.DataFrame,
    *,
    runs_lookup: dict[str, dict[str, Any]],
    repo_root: str | Path,
    plots_dir: str | Path,
    representative_run_strategy: str,
    bins: int,
    vz_range: tuple[float, float],
    chi2pid_abs_max: float | None,
    figure_width: float,
    figure_height: float,
    dpi: int,
    logy: bool,
    show_text_box: bool,
    legend_fontsize: float,
    title_fontsize: float,
) -> list[dict[str, Any]]:
    plots_dir = Path(plots_dir)
    rows: list[dict[str, Any]] = []

    for _, reco_row in reco_df.iterrows():
        representative_run = choose_representative_run(
            reco_row,
            runs_lookup=runs_lookup,
            strategy=representative_run_strategy,
        )
        if representative_run is None:
            rows.append(
                {
                    "polarity": reco_row["polarity"],
                    "solid_target": reco_row["solid_target"],
                    "charge": reco_row["charge"],
                    "detector_region": reco_row["detector_region"],
                    "sector": reco_row["sector_norm"],
                    "representative_run": None,
                    "plot_status": "missing_run",
                    "plot_reason": "no suitable representative run found",
                    "plot_path": "",
                }
            )
            continue

        run_meta = runs_lookup[representative_run]
        root_path = _run_root_path(representative_run, run_meta, repo_root)
        if not root_path.exists():
            rows.append(
                {
                    "polarity": reco_row["polarity"],
                    "solid_target": reco_row["solid_target"],
                    "charge": reco_row["charge"],
                    "detector_region": reco_row["detector_region"],
                    "sector": reco_row["sector_norm"],
                    "representative_run": representative_run,
                    "plot_status": "missing_root",
                    "plot_reason": f"ntuple not found: {root_path}",
                    "plot_path": "",
                }
            )
            continue

        rel_plot = (
            Path(str(reco_row["polarity"]))
            / str(reco_row["solid_target"])
            / str(reco_row["charge"])
            / str(reco_row["detector_region"])
            / f"{sector_label_for_path(reco_row['detector_region'], reco_row['sector_norm'])}.png"
        )
        plot_path = plots_dir / rel_plot

        plot_input_row = reco_row.copy()
        plot_input_row["representative_run"] = representative_run

        plot_info = make_validation_plot(
            root_path=root_path,
            row=plot_input_row,
            plot_path=plot_path,
            bins=bins,
            vz_range=vz_range,
            chi2pid_abs_max=chi2pid_abs_max,
            figure_width=figure_width,
            figure_height=figure_height,
            dpi=dpi,
            logy=logy,
            show_text_box=show_text_box,
            legend_fontsize=legend_fontsize,
            title_fontsize=title_fontsize,
        )

        rows.append(
            {
                "polarity": reco_row["polarity"],
                "solid_target": reco_row["solid_target"],
                "charge": reco_row["charge"],
                "vertex_source": reco_row["vertex_source"],
                "detector_region": reco_row["detector_region"],
                "sector": reco_row["sector_norm"],
                "representative_run": representative_run,
                "fallback_level": reco_row.get("fallback_level", ""),
                "n_unique_runs": reco_row.get("n_unique_runs", np.nan),
                "recommendation_status": reco_row.get("recommendation_status", ""),
                "plot_status": "made",
                "plot_reason": "",
                **plot_info,
            }
        )

    return rows


def save_plot_index(rows: list[dict[str, Any]], output_csv: str | Path) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def save_pdf_bundle(plot_index_df: pd.DataFrame, output_pdf: str | Path) -> None:
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    made = plot_index_df[plot_index_df["plot_status"] == "made"].copy()
    made = made.sort_values(
        ["polarity", "solid_target", "charge", "detector_region", "sector"]
    )

    if made.empty:
        return

    with PdfPages(output_pdf) as pdf:
        for _, row in made.iterrows():
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")

            img = plt.imread(row["plot_path"])
            ax.imshow(img)
            ax.set_title(
                f"{row['polarity']}, LD2+{row['solid_target']}, {row['charge']}, "
                f"{row['detector_region']}, sector={row['sector']}, run {row['representative_run']}"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

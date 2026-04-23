from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rge_vertex.fitting.local_peak import (
    fit_local_peak_poisson,
    local_fit_to_row,
    local_signal_background_model,
)
from rge_vertex.plotting.histograms import HistogramResult, collect_vz_histogram


def _component_fit_arrays(hist: HistogramResult, fit_row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = hist.centers
    edges = hist.edges
    mask = (centers >= fit_row["fit_window_low"]) & (centers <= fit_row["fit_window_high"])
    selected = np.where(mask)[0]
    if selected.size == 0:
        return np.array([]), np.array([]), np.array([])
    local_edges = edges[int(selected[0]) : int(selected[-1]) + 2]
    local_centers = 0.5 * (local_edges[:-1] + local_edges[1:])

    model = local_signal_background_model(
        local_edges,
        signal_model=fit_row.get("signal_model", "gaussian"),
        yield_signal=fit_row["yield_signal"],
        mean=fit_row["mean"],
        sigma=fit_row["sigma"],
        bkg_c0=fit_row["bkg_c0"] if fit_row["background_enabled"] else 0.0,
        bkg_c1=fit_row["bkg_c1"] if fit_row["background_enabled"] else 0.0,
        bkg_c2=fit_row["bkg_c2"] if fit_row["background_enabled"] else 0.0,
        box_width=fit_row.get("box_width"),
    )

    return local_centers, model, mask


def plot_local_ld2_solid_summary(
    hist: HistogramResult,
    fit_rows: list[dict[str, Any]],
    *,
    title: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(hist.centers, hist.counts, where="mid", label=f"data, N={hist.entries}")

    for row in fit_rows:
        comp = row["component"]
        ax.axvspan(row["fit_window_low"], row["fit_window_high"], alpha=0.08)
        if row["fit_status"] in ("good", "bad_fit") and np.isfinite(row.get("mean", np.nan)):
            local_centers, model, _ = _component_fit_arrays(hist, row)
            if local_centers.size > 0:
                if row.get("signal_model") == "box_gaussian" and np.isfinite(row.get("box_width", np.nan)):
                    lbl = f"{comp}: {row['fit_status']}, mean={row['mean']:.3f}, w={row['box_width']:.3f}"
                else:
                    lbl = f"{comp}: {row['fit_status']}, mean={row['mean']:.3f}"
                ax.plot(local_centers, model, label=lbl)
            ax.axvline(row["mean"], linestyle=":", alpha=0.7)
        else:
            expected = row.get("expected_mean_cm", np.nan)
            if np.isfinite(expected):
                ax.axvline(expected, linestyle="--", alpha=0.4, label=f"{comp}: {row['fit_status']}")

    ax.set_title(title)
    ax.set_xlabel("vz [cm]")
    ax.set_ylabel("counts")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _component_row_lookup(fit_rows: list[dict[str, Any]], component: str) -> dict[str, Any] | None:
    matches = [r for r in fit_rows if r["component"] == component]
    return matches[0] if matches else None


def build_category_summary(
    *,
    run: str,
    meta: dict[str, Any],
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
    hist: HistogramResult,
    fit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ld2 = _component_row_lookup(fit_rows, "ld2")
    solid = _component_row_lookup(fit_rows, "solid")

    summary = {
        "run": run,
        "label": meta.get("label", ""),
        "run_class": meta.get("run_class", ""),
        "polarity": meta.get("polarity", ""),
        "solid_target": meta.get("solid_target", ""),
        "target_config": meta.get("target_config", ""),
        "vertex_source": vertex_source,
        "charge": charge,
        "detector_region": detector_region,
        "sector": sector,
        "entries_category": hist.entries,
        "ld2_fit_status": None,
        "solid_fit_status": None,
        "ld2_signal_model": None,
        "solid_signal_model": None,
        "ld2_mean": np.nan,
        "ld2_mean_error": np.nan,
        "ld2_sigma": np.nan,
        "ld2_sigma_error": np.nan,
        "ld2_box_width": np.nan,
        "ld2_box_width_error": np.nan,
        "solid_mean": np.nan,
        "solid_mean_error": np.nan,
        "solid_sigma": np.nan,
        "solid_sigma_error": np.nan,
        "solid_box_width": np.nan,
        "solid_box_width_error": np.nan,
        "ld2_deviance_ndof": np.nan,
        "solid_deviance_ndof": np.nan,
        "mean_gap_solid_minus_ld2": np.nan,
        "both_good": False,
    }

    if ld2 is not None:
        summary["ld2_fit_status"] = ld2["fit_status"]
        summary["ld2_signal_model"] = ld2.get("signal_model")
        summary["ld2_mean"] = ld2.get("mean", np.nan)
        summary["ld2_mean_error"] = ld2.get("mean_error", np.nan)
        summary["ld2_sigma"] = ld2.get("sigma", np.nan)
        summary["ld2_sigma_error"] = ld2.get("sigma_error", np.nan)
        summary["ld2_box_width"] = ld2.get("box_width", np.nan)
        summary["ld2_box_width_error"] = ld2.get("box_width_error", np.nan)
        summary["ld2_deviance_ndof"] = ld2.get("deviance_ndof", np.nan)

    if solid is not None:
        summary["solid_fit_status"] = solid["fit_status"]
        summary["solid_signal_model"] = solid.get("signal_model")
        summary["solid_mean"] = solid.get("mean", np.nan)
        summary["solid_mean_error"] = solid.get("mean_error", np.nan)
        summary["solid_sigma"] = solid.get("sigma", np.nan)
        summary["solid_sigma_error"] = solid.get("sigma_error", np.nan)
        summary["solid_box_width"] = solid.get("box_width", np.nan)
        summary["solid_box_width_error"] = solid.get("box_width_error", np.nan)
        summary["solid_deviance_ndof"] = solid.get("deviance_ndof", np.nan)

    if (
        ld2 is not None
        and solid is not None
        and ld2.get("fit_status") == "good"
        and solid.get("fit_status") == "good"
        and np.isfinite(ld2.get("mean", np.nan))
        and np.isfinite(solid.get("mean", np.nan))
    ):
        summary["mean_gap_solid_minus_ld2"] = solid["mean"] - ld2["mean"]
        summary["both_good"] = True

    return summary


def fit_ld2_solid_local_category(
    *,
    run: str,
    meta: dict[str, Any],
    root_path: str | Path,
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
    config: dict[str, Any],
    output_plot: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hist_cfg = config["histogram"]
    quality_cfg = config.get("quality", {})
    peak_cfg = config.get("peak_finding", {})
    background_cfg = config.get("local_background", {})
    component_cfg = config["components"]

    hist = collect_vz_histogram(
        root_path,
        vertex_source=vertex_source,
        charge=charge,
        detector_region=detector_region,
        sector=sector,
        chi2pid_abs_max=quality_cfg.get("chi2pid_abs_max"),
        bins=int(hist_cfg["bins"]),
        vz_range=(float(hist_cfg["vz_min_cm"]), float(hist_cfg["vz_max_cm"])),
    )

    background_enabled = bool(background_cfg.get("enabled", True))
    fit_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []

    min_entries_category = int(quality_cfg.get("min_entries_per_category", 0) or 0)
    min_entries_window = int(quality_cfg.get("min_entries_per_local_window", 100) or 100)

    if hist.entries < min_entries_category:
        for component, comp_cfg in component_cfg.items():
            row = {
                "run": run,
                "label": meta.get("label", ""),
                "run_class": meta.get("run_class", ""),
                "polarity": meta.get("polarity", ""),
                "solid_target": meta.get("solid_target", ""),
                "target_config": meta.get("target_config", ""),
                "vertex_source": vertex_source,
                "charge": charge,
                "detector_region": detector_region,
                "sector": sector,
                "component": component,
                "signal_model": comp_cfg.get("signal_model", "gaussian"),
                "fit_status": "low_statistics_category",
                "entries_category": hist.entries,
                "entries_window": 0,
                "message": f"entries_category {hist.entries} < min_entries_per_category {min_entries_category}",
                "expected_mean_cm": comp_cfg.get("expected_mean_cm"),
                "search_window_low": comp_cfg.get("search_window_cm", [None, None])[0],
                "search_window_high": comp_cfg.get("search_window_cm", [None, None])[1],
                "fit_window_low": comp_cfg.get("fit_window_cm", [None, None])[0],
                "fit_window_high": comp_cfg.get("fit_window_cm", [None, None])[1],
                "mean": np.nan,
                "mean_error": np.nan,
                "sigma": np.nan,
                "sigma_error": np.nan,
                "yield_signal": np.nan,
                "yield_signal_error": np.nan,
                "box_width": np.nan,
                "box_width_error": np.nan,
                "bkg_c0": np.nan,
                "bkg_c0_error": np.nan,
                "bkg_c1": np.nan,
                "bkg_c1_error": np.nan,
                "bkg_c2": np.nan,
                "bkg_c2_error": np.nan,
                "background_enabled": background_enabled,
            }
            fit_rows.append(row)
            unresolved_rows.append(row)

        category_summary = build_category_summary(run=run, meta=meta, vertex_source=vertex_source, charge=charge, detector_region=detector_region, sector=sector, hist=hist, fit_rows=fit_rows)
        plot_local_ld2_solid_summary(hist, fit_rows, title=f"Run {run}: low-stat category", output_path=output_plot)
        return fit_rows, unresolved_rows, category_summary

    for component, comp_cfg in component_cfg.items():
        fit_result = fit_local_peak_poisson(
            hist.counts,
            hist.edges,
            component=component,
            component_config=comp_cfg,
            entries_category=hist.entries,
            peak_config=peak_cfg,
            min_entries_window=min_entries_window,
            background_enabled=background_enabled,
        )

        row = local_fit_to_row(
            run=run,
            vertex_source=vertex_source,
            charge=charge,
            detector_region=detector_region,
            sector=sector,
            fit_result=fit_result,
            component_config=comp_cfg,
            background_enabled=background_enabled,
        )
        row["label"] = meta.get("label", "")
        row["run_class"] = meta.get("run_class", "")
        row["polarity"] = meta.get("polarity", "")
        row["solid_target"] = meta.get("solid_target", "")
        row["target_config"] = meta.get("target_config", "")
        fit_rows.append(row)
        if row["fit_status"] != "good":
            unresolved_rows.append(row)

    title = (
        f"Run {run}: local LD2 + solid fits\n"
        f"{meta.get('polarity', '')}, LD2+{meta.get('solid_target', '')}, "
        f"{vertex_source}.vz, {charge}, {detector_region}, sector={sector}"
    )
    if detector_region == "central":
        title = (
            f"Run {run}: local LD2 + solid fits\n"
            f"{meta.get('polarity', '')}, LD2+{meta.get('solid_target', '')}, "
            f"{vertex_source}.vz, {charge}, central detector"
        )

    plot_local_ld2_solid_summary(hist, fit_rows, title=title, output_path=output_plot)
    category_summary = build_category_summary(run=run, meta=meta, vertex_source=vertex_source, charge=charge, detector_region=detector_region, sector=sector, hist=hist, fit_rows=fit_rows)
    return fit_rows, unresolved_rows, category_summary


def save_ld2_solid_local_outputs(
    *,
    fit_rows: list[dict[str, Any]],
    unresolved_rows: list[dict[str, Any]],
    category_summary_rows: list[dict[str, Any]],
    fit_results_csv: str | Path,
    unresolved_csv: str | Path,
    category_summary_csv: str | Path,
) -> None:
    fit_results_csv = Path(fit_results_csv)
    unresolved_csv = Path(unresolved_csv)
    category_summary_csv = Path(category_summary_csv)
    fit_results_csv.parent.mkdir(parents=True, exist_ok=True)
    unresolved_csv.parent.mkdir(parents=True, exist_ok=True)
    category_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit_rows).to_csv(fit_results_csv, index=False)
    pd.DataFrame(unresolved_rows).to_csv(unresolved_csv, index=False)
    pd.DataFrame(category_summary_rows).to_csv(category_summary_csv, index=False)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rge_vertex.fitting.binned_fit import BinnedFitResult, fit_four_gaussians_chi2
from rge_vertex.fitting.models import (
    gaussian_counts_at_centers,
    model_counts,
    poly2_background_counts,
)
from rge_vertex.plotting.histograms import HistogramResult, collect_vz_histogram


@dataclass
class EmptyWireFitOutputs:
    fit_rows: list[dict[str, Any]]
    reference_rows: list[dict[str, Any]]


def _component_model(
    hist: HistogramResult,
    fit_result: BinnedFitResult,
    component_name: str,
) -> np.ndarray:
    widths = np.diff(hist.edges)
    p = fit_result.values
    return gaussian_counts_at_centers(
        hist.centers,
        widths,
        yield_=p[f"yield_{component_name}"],
        mean=p[f"mean_{component_name}"],
        sigma=p[f"sigma_{component_name}"],
    )


def _background_model(hist: HistogramResult, fit_result: BinnedFitResult) -> np.ndarray:
    p = fit_result.values
    return poly2_background_counts(
        hist.centers,
        c0=p.get("bkg_c0", 0.0),
        c1=p.get("bkg_c1", 0.0),
        c2=p.get("bkg_c2", 0.0),
    )


def plot_empty_wire_fit(
    hist: HistogramResult,
    fit_result: BinnedFitResult,
    component_names: list[str],
    *,
    background_enabled: bool,
    title: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(hist.centers, hist.counts, where="mid", label=f"data, N={hist.entries}")

    if fit_result.success and fit_result.values:
        widths = np.diff(hist.edges)
        total_model = model_counts(
            hist.centers,
            widths,
            fit_result.values,
            component_names,
            background_enabled=background_enabled,
        )

        ax.plot(hist.centers, total_model, label=f"total fit, chi2/ndf={fit_result.reduced_chi2:.2f}")

        if background_enabled:
            bkg = _background_model(hist, fit_result)
            ax.plot(hist.centers, bkg, linestyle="-.", label="poly2 background")

        for component in component_names:
            comp_counts = _component_model(hist, fit_result, component)
            mean = fit_result.values[f"mean_{component}"]
            sigma = fit_result.values[f"sigma_{component}"]
            ax.plot(hist.centers, comp_counts, linestyle="--", label=f"{component}: {mean:.3f} ± {sigma:.3f}")
            ax.axvline(mean, linestyle=":", alpha=0.6)

    else:
        ax.text(
            0.02,
            0.95,
            f"Fit failed: {fit_result.message}",
            transform=ax.transAxes,
            va="top",
        )

    ax.set_title(title)
    ax.set_xlabel("vz [cm]")
    ax.set_ylabel("counts")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_fit_rows(
    *,
    run: str,
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
    hist: HistogramResult,
    fit_result: BinnedFitResult,
    component_names: list[str],
    background_enabled: bool,
) -> list[dict[str, Any]]:
    rows = []

    for component in component_names:
        row = {
            "run": run,
            "vertex_source": vertex_source,
            "charge": charge,
            "detector_region": detector_region,
            "sector": sector,
            "component": component,
            "component_type": "gaussian",
            "background_enabled": background_enabled,
            "entries": hist.entries,
            "fit_success": fit_result.success,
            "fit_valid": fit_result.valid,
            "fmin_valid": fit_result.fmin_valid,
            "edm": fit_result.edm,
            "chi2": fit_result.chi2,
            "ndof": fit_result.ndof,
            "reduced_chi2": fit_result.reduced_chi2,
            "message": fit_result.message,
        }

        if fit_result.success and fit_result.values:
            for prefix in ("yield", "mean", "sigma"):
                name = f"{prefix}_{component}"
                row[prefix] = fit_result.values.get(name, np.nan)
                row[f"{prefix}_error"] = fit_result.errors.get(name, np.nan)
        else:
            row.update(
                {
                    "yield": np.nan,
                    "yield_error": np.nan,
                    "mean": np.nan,
                    "mean_error": np.nan,
                    "sigma": np.nan,
                    "sigma_error": np.nan,
                }
            )

        rows.append(row)

    if background_enabled:
        row = {
            "run": run,
            "vertex_source": vertex_source,
            "charge": charge,
            "detector_region": detector_region,
            "sector": sector,
            "component": "background_poly2",
            "component_type": "poly2",
            "background_enabled": True,
            "entries": hist.entries,
            "fit_success": fit_result.success,
            "fit_valid": fit_result.valid,
            "fmin_valid": fit_result.fmin_valid,
            "edm": fit_result.edm,
            "chi2": fit_result.chi2,
            "ndof": fit_result.ndof,
            "reduced_chi2": fit_result.reduced_chi2,
            "message": fit_result.message,
            "yield": np.nan,
            "yield_error": np.nan,
            "mean": np.nan,
            "mean_error": np.nan,
            "sigma": np.nan,
            "sigma_error": np.nan,
        }

        if fit_result.success and fit_result.values:
            for name in ("bkg_c0", "bkg_c1", "bkg_c2"):
                row[name] = fit_result.values.get(name, np.nan)
                row[f"{name}_error"] = fit_result.errors.get(name, np.nan)

        rows.append(row)

    return rows


def build_reference_row(
    *,
    run: str,
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
    hist: HistogramResult,
    fit_result: BinnedFitResult,
    reference_component: str,
    n_sigma: float,
    background_enabled: bool,
) -> dict[str, Any]:
    row = {
        "run": run,
        "vertex_source": vertex_source,
        "charge": charge,
        "detector_region": detector_region,
        "sector": sector,
        "reference_component": reference_component,
        "background_enabled": background_enabled,
        "entries": hist.entries,
        "fit_success": fit_result.success,
        "fit_valid": fit_result.valid,
        "reduced_chi2": fit_result.reduced_chi2,
        "n_sigma": n_sigma,
        "reference_mean": np.nan,
        "reference_mean_error": np.nan,
        "reference_sigma": np.nan,
        "reference_sigma_error": np.nan,
        "reference_low": np.nan,
        "reference_high": np.nan,
        "message": fit_result.message,
    }

    if fit_result.success and fit_result.values:
        mean_key = f"mean_{reference_component}"
        sigma_key = f"sigma_{reference_component}"

        mean = fit_result.values.get(mean_key, np.nan)
        sigma = abs(fit_result.values.get(sigma_key, np.nan))

        row["reference_mean"] = mean
        row["reference_mean_error"] = fit_result.errors.get(mean_key, np.nan)
        row["reference_sigma"] = sigma
        row["reference_sigma_error"] = fit_result.errors.get(sigma_key, np.nan)
        row["reference_low"] = mean - n_sigma * sigma
        row["reference_high"] = mean + n_sigma * sigma

    return row


def fit_empty_wire_category(
    *,
    run: str,
    root_path: str | Path,
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
    config: dict[str, Any],
    output_plot: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hist_cfg = config["histogram"]
    quality_cfg = config.get("quality", {})
    component_cfg = config["components"]
    reference_cfg = config["reference_foil"]
    background_cfg = config.get("background", {})
    background_enabled = bool(background_cfg.get("enabled", False))

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

    component_names = list(component_cfg.keys())
    min_entries = int(quality_cfg.get("min_entries", 0) or 0)

    if hist.entries < min_entries:
        n_parameters = 3 * len(component_names) + (3 if background_enabled else 0)
        fit_result = BinnedFitResult(
            success=False,
            valid=False,
            fmin_valid=False,
            edm=None,
            chi2=np.nan,
            ndof=int(len(hist.counts) - n_parameters),
            reduced_chi2=np.nan,
            values={},
            errors={},
            limits={},
            message=f"Skipped: entries {hist.entries} < min_entries {min_entries}",
        )
    else:
        fit_result = fit_four_gaussians_chi2(
            hist.counts,
            hist.edges,
            component_cfg,
            background_config=background_cfg,
        )

    title = (
        f"Run {run}: empty+wire fit\\n"
        f"{vertex_source}.vz, {charge}, {detector_region}, sector={sector}"
    )
    if detector_region == "central":
        title = (
            f"Run {run}: empty+wire fit\\n"
            f"{vertex_source}.vz, {charge}, central detector"
        )

    plot_empty_wire_fit(
        hist,
        fit_result,
        component_names,
        background_enabled=background_enabled,
        title=title,
        output_path=output_plot,
    )

    fit_rows = build_fit_rows(
        run=run,
        vertex_source=vertex_source,
        charge=charge,
        detector_region=detector_region,
        sector=sector,
        hist=hist,
        fit_result=fit_result,
        component_names=component_names,
        background_enabled=background_enabled,
    )

    reference_row = build_reference_row(
        run=run,
        vertex_source=vertex_source,
        charge=charge,
        detector_region=detector_region,
        sector=sector,
        hist=hist,
        fit_result=fit_result,
        reference_component=reference_cfg["component_name"],
        n_sigma=float(reference_cfg["n_sigma_window"]),
        background_enabled=background_enabled,
    )

    return fit_rows, reference_row


def save_empty_wire_outputs(
    outputs: EmptyWireFitOutputs,
    *,
    fit_results_csv: str | Path,
    reference_csv: str | Path,
) -> None:
    fit_results_csv = Path(fit_results_csv)
    reference_csv = Path(reference_csv)

    fit_results_csv.parent.mkdir(parents=True, exist_ok=True)
    reference_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(outputs.fit_rows).to_csv(fit_results_csv, index=False)
    pd.DataFrame(outputs.reference_rows).to_csv(reference_csv, index=False)

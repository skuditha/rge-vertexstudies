from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Any

import numpy as np
from iminuit import Minuit


def _erf_array(x: np.ndarray) -> np.ndarray:
    return np.vectorize(erf, otypes=[float])(x)


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf_array(z / np.sqrt(2.0)))


def _G(z: np.ndarray) -> np.ndarray:
    return z * normal_cdf(z) + normal_pdf(z)


def gaussian_bin_fractions(edges: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1.0e-12)
    z_hi = (edges[1:] - float(mean)) / (sqrt(2.0) * sigma)
    z_lo = (edges[:-1] - float(mean)) / (sqrt(2.0) * sigma)
    return 0.5 * (_erf_array(z_hi) - _erf_array(z_lo))


def box_gaussian_cdf(x: np.ndarray, mean: float, sigma: float, box_width: float) -> np.ndarray:
    sigma = max(float(sigma), 1.0e-12)
    box_width = max(float(box_width), 1.0e-12)
    half_width = 0.5 * box_width
    z_hi = (np.asarray(x, dtype=float) - float(mean) + half_width) / sigma
    z_lo = (np.asarray(x, dtype=float) - float(mean) - half_width) / sigma
    return sigma * (_G(z_hi) - _G(z_lo)) / box_width


def box_gaussian_bin_fractions(edges: np.ndarray, mean: float, sigma: float, box_width: float) -> np.ndarray:
    return box_gaussian_cdf(edges[1:], mean, sigma, box_width) - box_gaussian_cdf(edges[:-1], mean, sigma, box_width)


def scaled_centers(centers: np.ndarray) -> np.ndarray:
    lo = float(np.min(centers))
    hi = float(np.max(centers))
    if hi <= lo:
        return np.zeros_like(centers, dtype=float)
    return 2.0 * (centers - lo) / (hi - lo) - 1.0


def quadratic_background_counts(centers: np.ndarray, c0: float, c1: float, c2: float) -> np.ndarray:
    u = scaled_centers(centers)
    return float(c0) + float(c1) * u + float(c2) * u * u


def local_signal_background_model(
    edges: np.ndarray,
    *,
    signal_model: str,
    yield_signal: float,
    mean: float,
    sigma: float,
    bkg_c0: float,
    bkg_c1: float,
    bkg_c2: float,
    box_width: float | None = None,
) -> np.ndarray:
    centers = 0.5 * (edges[:-1] + edges[1:])

    if signal_model == "gaussian":
        signal = float(yield_signal) * gaussian_bin_fractions(edges, mean, sigma)
    elif signal_model == "box_gaussian":
        if box_width is None:
            raise ValueError("box_width is required for signal_model='box_gaussian'")
        signal = float(yield_signal) * box_gaussian_bin_fractions(edges, mean, sigma, box_width)
    else:
        raise ValueError(f"Unsupported signal_model: {signal_model}")

    background = quadratic_background_counts(centers, bkg_c0, bkg_c1, bkg_c2)
    return signal + background


def local_gaussian_poly2_model(
    edges: np.ndarray,
    *,
    yield_signal: float,
    mean: float,
    sigma: float,
    bkg_c0: float,
    bkg_c1: float,
    bkg_c2: float,
) -> np.ndarray:
    return local_signal_background_model(
        edges,
        signal_model="gaussian",
        yield_signal=yield_signal,
        mean=mean,
        sigma=sigma,
        bkg_c0=bkg_c0,
        bkg_c1=bkg_c1,
        bkg_c2=bkg_c2,
    )


def poisson_nll(observed: np.ndarray, expected: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if np.any(~np.isfinite(expected)) or np.any(expected <= 0.0):
        return 1.0e30
    return float(np.sum(expected - observed * np.log(expected)))


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if np.any(~np.isfinite(expected)) or np.any(expected <= 0.0):
        return float("nan")
    terms = np.zeros_like(observed, dtype=float)
    mask = observed > 0
    terms[mask] = observed[mask] * np.log(observed[mask] / expected[mask])
    return float(2.0 * np.sum(terms - observed + expected))


@dataclass
class PeakCandidate:
    found: bool
    center: float
    height: float
    prominence: float
    message: str = ""


@dataclass
class LocalPeakFitResult:
    component: str
    fit_status: str
    fit_success: bool
    fit_valid: bool
    fmin_valid: bool
    entries_category: int
    entries_window: int
    peak_found: bool
    peak_center: float
    peak_height: float
    peak_prominence: float
    nll: float
    deviance: float
    ndof: int
    deviance_ndof: float
    values: dict[str, float]
    errors: dict[str, float]
    message: str = ""


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    width = int(width)
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_peak_in_window(
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    search_window: tuple[float, float],
    expected_mean: float,
    smoothing_bins: int = 5,
    min_peak_counts: float = 20.0,
    min_relative_prominence: float = 0.04,
) -> PeakCandidate:
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lo, hi = search_window
    mask = (centers >= lo) & (centers <= hi)
    if np.sum(mask) < 3:
        return PeakCandidate(False, float("nan"), float("nan"), float("nan"), "search window has too few bins")

    local_centers = centers[mask]
    local_counts = counts[mask]
    smooth = moving_average(local_counts, smoothing_bins)

    candidate_indices = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            candidate_indices.append(i)

    if not candidate_indices:
        return PeakCandidate(False, float("nan"), float("nan"), float("nan"), "no local maximum found")

    baseline = float(np.percentile(smooth, 20))
    max_smooth = max(float(np.max(smooth)), 1.0)
    candidates = []
    for i in candidate_indices:
        height = float(smooth[i])
        prominence = height - baseline
        if height < min_peak_counts:
            continue
        if prominence < min_relative_prominence * max_smooth:
            continue
        distance = abs(float(local_centers[i]) - float(expected_mean))
        candidates.append((distance, -prominence, i, height, prominence))

    if not candidates:
        return PeakCandidate(False, float("nan"), float("nan"), float("nan"), "local maxima failed height/prominence thresholds")

    _, _, idx, height, prominence = sorted(candidates)[0]
    return PeakCandidate(True, float(local_centers[idx]), float(height), float(prominence), "")


def _limit_pair(value, default=(None, None)):
    if value is None:
        return default
    lo, hi = value
    return (None if lo is None else float(lo), None if hi is None else float(hi))


def fit_local_peak_poisson(
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    component: str,
    component_config: dict[str, Any],
    entries_category: int,
    peak_config: dict[str, Any],
    min_entries_window: int = 100,
    background_enabled: bool = True,
) -> LocalPeakFitResult:
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    signal_model = component_config.get("signal_model", "gaussian")
    expected_mean = float(component_config["expected_mean_cm"])
    search_window = tuple(component_config["search_window_cm"])
    fit_window = tuple(component_config["fit_window_cm"])

    peak = find_peak_in_window(
        counts,
        edges,
        search_window=search_window,
        expected_mean=expected_mean,
        smoothing_bins=int(peak_config.get("smoothing_bins", 5)),
        min_peak_counts=float(peak_config.get("min_peak_counts", 20.0)),
        min_relative_prominence=float(peak_config.get("min_relative_prominence", 0.04)),
    )

    fit_mask = (centers >= fit_window[0]) & (centers <= fit_window[1])
    local_counts = counts[fit_mask]
    selected_bin_indices = np.where(fit_mask)[0]
    if selected_bin_indices.size == 0:
        local_edges = np.array([], dtype=float)
    else:
        first = int(selected_bin_indices[0])
        last = int(selected_bin_indices[-1])
        local_edges = edges[first : last + 2]

    entries_window = int(np.sum(local_counts))
    n_base_params = 6 + (1 if signal_model == "box_gaussian" else 0)

    if entries_window < min_entries_window:
        return LocalPeakFitResult(
            component, "low_statistics", False, False, False,
            int(entries_category), entries_window, peak.found, peak.center, peak.height, peak.prominence,
            float("nan"), float("nan"), int(max(len(local_counts)-n_base_params, 0)), float("nan"),
            {}, {}, f"entries_window {entries_window} < min_entries_window {min_entries_window}"
        )

    if not peak.found:
        return LocalPeakFitResult(
            component, "unresolved_peak", False, False, False,
            int(entries_category), entries_window, False, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), int(max(len(local_counts)-n_base_params, 0)), float("nan"),
            {}, {}, peak.message
        )

    initial_mean = peak.center
    initial_sigma = float(component_config.get("initial_sigma_cm", 0.2))
    sigma_lo, sigma_hi = _limit_pair(component_config.get("sigma_bounds_cm"), default=(0.03, 1.50))
    initial_box_width = float(component_config.get("initial_box_width_cm", 2.0))
    box_width_lo, box_width_hi = _limit_pair(component_config.get("box_width_bounds_cm"), default=(0.5, 4.0))

    positive_counts = local_counts[local_counts > 0]
    bkg_c0_initial = max(float(np.percentile(positive_counts, 15)) if positive_counts.size else 1.0, 1.0)
    signal_yield_initial = max(float(np.sum(local_counts) - bkg_c0_initial * len(local_counts)), 1.0)
    if not background_enabled:
        bkg_c0_initial = 0.0

    try:
        if signal_model == "box_gaussian":
            def nll_func(yield_signal, mean, sigma, bkg_c0, bkg_c1, bkg_c2, box_width):
                mu = local_signal_background_model(
                    local_edges,
                    signal_model=signal_model,
                    yield_signal=yield_signal,
                    mean=mean,
                    sigma=sigma,
                    bkg_c0=bkg_c0 if background_enabled else 0.0,
                    bkg_c1=bkg_c1 if background_enabled else 0.0,
                    bkg_c2=bkg_c2 if background_enabled else 0.0,
                    box_width=box_width,
                )
                return poisson_nll(local_counts, mu)

            m = Minuit(
                nll_func,
                yield_signal=signal_yield_initial,
                mean=initial_mean,
                sigma=initial_sigma,
                bkg_c0=bkg_c0_initial,
                bkg_c1=0.0,
                bkg_c2=0.0,
                box_width=initial_box_width,
            )
            m.limits["box_width"] = (box_width_lo, box_width_hi)

        elif signal_model == "gaussian":
            def nll_func(yield_signal, mean, sigma, bkg_c0, bkg_c1, bkg_c2):
                mu = local_signal_background_model(
                    local_edges,
                    signal_model=signal_model,
                    yield_signal=yield_signal,
                    mean=mean,
                    sigma=sigma,
                    bkg_c0=bkg_c0 if background_enabled else 0.0,
                    bkg_c1=bkg_c1 if background_enabled else 0.0,
                    bkg_c2=bkg_c2 if background_enabled else 0.0,
                )
                return poisson_nll(local_counts, mu)

            m = Minuit(
                nll_func,
                yield_signal=signal_yield_initial,
                mean=initial_mean,
                sigma=initial_sigma,
                bkg_c0=bkg_c0_initial,
                bkg_c1=0.0,
                bkg_c2=0.0,
            )
        else:
            raise ValueError(f"Unsupported signal_model: {signal_model}")

        m.errordef = Minuit.LIKELIHOOD
        m.limits["yield_signal"] = (0.0, None)
        m.limits["mean"] = search_window
        m.limits["sigma"] = (sigma_lo, sigma_hi)
        m.limits["bkg_c0"] = (0.0, None)

        max_count = max(float(np.max(local_counts)), 1.0)
        m.limits["bkg_c1"] = (-5.0 * max_count, 5.0 * max_count)
        m.limits["bkg_c2"] = (-5.0 * max_count, 5.0 * max_count)

        if not background_enabled:
            m.fixed["bkg_c0"] = True
            m.fixed["bkg_c1"] = True
            m.fixed["bkg_c2"] = True

        m.migrad()
        m.hesse()

        values = {name: float(m.values[name]) for name in m.parameters}
        errors = {name: float(m.errors[name]) for name in m.parameters}

        mu = local_signal_background_model(
            local_edges,
            signal_model=signal_model,
            yield_signal=values["yield_signal"],
            mean=values["mean"],
            sigma=values["sigma"],
            bkg_c0=values["bkg_c0"] if background_enabled else 0.0,
            bkg_c1=values["bkg_c1"] if background_enabled else 0.0,
            bkg_c2=values["bkg_c2"] if background_enabled else 0.0,
            box_width=values.get("box_width"),
        )
        dev = poisson_deviance(local_counts, mu)
        n_free = sum(1 for name in m.parameters if not m.fixed[name])
        ndof = int(len(local_counts) - n_free)
        dev_ndof = dev / ndof if ndof > 0 else float("nan")
        status = "good" if bool(m.valid) and np.isfinite(dev_ndof) else "bad_fit"

        return LocalPeakFitResult(
            component, status, True, bool(m.valid), bool(m.fmin.is_valid),
            int(entries_category), entries_window, True, peak.center, peak.height, peak.prominence,
            float(m.fval), float(dev), ndof, float(dev_ndof), values, errors, ""
        )

    except Exception as exc:
        return LocalPeakFitResult(
            component, "bad_fit", False, False, False,
            int(entries_category), entries_window, True, peak.center, peak.height, peak.prominence,
            float("nan"), float("nan"), int(max(len(local_counts)-n_base_params, 0)), float("nan"),
            {}, {}, str(exc)
        )


def local_fit_to_row(
    *,
    run: str,
    vertex_source: str,
    charge: str,
    detector_region: str,
    sector: int | str,
    fit_result: LocalPeakFitResult,
    component_config: dict[str, Any],
    background_enabled: bool,
) -> dict[str, Any]:
    values = fit_result.values
    errors = fit_result.errors
    signal_model = component_config.get("signal_model", "gaussian")
    return {
        "run": run,
        "vertex_source": vertex_source,
        "charge": charge,
        "detector_region": detector_region,
        "sector": sector,
        "component": fit_result.component,
        "signal_model": signal_model,
        "fit_status": fit_result.fit_status,
        "fit_success": fit_result.fit_success,
        "fit_valid": fit_result.fit_valid,
        "fmin_valid": fit_result.fmin_valid,
        "entries_category": fit_result.entries_category,
        "entries_window": fit_result.entries_window,
        "peak_found": fit_result.peak_found,
        "peak_center": fit_result.peak_center,
        "peak_height": fit_result.peak_height,
        "peak_prominence": fit_result.peak_prominence,
        "nll": fit_result.nll,
        "deviance": fit_result.deviance,
        "ndof": fit_result.ndof,
        "deviance_ndof": fit_result.deviance_ndof,
        "background_enabled": background_enabled,
        "expected_mean_cm": component_config.get("expected_mean_cm"),
        "search_window_low": component_config.get("search_window_cm", [None, None])[0],
        "search_window_high": component_config.get("search_window_cm", [None, None])[1],
        "fit_window_low": component_config.get("fit_window_cm", [None, None])[0],
        "fit_window_high": component_config.get("fit_window_cm", [None, None])[1],
        "mean": values.get("mean", float("nan")),
        "mean_error": errors.get("mean", float("nan")),
        "sigma": values.get("sigma", float("nan")),
        "sigma_error": errors.get("sigma", float("nan")),
        "yield_signal": values.get("yield_signal", float("nan")),
        "yield_signal_error": errors.get("yield_signal", float("nan")),
        "box_width": values.get("box_width", float("nan")),
        "box_width_error": errors.get("box_width", float("nan")),
        "bkg_c0": values.get("bkg_c0", float("nan")),
        "bkg_c0_error": errors.get("bkg_c0", float("nan")),
        "bkg_c1": values.get("bkg_c1", float("nan")),
        "bkg_c1_error": errors.get("bkg_c1", float("nan")),
        "bkg_c2": values.get("bkg_c2", float("nan")),
        "bkg_c2_error": errors.get("bkg_c2", float("nan")),
        "message": fit_result.message,
    }

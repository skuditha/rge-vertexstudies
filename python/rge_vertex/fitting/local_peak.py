from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Any

import numpy as np
from iminuit import Minuit


def _erf_array(x: np.ndarray) -> np.ndarray:
    return np.vectorize(erf, otypes=[float])(x)


def gaussian_bin_fractions(edges: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Exact Gaussian probability in each histogram bin."""
    sigma = max(float(sigma), 1.0e-12)
    z_hi = (edges[1:] - float(mean)) / (sqrt(2.0) * sigma)
    z_lo = (edges[:-1] - float(mean)) / (sqrt(2.0) * sigma)
    return 0.5 * (_erf_array(z_hi) - _erf_array(z_lo))


def scaled_centers(centers: np.ndarray) -> np.ndarray:
    lo = float(np.min(centers))
    hi = float(np.max(centers))
    if hi <= lo:
        return np.zeros_like(centers, dtype=float)
    return 2.0 * (centers - lo) / (hi - lo) - 1.0


def quadratic_background_counts(centers: np.ndarray, c0: float, c1: float, c2: float) -> np.ndarray:
    """Quadratic background in counts per bin, using scaled coordinate u in [-1, 1]."""
    u = scaled_centers(centers)
    return float(c0) + float(c1) * u + float(c2) * u * u


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
    centers = 0.5 * (edges[:-1] + edges[1:])
    signal = float(yield_signal) * gaussian_bin_fractions(edges, mean, sigma)
    background = quadratic_background_counts(centers, bkg_c0, bkg_c1, bkg_c2)
    return signal + background


def poisson_nll(observed: np.ndarray, expected: np.ndarray) -> float:
    """Poisson negative log-likelihood, dropping constant factorial term."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)

    if np.any(~np.isfinite(expected)) or np.any(expected <= 0.0):
        return 1.0e30

    return float(np.sum(expected - observed * np.log(expected)))


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    """Poisson deviance, useful as a chi2-like goodness-of-fit metric."""
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
        return PeakCandidate(
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            "local maxima failed height/prominence thresholds",
        )

    # Choose the acceptable peak closest to the expected physical location.
    _, _, idx, height, prominence = sorted(candidates)[0]
    return PeakCandidate(
        True,
        float(local_centers[idx]),
        float(height),
        float(prominence),
        "",
    )


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
    local_edges_full = edges

    # Select edges corresponding to selected bins.
    selected_bin_indices = np.where(fit_mask)[0]
    if selected_bin_indices.size == 0:
        local_edges = np.array([], dtype=float)
    else:
        first = int(selected_bin_indices[0])
        last = int(selected_bin_indices[-1])
        local_edges = local_edges_full[first : last + 2]

    entries_window = int(np.sum(local_counts))

    if entries_window < min_entries_window:
        return LocalPeakFitResult(
            component=component,
            fit_status="low_statistics",
            fit_success=False,
            fit_valid=False,
            fmin_valid=False,
            entries_category=int(entries_category),
            entries_window=entries_window,
            peak_found=peak.found,
            peak_center=peak.center,
            peak_height=peak.height,
            peak_prominence=peak.prominence,
            nll=float("nan"),
            deviance=float("nan"),
            ndof=int(max(len(local_counts) - 6, 0)),
            deviance_ndof=float("nan"),
            values={},
            errors={},
            message=f"entries_window {entries_window} < min_entries_window {min_entries_window}",
        )

    if not peak.found:
        return LocalPeakFitResult(
            component=component,
            fit_status="unresolved_peak",
            fit_success=False,
            fit_valid=False,
            fmin_valid=False,
            entries_category=int(entries_category),
            entries_window=entries_window,
            peak_found=False,
            peak_center=float("nan"),
            peak_height=float("nan"),
            peak_prominence=float("nan"),
            nll=float("nan"),
            deviance=float("nan"),
            ndof=int(max(len(local_counts) - 6, 0)),
            deviance_ndof=float("nan"),
            values={},
            errors={},
            message=peak.message,
        )

    initial_mean = peak.center
    initial_sigma = float(component_config.get("initial_sigma_cm", 0.2))
    sigma_lo, sigma_hi = _limit_pair(component_config.get("sigma_bounds_cm"), default=(0.03, 1.50))

    local_centers = 0.5 * (local_edges[:-1] + local_edges[1:])
    bkg_c0_initial = max(float(np.percentile(local_counts[local_counts > 0], 15)) if np.any(local_counts > 0) else 1.0, 1.0)
    signal_yield_initial = max(float(np.sum(local_counts) - bkg_c0_initial * len(local_counts)), 1.0)

    if not background_enabled:
        # Fit with effectively no background. Keep parameters present but fixed.
        bkg_c0_initial = 0.0

    def nll_func(yield_signal, mean, sigma, bkg_c0, bkg_c1, bkg_c2):
        mu = local_gaussian_poly2_model(
            local_edges,
            yield_signal=yield_signal,
            mean=mean,
            sigma=sigma,
            bkg_c0=bkg_c0 if background_enabled else 0.0,
            bkg_c1=bkg_c1 if background_enabled else 0.0,
            bkg_c2=bkg_c2 if background_enabled else 0.0,
        )
        return poisson_nll(local_counts, mu)

    try:
        m = Minuit(
            nll_func,
            yield_signal=signal_yield_initial,
            mean=initial_mean,
            sigma=initial_sigma,
            bkg_c0=bkg_c0_initial,
            bkg_c1=0.0,
            bkg_c2=0.0,
        )
        m.errordef = Minuit.LIKELIHOOD

        m.limits["yield_signal"] = (0.0, None)
        # Keep mean inside the physical search window, not the broader fit window.
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

        mu = local_gaussian_poly2_model(
            local_edges,
            yield_signal=values["yield_signal"],
            mean=values["mean"],
            sigma=values["sigma"],
            bkg_c0=values["bkg_c0"] if background_enabled else 0.0,
            bkg_c1=values["bkg_c1"] if background_enabled else 0.0,
            bkg_c2=values["bkg_c2"] if background_enabled else 0.0,
        )

        dev = poisson_deviance(local_counts, mu)
        n_free = sum(1 for name in m.parameters if not m.fixed[name])
        ndof = int(len(local_counts) - n_free)
        dev_ndof = dev / ndof if ndof > 0 else float("nan")

        status = "good" if bool(m.valid) and np.isfinite(dev_ndof) else "bad_fit"

        return LocalPeakFitResult(
            component=component,
            fit_status=status,
            fit_success=True,
            fit_valid=bool(m.valid),
            fmin_valid=bool(m.fmin.is_valid),
            entries_category=int(entries_category),
            entries_window=entries_window,
            peak_found=True,
            peak_center=peak.center,
            peak_height=peak.height,
            peak_prominence=peak.prominence,
            nll=float(m.fval),
            deviance=float(dev),
            ndof=ndof,
            deviance_ndof=float(dev_ndof),
            values=values,
            errors=errors,
            message="",
        )

    except Exception as exc:
        return LocalPeakFitResult(
            component=component,
            fit_status="bad_fit",
            fit_success=False,
            fit_valid=False,
            fmin_valid=False,
            entries_category=int(entries_category),
            entries_window=entries_window,
            peak_found=True,
            peak_center=peak.center,
            peak_height=peak.height,
            peak_prominence=peak.prominence,
            nll=float("nan"),
            deviance=float("nan"),
            ndof=int(max(len(local_counts) - 6, 0)),
            deviance_ndof=float("nan"),
            values={},
            errors={},
            message=str(exc),
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

    row = {
        "run": run,
        "vertex_source": vertex_source,
        "charge": charge,
        "detector_region": detector_region,
        "sector": sector,
        "component": fit_result.component,
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
        "bkg_c0": values.get("bkg_c0", float("nan")),
        "bkg_c0_error": errors.get("bkg_c0", float("nan")),
        "bkg_c1": values.get("bkg_c1", float("nan")),
        "bkg_c1_error": errors.get("bkg_c1", float("nan")),
        "bkg_c2": values.get("bkg_c2", float("nan")),
        "bkg_c2_error": errors.get("bkg_c2", float("nan")),
        "message": fit_result.message,
    }
    return row

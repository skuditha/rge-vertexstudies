from __future__ import annotations

import numpy as np


def gaussian_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Normalized Gaussian PDF evaluated at x."""
    sigma = max(float(sigma), 1.0e-12)
    arg = (x - float(mean)) / sigma
    return np.exp(-0.5 * arg * arg) / (np.sqrt(2.0 * np.pi) * sigma)


def gaussian_counts_at_centers(
    centers: np.ndarray,
    bin_widths: np.ndarray,
    *,
    yield_: float,
    mean: float,
    sigma: float,
) -> np.ndarray:
    """Approximate expected bin counts using PDF at bin centers."""
    return float(yield_) * gaussian_pdf(centers, mean, sigma) * bin_widths


def scaled_coordinate(centers: np.ndarray) -> np.ndarray:
    """Map bin centers to u in [-1, 1]."""
    lo = float(np.min(centers))
    hi = float(np.max(centers))
    if hi <= lo:
        return np.zeros_like(centers, dtype=float)
    return 2.0 * (centers - lo) / (hi - lo) - 1.0


def poly2_background_counts(
    centers: np.ndarray,
    *,
    c0: float,
    c1: float,
    c2: float,
) -> np.ndarray:
    """Second-order polynomial background in scaled coordinate.

    The coefficients are in counts per bin, not a normalized density.
    """
    u = scaled_coordinate(centers)
    return float(c0) + float(c1) * u + float(c2) * u * u


def multi_gaussian_counts(
    centers: np.ndarray,
    bin_widths: np.ndarray,
    params: dict[str, float],
    component_names: list[str],
) -> np.ndarray:
    """Expected counts for a sum of Gaussian components."""
    model = np.zeros_like(centers, dtype=float)

    for name in component_names:
        model += gaussian_counts_at_centers(
            centers,
            bin_widths,
            yield_=params[f"yield_{name}"],
            mean=params[f"mean_{name}"],
            sigma=params[f"sigma_{name}"],
        )

    return model


def model_counts(
    centers: np.ndarray,
    bin_widths: np.ndarray,
    params: dict[str, float],
    component_names: list[str],
    *,
    background_enabled: bool = False,
) -> np.ndarray:
    """Full model: Gaussian signal components plus optional poly2 background."""
    model = multi_gaussian_counts(centers, bin_widths, params, component_names)

    if background_enabled:
        model += poly2_background_counts(
            centers,
            c0=params.get("bkg_c0", 0.0),
            c1=params.get("bkg_c1", 0.0),
            c2=params.get("bkg_c2", 0.0),
        )

    return model

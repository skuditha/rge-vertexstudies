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
    """Approximate expected bin counts using PDF at bin centers.

    This is sufficient for the first-pass vertex fits with reasonably fine bins.
    If needed later, we can replace this with exact bin integration using erf.
    """
    return float(yield_) * gaussian_pdf(centers, mean, sigma) * bin_widths


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

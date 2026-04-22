from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from iminuit import Minuit

from rge_vertex.fitting.models import model_counts


@dataclass
class BinnedFitResult:
    success: bool
    valid: bool
    fmin_valid: bool
    edm: float | None
    chi2: float
    ndof: int
    reduced_chi2: float
    values: dict[str, float]
    errors: dict[str, float]
    limits: dict[str, tuple[float | None, float | None]]
    message: str = ""


def _yaml_limit_pair(value, default=(None, None)) -> tuple[float | None, float | None]:
    if value is None:
        return default
    if len(value) != 2:
        raise ValueError(f"Limit pair must have length 2, got {value}")
    lo, hi = value
    return (None if lo is None else float(lo), None if hi is None else float(hi))


def estimate_initial_yields(
    counts: np.ndarray,
    centers: np.ndarray,
    component_names: list[str],
    component_config: dict[str, Any],
) -> dict[str, float]:
    """Crude initial yield estimate by partitioning counts around initial means."""
    total = float(np.sum(counts))
    if total <= 0:
        return {name: 1.0 for name in component_names}

    means = np.array([component_config[name]["initial_mean_cm"] for name in component_names], dtype=float)
    order = np.argsort(means)
    sorted_names = [component_names[i] for i in order]
    sorted_means = means[order]

    boundaries = []
    for i in range(len(sorted_means) - 1):
        boundaries.append(0.5 * (sorted_means[i] + sorted_means[i + 1]))

    yields: dict[str, float] = {}
    for i, name in enumerate(sorted_names):
        if i == 0:
            mask = centers <= boundaries[0]
        elif i == len(sorted_names) - 1:
            mask = centers > boundaries[-1]
        else:
            mask = (centers > boundaries[i - 1]) & (centers <= boundaries[i])

        y = float(np.sum(counts[mask]))
        yields[name] = max(y, 1.0)

    return yields


def estimate_background_c0(counts: np.ndarray) -> float:
    """Robust first guess for flat background counts per bin."""
    counts = np.asarray(counts, dtype=float)
    positive = counts[counts > 0]
    if positive.size == 0:
        return 1.0

    # Use a low percentile instead of the median because peak bins dominate.
    return max(float(np.percentile(positive, 15)), 1.0)


def fit_four_gaussians_chi2(
    counts: np.ndarray,
    edges: np.ndarray,
    component_config: dict[str, Any],
    background_config: dict[str, Any] | None = None,
) -> BinnedFitResult:
    """Fit a binned vertex distribution with four Gaussian components.

    Optionally includes a second-order polynomial background:
        bkg_c0 + bkg_c1*u + bkg_c2*u^2
    where u maps the fit range to [-1, 1].
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    component_names = list(component_config.keys())

    background_config = background_config or {}
    background_enabled = bool(background_config.get("enabled", False))
    if background_enabled and background_config.get("shape", "poly2") != "poly2":
        raise ValueError("Only background shape 'poly2' is currently supported.")

    initial_yields = estimate_initial_yields(counts, centers, component_names, component_config)

    names: list[str] = []
    starts: list[float] = []
    limits: dict[str, tuple[float | None, float | None]] = {}

    for comp in component_names:
        yname = f"yield_{comp}"
        mname = f"mean_{comp}"
        sname = f"sigma_{comp}"

        names.extend([yname, mname, sname])
        starts.extend(
            [
                initial_yields[comp],
                float(component_config[comp]["initial_mean_cm"]),
                float(component_config[comp]["initial_sigma_cm"]),
            ]
        )

        limits[yname] = (0.0, None)
        limits[mname] = _yaml_limit_pair(component_config[comp].get("mean_bounds_cm"))
        limits[sname] = _yaml_limit_pair(component_config[comp].get("sigma_bounds_cm"), default=(0.001, None))

    if background_enabled:
        initial_c0 = background_config.get("initial_c0")
        if initial_c0 is None:
            initial_c0 = estimate_background_c0(counts)

        names.extend(["bkg_c0", "bkg_c1", "bkg_c2"])
        starts.extend(
            [
                float(initial_c0),
                float(background_config.get("initial_c1", 0.0)),
                float(background_config.get("initial_c2", 0.0)),
            ]
        )
        limits["bkg_c0"] = _yaml_limit_pair(background_config.get("c0_bounds"), default=(0.0, None))
        limits["bkg_c1"] = _yaml_limit_pair(background_config.get("c1_bounds"), default=(None, None))
        limits["bkg_c2"] = _yaml_limit_pair(background_config.get("c2_bounds"), default=(None, None))

    sigma_obs = np.sqrt(np.maximum(counts, 1.0))

    def chi2_function(*pars):
        p = dict(zip(names, pars))
        model = model_counts(
            centers,
            widths,
            p,
            component_names,
            background_enabled=background_enabled,
        )

        # Penalize pathological parameter choices where the polynomial drives
        # the full expectation non-positive in any bin.
        if np.any(~np.isfinite(model)) or np.any(model <= 0.0):
            return 1.0e30

        return float(np.sum(((counts - model) / sigma_obs) ** 2))

    try:
        minuit = Minuit(chi2_function, *starts, name=names)
        minuit.errordef = Minuit.LEAST_SQUARES

        for pname, lim in limits.items():
            minuit.limits[pname] = lim

        minuit.migrad()
        minuit.hesse()

        values = {name: float(minuit.values[name]) for name in names}
        errors = {name: float(minuit.errors[name]) for name in names}
        chi2 = float(minuit.fval)
        ndof = int(len(counts) - len(names))
        reduced = chi2 / ndof if ndof > 0 else float("nan")

        return BinnedFitResult(
            success=True,
            valid=bool(minuit.valid),
            fmin_valid=bool(minuit.fmin.is_valid),
            edm=float(minuit.fmin.edm),
            chi2=chi2,
            ndof=ndof,
            reduced_chi2=reduced,
            values=values,
            errors=errors,
            limits=limits,
            message="",
        )
    except Exception as exc:
        return BinnedFitResult(
            success=False,
            valid=False,
            fmin_valid=False,
            edm=None,
            chi2=float("nan"),
            ndof=int(len(counts) - len(names)),
            reduced_chi2=float("nan"),
            values={},
            errors={},
            limits=limits,
            message=str(exc),
        )

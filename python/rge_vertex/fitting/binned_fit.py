from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from iminuit import Minuit

from rge_vertex.fitting.models import multi_gaussian_counts


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

    # Boundaries halfway between initial means.
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


def fit_four_gaussians_chi2(
    counts: np.ndarray,
    edges: np.ndarray,
    component_config: dict[str, Any],
) -> BinnedFitResult:
    """Fit a binned vertex distribution with configured Gaussian components."""
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    component_names = list(component_config.keys())

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
        limits[mname] = tuple(component_config[comp].get("mean_bounds_cm", (None, None)))
        limits[sname] = tuple(component_config[comp].get("sigma_bounds_cm", (0.001, None)))

    sigma_obs = np.sqrt(np.maximum(counts, 1.0))

    def chi2_function(*pars):
        p = dict(zip(names, pars))
        model = multi_gaussian_counts(centers, widths, p, component_names)
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

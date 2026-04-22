from __future__ import annotations
import numpy as np

DETECTOR_REGION_IDS = {"other": 0, "forward": 1, "central": 2}
CHARGE_VALUES = {"negative": -1, "positive": 1}

VALID_VERTEX_SOURCES = {"particle", "ftrack", "hybrid"}


def finite_mask(values) -> np.ndarray:
    return np.isfinite(np.asarray(values))


def get_vz_values(arrays, vertex_source: str) -> np.ndarray:
    """Return the vz array for a requested vertex source.

    vertex_source options:
      particle:
        Always use REC::Particle.vz.

      ftrack:
        Use REC::FTrack.vz only. Entries without FTrack remain NaN
        and are removed by vertex_source_mask.

      hybrid:
        Use REC::FTrack.vz when available and finite.
        Fall back to REC::Particle.vz otherwise.

    This implements the group-requested option:
      FTrack when available, Particle when not.
    """
    if vertex_source == "particle":
        return np.asarray(arrays["vz_particle"], dtype=float)

    if vertex_source == "ftrack":
        return np.asarray(arrays["vz_ftrack"], dtype=float)

    if vertex_source == "hybrid":
        vz_particle = np.asarray(arrays["vz_particle"], dtype=float)
        vz_ftrack = np.asarray(arrays["vz_ftrack"], dtype=float)
        has_ftrack = np.asarray(arrays["has_ftrack"]) == 1
        use_ftrack = has_ftrack & np.isfinite(vz_ftrack)
        return np.where(use_ftrack, vz_ftrack, vz_particle)

    raise ValueError(f"Unknown vertex source: {vertex_source}. Valid options are {sorted(VALID_VERTEX_SOURCES)}")


def charge_mask(arrays, charge: str | int | None) -> np.ndarray:
    if charge is None or charge == "all":
        return np.ones(len(arrays["charge"]), dtype=bool)
    value = CHARGE_VALUES[charge] if isinstance(charge, str) else int(charge)
    return np.asarray(arrays["charge"]) == value


def detector_region_mask(arrays, detector_region: str | int | None) -> np.ndarray:
    if detector_region is None or detector_region == "all":
        return np.ones(len(arrays["detector_region"]), dtype=bool)
    value = DETECTOR_REGION_IDS[detector_region] if isinstance(detector_region, str) else int(detector_region)
    return np.asarray(arrays["detector_region"]) == value


def detector_region_is_central(detector_region: str | int | None) -> bool:
    if detector_region is None or detector_region == "all":
        return False
    if isinstance(detector_region, str):
        return detector_region == "central"
    return int(detector_region) == DETECTOR_REGION_IDS["central"]


def sector_mask(arrays, sector: int | str | None) -> np.ndarray:
    if sector is None or sector == "all":
        return np.ones(len(arrays["sector"]), dtype=bool)
    return np.asarray(arrays["sector"]) == int(sector)


def pid_mask(arrays, pid: int | None) -> np.ndarray:
    if pid is None:
        return np.ones(len(arrays["pid"]), dtype=bool)
    return np.asarray(arrays["pid"]) == int(pid)


def vertex_source_mask(arrays, vertex_source: str) -> np.ndarray:
    if vertex_source not in VALID_VERTEX_SOURCES:
        raise ValueError(f"Unknown vertex source: {vertex_source}. Valid options are {sorted(VALID_VERTEX_SOURCES)}")

    if vertex_source == "ftrack":
        return (np.asarray(arrays["has_ftrack"]) == 1) & finite_mask(arrays["vz_ftrack"])

    if vertex_source == "particle":
        return finite_mask(arrays["vz_particle"])

    if vertex_source == "hybrid":
        return finite_mask(get_vz_values(arrays, "hybrid"))

    raise AssertionError("unreachable")


def combined_track_mask(
    arrays,
    *,
    vertex_source: str,
    charge=None,
    detector_region=None,
    sector=None,
    pid=None,
    chi2pid_abs_max=None,
) -> np.ndarray:
    mask = vertex_source_mask(arrays, vertex_source)
    mask &= charge_mask(arrays, charge)
    mask &= detector_region_mask(arrays, detector_region)

    # The central detector is treated as one sectorless entity.
    # If a caller accidentally passes sector=1..6 with detector_region="central",
    # ignore that sector request instead of splitting central tracks.
    effective_sector = "all" if detector_region_is_central(detector_region) else sector
    mask &= sector_mask(arrays, effective_sector)

    mask &= pid_mask(arrays, pid)

    if chi2pid_abs_max is not None:
        chi2 = np.asarray(arrays["chi2pid"])
        mask &= np.isfinite(chi2)
        mask &= np.abs(chi2) < float(chi2pid_abs_max)

    return mask

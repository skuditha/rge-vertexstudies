from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _norm_pid_category(row: pd.Series) -> str:
    if "pid_category" not in row or pd.isna(row.get("pid_category", np.nan)):
        return "charge_only"
    return str(row["pid_category"])


def _load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def load_reference_table(reference_csv: str | Path) -> pd.DataFrame:
    df = _load_csv(reference_csv).copy()
    if df.empty:
        return df
    if "run" in df.columns:
        df["run"] = df["run"].map(_norm_run)
    df["charge"] = df["charge"].astype(str)
    df["detector_region"] = df["detector_region"].astype(str)
    df["sector_norm"] = [_norm_sector(det, sec) for det, sec in zip(df["detector_region"], df["sector"])]
    if "vertex_source" in df.columns:
        df["vertex_source"] = df["vertex_source"].astype(str)
    else:
        df["vertex_source"] = "particle"
    return df


def load_category_summary(category_summary_csv: str | Path) -> pd.DataFrame:
    df = _load_csv(category_summary_csv).copy()
    if df.empty:
        return df
    df["run"] = df["run"].map(_norm_run)
    df["charge"] = df["charge"].astype(str)
    df["detector_region"] = df["detector_region"].astype(str)
    df["sector_norm"] = [_norm_sector(det, sec) for det, sec in zip(df["detector_region"], df["sector"])]
    df["vertex_source"] = df["vertex_source"].astype(str)
    df["pid_category_norm"] = df.apply(_norm_pid_category, axis=1)
    return df


def _reference_aggregates(ref_df: pd.DataFrame, *, vertex_sources: list[str], charge: str, detector_region: str, sector_norm: str) -> dict[str, Any] | None:
    attempts = []
    for vs in vertex_sources:
        attempts.append((vs, detector_region, sector_norm, "exact"))
    if detector_region == "forward" and sector_norm != "all":
        for vs in vertex_sources:
            attempts.append((vs, detector_region, "all", "forward_all_sector"))
    for vs in vertex_sources:
        attempts.append((vs, detector_region, "all", "detector_region_all"))

    for vertex_source, det, sec, match_level in attempts:
        sub = ref_df[
            (ref_df["vertex_source"] == vertex_source)
            & (ref_df["charge"] == charge)
            & (ref_df["detector_region"] == det)
            & (ref_df["sector_norm"] == sec)
            & np.isfinite(pd.to_numeric(ref_df["reference_low"], errors="coerce"))
            & np.isfinite(pd.to_numeric(ref_df["reference_high"], errors="coerce"))
        ].copy()
        if sub.empty:
            continue
        return {
            "reference_vertex_source": vertex_source,
            "reference_match_level": match_level,
            "reference_detector_region": det,
            "reference_sector": sec,
            "reference_mean": float(np.nanmedian(pd.to_numeric(sub["reference_mean"], errors="coerce"))),
            "reference_sigma": float(np.nanmedian(pd.to_numeric(sub["reference_sigma"], errors="coerce"))),
            "reference_low": float(np.nanmedian(pd.to_numeric(sub["reference_low"], errors="coerce"))),
            "reference_high": float(np.nanmedian(pd.to_numeric(sub["reference_high"], errors="coerce"))),
        }
    return None


def _ld2_raw_bounds(row: pd.Series, ld2_n_sigma: float) -> tuple[float, float]:
    mean = float(row["ld2_mean"])
    sigma = abs(float(row["ld2_sigma"]))
    signal_model = row.get("ld2_signal_model", "gaussian")
    box_width = row.get("ld2_box_width", np.nan)
    if signal_model == "box_gaussian" and np.isfinite(box_width):
        half_width = 0.5 * float(box_width)
        return (mean - half_width - ld2_n_sigma * sigma, mean + half_width + ld2_n_sigma * sigma)
    return (mean - ld2_n_sigma * sigma, mean + ld2_n_sigma * sigma)


def _solid_raw_bounds(row: pd.Series, solid_n_sigma: float) -> tuple[float, float]:
    mean = float(row["solid_mean"])
    sigma = abs(float(row["solid_sigma"]))
    signal_model = row.get("solid_signal_model", "gaussian")
    box_width = row.get("solid_box_width", np.nan)
    if signal_model == "box_gaussian" and np.isfinite(box_width):
        half_width = 0.5 * float(box_width)
        return (mean - half_width - solid_n_sigma * sigma, mean + half_width + solid_n_sigma * sigma)
    return (mean - solid_n_sigma * sigma, mean + solid_n_sigma * sigma)


def build_all_candidate_cuts(
    *,
    category_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    recommended_vertex_source: str,
    accepted_pid_categories: list[str],
    reference_vertex_source_fallback_order: list[str],
    ld2_n_sigma: float,
    solid_n_sigma: float,
    clip_to_reference: bool,
    safety_margin_cm: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    df = category_df.copy()
    df = df[df["vertex_source"] == recommended_vertex_source]
    df = df[df["pid_category_norm"].isin(accepted_pid_categories)]

    for _, row in df.iterrows():
        out = {
            "run": row["run"],
            "label": row.get("label", ""),
            "polarity": row.get("polarity", ""),
            "solid_target": row.get("solid_target", ""),
            "target_config": row.get("target_config", ""),
            "vertex_source": row.get("vertex_source", ""),
            "charge": row.get("charge", ""),
            "detector_region": row.get("detector_region", ""),
            "sector": row.get("sector_norm", ""),
            "pid_category": row.get("pid_category_norm", "charge_only"),
            "entries_category": row.get("entries_category", np.nan),
            "ld2_fit_status": row.get("ld2_fit_status", ""),
            "solid_fit_status": row.get("solid_fit_status", ""),
            "both_good": bool(row.get("both_good", False)),
            "candidate_status": "unknown",
            "candidate_reason": "",
        }

        ref = _reference_aggregates(
            reference_df,
            vertex_sources=reference_vertex_source_fallback_order,
            charge=str(row["charge"]),
            detector_region=str(row["detector_region"]),
            sector_norm=str(row["sector_norm"]),
        )
        if ref is not None:
            out.update(ref)
        else:
            out.update({
                "reference_vertex_source": None,
                "reference_match_level": "missing",
                "reference_detector_region": row["detector_region"],
                "reference_sector": row["sector_norm"],
                "reference_mean": np.nan,
                "reference_sigma": np.nan,
                "reference_low": np.nan,
                "reference_high": np.nan,
            })

        if not bool(row.get("both_good", False)):
            out["candidate_status"] = "bad_fit"
            out["candidate_reason"] = "ld2 and/or solid fit not good"
            rows.append(out)
            continue

        if not (
            np.isfinite(row.get("ld2_mean", np.nan))
            and np.isfinite(row.get("ld2_sigma", np.nan))
            and np.isfinite(row.get("solid_mean", np.nan))
            and np.isfinite(row.get("solid_sigma", np.nan))
        ):
            out["candidate_status"] = "bad_fit"
            out["candidate_reason"] = "missing fit parameters"
            rows.append(out)
            continue

        ld2_raw_low, ld2_raw_high = _ld2_raw_bounds(row, ld2_n_sigma)
        solid_raw_low, solid_raw_high = _solid_raw_bounds(row, solid_n_sigma)

        out["ld2_raw_low"] = ld2_raw_low
        out["ld2_raw_high"] = ld2_raw_high
        out["solid_raw_low"] = solid_raw_low
        out["solid_raw_high"] = solid_raw_high

        ld2_cut_low, ld2_cut_high = ld2_raw_low, ld2_raw_high
        solid_cut_low, solid_cut_high = solid_raw_low, solid_raw_high

        if clip_to_reference:
            if not (np.isfinite(out["reference_low"]) and np.isfinite(out["reference_high"])):
                out["candidate_status"] = "missing_reference"
                out["candidate_reason"] = "reference forbidden region unavailable"
                rows.append(out)
                continue
            ld2_cut_high = min(ld2_cut_high, float(out["reference_low"]) - safety_margin_cm)
            solid_cut_low = max(solid_cut_low, float(out["reference_high"]) + safety_margin_cm)

        out["ld2_cut_low"] = ld2_cut_low
        out["ld2_cut_high"] = ld2_cut_high
        out["solid_cut_low"] = solid_cut_low
        out["solid_cut_high"] = solid_cut_high

        if not (ld2_cut_low < ld2_cut_high):
            out["candidate_status"] = "invalid_ld2_cut"
            out["candidate_reason"] = "LD2 cut collapsed after clipping"
            rows.append(out)
            continue
        if not (solid_cut_low < solid_cut_high):
            out["candidate_status"] = "invalid_solid_cut"
            out["candidate_reason"] = "Solid cut collapsed after clipping"
            rows.append(out)
            continue

        out["candidate_status"] = "good"
        out["candidate_reason"] = "good fits and valid clipping"
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        sort_cols = [c for c in ["polarity", "solid_target", "charge", "detector_region", "sector", "run"] if c in out_df.columns]
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
    return out_df


def _aggregate_stat(values: pd.Series, statistic: str) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if statistic == "median":
        return float(np.median(arr))
    if statistic == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unsupported aggregation statistic: {statistic}")


def _choose_fallback_subset(
    candidates: pd.DataFrame,
    *,
    polarity: str,
    solid_target: str,
    charge: str,
    detector_region: str,
    sector: str,
    allow_sector_all_fallback: bool,
    allow_target_broadening: bool,
    allow_polarity_broadening: bool,
) -> tuple[pd.DataFrame, str]:
    valid = candidates[candidates["candidate_status"] == "good"].copy()

    def sel(**kwargs) -> pd.DataFrame:
        sub = valid.copy()
        for k, v in kwargs.items():
            if v is None:
                continue
            sub = sub[sub[k] == v]
        return sub

    exact = sel(polarity=polarity, solid_target=solid_target, charge=charge, detector_region=detector_region, sector=sector)
    if not exact.empty:
        return exact, "exact"

    if detector_region == "forward" and allow_sector_all_fallback:
        sub = sel(polarity=polarity, solid_target=solid_target, charge=charge, detector_region=detector_region, sector="all")
        if not sub.empty:
            return sub, "forward_all_sector_same_target"

        sub = sel(polarity=polarity, solid_target=solid_target, charge=charge, detector_region=detector_region)
        if not sub.empty:
            return sub, "combined_forward_sectors_same_target"

    if allow_target_broadening:
        sub = sel(polarity=polarity, charge=charge, detector_region=detector_region, sector=sector)
        if not sub.empty:
            return sub, "same_polarity_broaden_target_exact_sector"

        if detector_region == "forward" and allow_sector_all_fallback:
            sub = sel(polarity=polarity, charge=charge, detector_region=detector_region, sector="all")
            if not sub.empty:
                return sub, "same_polarity_broaden_target_forward_all_sector"

            sub = sel(polarity=polarity, charge=charge, detector_region=detector_region)
            if not sub.empty:
                return sub, "same_polarity_broaden_target_combined_forward_sectors"

    if allow_polarity_broadening:
        sub = valid[(valid["charge"] == charge) & (valid["detector_region"] == detector_region) & (valid["sector"] == sector)]
        if not sub.empty:
            return sub, "broaden_polarity_and_target_exact_sector"

        if detector_region == "forward" and allow_sector_all_fallback:
            sub = valid[(valid["charge"] == charge) & (valid["detector_region"] == detector_region) & (valid["sector"] == "all")]
            if not sub.empty:
                return sub, "broaden_polarity_and_target_forward_all_sector"

            sub = valid[(valid["charge"] == charge) & (valid["detector_region"] == detector_region)]
            if not sub.empty:
                return sub, "broaden_polarity_and_target_combined_forward_sectors"

    return valid.iloc[0:0].copy(), "missing"


def _aggregate_recommendation_row(
    *,
    subset: pd.DataFrame,
    polarity: str,
    solid_target: str,
    charge: str,
    detector_region: str,
    sector: str,
    statistic: str,
    min_run_candidates: int,
    fallback_level: str,
) -> dict[str, Any]:
    row = {
        "polarity": polarity,
        "solid_target": solid_target,
        "charge": charge,
        "vertex_source": "hybrid",
        "detector_region": detector_region,
        "sector": sector,
        "recommendation_status": "missing",
        "recommendation_reason": "",
        "fallback_level": fallback_level,
        "n_candidate_rows": 0,
        "n_unique_runs": 0,
        "ld2_cut_low": np.nan,
        "ld2_cut_high": np.nan,
        "solid_cut_low": np.nan,
        "solid_cut_high": np.nan,
        "reference_low": np.nan,
        "reference_high": np.nan,
        "ld2_cut_low_spread": np.nan,
        "ld2_cut_high_spread": np.nan,
        "solid_cut_low_spread": np.nan,
        "solid_cut_high_spread": np.nan,
        "source_runs": "",
    }

    if subset.empty:
        row["recommendation_status"] = "missing"
        row["recommendation_reason"] = "no valid candidate cuts found"
        return row

    unique_runs = sorted(set(subset["run"].astype(str)))
    row["n_candidate_rows"] = int(len(subset))
    row["n_unique_runs"] = int(len(unique_runs))
    row["source_runs"] = ",".join(unique_runs)

    if len(unique_runs) < min_run_candidates:
        row["recommendation_status"] = "insufficient_runs"
        row["recommendation_reason"] = f"only {len(unique_runs)} unique run(s), need at least {min_run_candidates}"
        return row

    for col in ["ld2_cut_low", "ld2_cut_high", "solid_cut_low", "solid_cut_high", "reference_low", "reference_high"]:
        row[col] = _aggregate_stat(subset[col], statistic)

    row["ld2_cut_low_spread"] = float(np.nanmax(subset["ld2_cut_low"]) - np.nanmin(subset["ld2_cut_low"]))
    row["ld2_cut_high_spread"] = float(np.nanmax(subset["ld2_cut_high"]) - np.nanmin(subset["ld2_cut_high"]))
    row["solid_cut_low_spread"] = float(np.nanmax(subset["solid_cut_low"]) - np.nanmin(subset["solid_cut_low"]))
    row["solid_cut_high_spread"] = float(np.nanmax(subset["solid_cut_high"]) - np.nanmin(subset["solid_cut_high"]))

    if not (np.isfinite(row["ld2_cut_low"]) and np.isfinite(row["ld2_cut_high"]) and np.isfinite(row["solid_cut_low"]) and np.isfinite(row["solid_cut_high"])):
        row["recommendation_status"] = "invalid"
        row["recommendation_reason"] = "aggregated cuts are not finite"
        return row
    if not (row["ld2_cut_low"] < row["ld2_cut_high"]):
        row["recommendation_status"] = "invalid"
        row["recommendation_reason"] = "aggregated LD2 cut collapsed"
        return row
    if not (row["solid_cut_low"] < row["solid_cut_high"]):
        row["recommendation_status"] = "invalid"
        row["recommendation_reason"] = "aggregated solid cut collapsed"
        return row

    row["recommendation_status"] = "recommended"
    row["recommendation_reason"] = "aggregated from valid candidate cuts"
    return row


def build_recommended_cuts(
    *,
    all_candidates_df: pd.DataFrame,
    final_charges: list[str],
    forward_sectors: list[int | str],
    central_sector: str,
    statistic: str,
    min_run_candidates: int,
    allow_sector_all_fallback: bool,
    allow_target_broadening: bool,
    allow_polarity_broadening: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if all_candidates_df.empty:
        return pd.DataFrame(rows)

    observed_targets = (
        all_candidates_df[["polarity", "solid_target"]]
        .drop_duplicates()
        .sort_values(["polarity", "solid_target"])
        .to_dict(orient="records")
    )

    for item in observed_targets:
        polarity = item["polarity"]
        solid_target = item["solid_target"]

        for charge in final_charges:
            for sector in [str(s) for s in forward_sectors]:
                subset, fallback_level = _choose_fallback_subset(
                    all_candidates_df,
                    polarity=polarity,
                    solid_target=solid_target,
                    charge=charge,
                    detector_region="forward",
                    sector=sector,
                    allow_sector_all_fallback=allow_sector_all_fallback,
                    allow_target_broadening=allow_target_broadening,
                    allow_polarity_broadening=allow_polarity_broadening,
                )
                rows.append(
                    _aggregate_recommendation_row(
                        subset=subset,
                        polarity=polarity,
                        solid_target=solid_target,
                        charge=charge,
                        detector_region="forward",
                        sector=sector,
                        statistic=statistic,
                        min_run_candidates=min_run_candidates,
                        fallback_level=fallback_level,
                    )
                )

            subset, fallback_level = _choose_fallback_subset(
                all_candidates_df,
                polarity=polarity,
                solid_target=solid_target,
                charge=charge,
                detector_region="central",
                sector=str(central_sector),
                allow_sector_all_fallback=False,
                allow_target_broadening=allow_target_broadening,
                allow_polarity_broadening=allow_polarity_broadening,
            )
            rows.append(
                _aggregate_recommendation_row(
                    subset=subset,
                    polarity=polarity,
                    solid_target=solid_target,
                    charge=charge,
                    detector_region="central",
                    sector=str(central_sector),
                    statistic=statistic,
                    min_run_candidates=min_run_candidates,
                    fallback_level=fallback_level,
                )
            )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        sort_cols = ["polarity", "solid_target", "charge", "detector_region", "sector"]
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
    return out_df


def save_cut_tables(*, all_candidate_cuts_df: pd.DataFrame, recommended_cuts_df: pd.DataFrame, all_candidate_cuts_csv: str | Path, recommended_cuts_csv: str | Path) -> None:
    all_candidate_cuts_csv = Path(all_candidate_cuts_csv)
    recommended_cuts_csv = Path(recommended_cuts_csv)
    all_candidate_cuts_csv.parent.mkdir(parents=True, exist_ok=True)
    recommended_cuts_csv.parent.mkdir(parents=True, exist_ok=True)
    all_candidate_cuts_df.to_csv(all_candidate_cuts_csv, index=False)
    recommended_cuts_df.to_csv(recommended_cuts_csv, index=False)

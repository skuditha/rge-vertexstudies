#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rge_vertex.io.configs import load_yaml, repo_path
from rge_vertex.cuts.extract_ld2_solid_cuts import (
    build_all_candidate_cuts,
    build_recommended_cuts,
    load_category_summary,
    load_reference_table,
    save_cut_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LD2+solid vertex cuts and recommendation tables.")
    parser.add_argument("--config", default="configs/cut_extraction.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--reference-foil-csv", default=None)
    parser.add_argument("--category-summary-csv", default=None)
    parser.add_argument("--all-candidate-cuts-csv", default=None)
    parser.add_argument("--recommended-cuts-csv", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    cfg_all = load_yaml(args.config)
    cfg = cfg_all["cut_extraction"]

    input_cfg = cfg.get("inputs", {})
    policy_cfg = cfg.get("policy", {})
    cut_cfg = cfg.get("cut_recipe", {})
    agg_cfg = cfg.get("aggregation", {})
    fallback_cfg = cfg.get("fallback", {})
    output_cfg = cfg.get("outputs", {})

    reference_csv = repo_path(args.reference_foil_csv or input_cfg.get("reference_foil_csv", "outputs/tables/reference_foil_positions.csv"), repo_root)
    category_summary_csv = repo_path(args.category_summary_csv or input_cfg.get("category_summary_csv", "outputs/tables/ld2_solid_category_summary.csv"), repo_root)
    all_candidate_cuts_csv = repo_path(args.all_candidate_cuts_csv or output_cfg.get("all_candidate_cuts_csv", "outputs/tables/ld2_solid_vertex_cuts_all.csv"), repo_root)
    recommended_cuts_csv = repo_path(args.recommended_cuts_csv or output_cfg.get("recommended_cuts_csv", "outputs/tables/ld2_solid_vertex_cuts_recommended.csv"), repo_root)

    reference_df = load_reference_table(reference_csv)
    category_df = load_category_summary(category_summary_csv)

    all_candidate_df = build_all_candidate_cuts(
        category_df=category_df,
        reference_df=reference_df,
        recommended_vertex_source=policy_cfg.get("recommended_vertex_source", "hybrid"),
        accepted_pid_categories=policy_cfg.get("accepted_pid_categories", ["charge_only", "charge_negative", "charge_positive"]),
        reference_vertex_source_fallback_order=policy_cfg.get("reference_vertex_source_fallback_order", ["hybrid", "particle", "ftrack"]),
        ld2_n_sigma=float(cut_cfg.get("ld2_n_sigma", 2.0)),
        solid_n_sigma=float(cut_cfg.get("solid_n_sigma", 2.0)),
        clip_to_reference=bool(cut_cfg.get("clip_to_reference", True)),
        safety_margin_cm=float(cut_cfg.get("safety_margin_cm", 0.0)),
    )

    recommended_df = build_recommended_cuts(
        all_candidates_df=all_candidate_df,
        final_charges=policy_cfg.get("final_charges", ["negative", "positive"]),
        forward_sectors=policy_cfg.get("forward_sectors", [1, 2, 3, 4, 5, 6]),
        central_sector=str(policy_cfg.get("central_sector", "all")),
        statistic=agg_cfg.get("statistic", "median"),
        min_run_candidates=int(agg_cfg.get("min_run_candidates", 1)),
        allow_sector_all_fallback=bool(fallback_cfg.get("allow_sector_all_fallback", True)),
        allow_target_broadening=bool(fallback_cfg.get("allow_target_broadening", True)),
        allow_polarity_broadening=bool(fallback_cfg.get("allow_polarity_broadening", False)),
    )

    save_cut_tables(
        all_candidate_cuts_df=all_candidate_df,
        recommended_cuts_df=recommended_df,
        all_candidate_cuts_csv=all_candidate_cuts_csv,
        recommended_cuts_csv=recommended_cuts_csv,
    )

    print(f"Wrote all candidate cuts: {all_candidate_cuts_csv}")
    print(f"Wrote recommended cuts: {recommended_cuts_csv}")
    print(f"Candidate rows: {len(all_candidate_df)}")
    print(f"Recommended rows: {len(recommended_df)}")


if __name__ == "__main__":
    main()

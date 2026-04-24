#!/usr/bin/env bash
set -euo pipefail

QA_CONFIG="configs/production_qa.yaml"
FIT_CONFIG="configs/ld2_solid_local_fit.yaml"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --qa-config PATH     Default: ${QA_CONFIG}
  --fit-config PATH    Default: ${FIT_CONFIG}
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qa-config) QA_CONFIG="$2"; shift 2 ;;
    --fit-config) FIT_CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

python scripts/04b_merge_production_qa_csvs.py --qa-config "${QA_CONFIG}"
python scripts/05b_merge_ld2_solid_fit_csvs.py --fit-config "${FIT_CONFIG}"

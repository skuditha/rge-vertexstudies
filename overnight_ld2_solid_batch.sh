#!/usr/bin/env bash
set -euo pipefail

# Overnight batch launcher for RG-E ld2_solid runs.
#
# What it does:
#   1) Optionally runs script 1 (HIPO -> ROOT ntuples) for selected runs
#   2) Runs script 4 (production QA histograms) one run per process
#   3) Runs script 5 (LD2+solid local fits) one run per process
#   4) Leaves per-run CSV outputs in parallel-safe directories
#
# Requirements:
#   - bash
#   - python environment with repo package installed
#   - xargs with -P support
#   - matplotlib in non-interactive mode
#
# Usage examples:
#
#   # Run QA + fits only, on all enabled ld2_solid runs, with 30-way parallelism
#   ./scripts/overnight_ld2_solid_batch.sh
#
#   # Include ntuple extraction too
#   ./scripts/overnight_ld2_solid_batch.sh --do-extract
#
#   # Only inbending
#   ./scripts/overnight_ld2_solid_batch.sh --polarities inbending
#
#   # Only a couple of targets
#   ./scripts/overnight_ld2_solid_batch.sh --solid-targets C,Al
#
#   # Dry run
#   ./scripts/overnight_ld2_solid_batch.sh --dry-run

RUNS_CONFIG="configs/runs.yaml"
QA_CONFIG="configs/production_qa.yaml"
FIT_CONFIG="configs/ld2_solid_local_fit.yaml"

NPROC=30
DO_EXTRACT=0
DO_QA=1
DO_FIT=1
SKIP_IF_DONE=1
DRY_RUN=0
POLARITIES=""
SOLID_TARGETS=""
RUNS=""
LOG_DIR="outputs/logs/overnight_ld2_solid"
MPLBACKEND_VALUE="Agg"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --runs-config PATH        Default: ${RUNS_CONFIG}
  --qa-config PATH          Default: ${QA_CONFIG}
  --fit-config PATH         Default: ${FIT_CONFIG}
  --nproc N                 Default: ${NPROC}
  --do-extract              Also run script 1 (ntuple extraction)
  --no-qa                   Skip script 4
  --no-fit                  Skip script 5
  --no-skip-if-done         Re-run even if per-run outputs already exist
  --polarities LIST         Comma-separated, e.g. inbending,outbending
  --solid-targets LIST      Comma-separated, e.g. C,Al,Cu,Sn,Pb
  --runs LIST               Comma-separated run numbers, e.g. 020026,020027
  --log-dir PATH            Default: ${LOG_DIR}
  --dry-run                 Print commands without running them
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs-config) RUNS_CONFIG="$2"; shift 2 ;;
    --qa-config) QA_CONFIG="$2"; shift 2 ;;
    --fit-config) FIT_CONFIG="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --do-extract) DO_EXTRACT=1; shift ;;
    --no-qa) DO_QA=0; shift ;;
    --no-fit) DO_FIT=0; shift ;;
    --no-skip-if-done) SKIP_IF_DONE=0; shift ;;
    --polarities) POLARITIES="$2"; shift 2 ;;
    --solid-targets) SOLID_TARGETS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}"
export MPLBACKEND="${MPLBACKEND_VALUE}"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_list_file="${LOG_DIR}/run_list_${timestamp}.txt"

python - "$RUNS_CONFIG" "$POLARITIES" "$SOLID_TARGETS" "$RUNS" > "${run_list_file}" <<'PY'
import sys, yaml
from pathlib import Path

runs_config, polarities_csv, solid_targets_csv, runs_csv = sys.argv[1:5]

with open(runs_config, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

runs = data.get("runs", {})

polarities = [x.strip() for x in polarities_csv.split(",") if x.strip()] if polarities_csv else None
solid_targets = [x.strip() for x in solid_targets_csv.split(",") if x.strip()] if solid_targets_csv else None
explicit_runs = {x.strip().zfill(6) for x in runs_csv.split(",") if x.strip()} if runs_csv else None

selected = []
for run, meta in sorted(runs.items(), key=lambda kv: str(kv[0]).zfill(6)):
    run = str(run).zfill(6)
    if explicit_runs is not None and run not in explicit_runs:
        continue
    if not bool(meta.get("enabled", True)):
        continue
    if meta.get("run_class") != "ld2_solid":
        continue
    if polarities is not None and meta.get("polarity") not in polarities:
        continue
    if solid_targets is not None and meta.get("solid_target") not in solid_targets:
        continue
    selected.append(run)

for run in selected:
    print(run)
PY

num_runs="$(wc -l < "${run_list_file}" | tr -d ' ')"
if [[ "${num_runs}" == "0" ]]; then
  echo "No runs selected." >&2
  exit 1
fi

echo "Selected ${num_runs} run(s)."
echo "Run list: ${run_list_file}"
echo "Logs: ${LOG_DIR}"
echo "Parallel workers: ${NPROC}"

run_extract() {
  local run="$1"
  local log="${LOG_DIR}/${run}.extract.log"
  local cmd=(python scripts/01_extract_ntuples.py --runs-config "${RUNS_CONFIG}" --run "${run}")
  echo "[extract] ${run}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  "${cmd[@]}" > "${log}" 2>&1
}

run_qa() {
  local run="$1"
  local log="${LOG_DIR}/${run}.qa.log"
  local cmd=(python scripts/04_make_production_qa_histograms.py
    --runs-config "${RUNS_CONFIG}"
    --qa-config "${QA_CONFIG}"
    --run "${run}")
  if [[ "${SKIP_IF_DONE}" == "1" ]]; then
    cmd+=(--skip-if-done)
  fi
  echo "[qa] ${run}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  "${cmd[@]}" > "${log}" 2>&1
}

run_fit() {
  local run="$1"
  local log="${LOG_DIR}/${run}.fit.log"
  local cmd=(python scripts/05_fit_ld2_solid.py
    --runs-config "${RUNS_CONFIG}"
    --fit-config "${FIT_CONFIG}"
    --run "${run}")
  if [[ "${SKIP_IF_DONE}" == "1" ]]; then
    cmd+=(--skip-if-done)
  fi
  echo "[fit] ${run}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  "${cmd[@]}" > "${log}" 2>&1
}

export -f run_extract
export -f run_qa
export -f run_fit
export RUNS_CONFIG QA_CONFIG FIT_CONFIG LOG_DIR DRY_RUN SKIP_IF_DONE MPLBACKEND

if [[ "${DO_EXTRACT}" == "1" ]]; then
  echo
  echo "=== Phase 1: extraction ==="
  xargs -a "${run_list_file}" -n 1 -P "${NPROC}" -I {} bash -lc 'run_extract "$@"' _ {}
fi

if [[ "${DO_QA}" == "1" ]]; then
  echo
  echo "=== Phase 2: production QA histograms ==="
  xargs -a "${run_list_file}" -n 1 -P "${NPROC}" -I {} bash -lc 'run_qa "$@"' _ {}
fi

if [[ "${DO_FIT}" == "1" ]]; then
  echo
  echo "=== Phase 3: LD2+solid local fits ==="
  xargs -a "${run_list_file}" -n 1 -P "${NPROC}" -I {} bash -lc 'run_fit "$@"' _ {}
fi

echo
echo "Done."
echo "Merge per-run CSVs afterward with:"
echo "  python scripts/04b_merge_production_qa_csvs.py --qa-config ${QA_CONFIG}"
echo "  python scripts/05b_merge_ld2_solid_fit_csvs.py --fit-config ${FIT_CONFIG}"

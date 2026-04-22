#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${CPP_DIR}/build"
cd "${CPP_DIR}/build"
cmake ..
cmake --build . -j
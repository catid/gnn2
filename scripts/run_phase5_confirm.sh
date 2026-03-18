#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:-}"
if [[ -z "${run_dir}" ]]; then
  echo "usage: $0 <run-dir>" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

uv run python -m src.utils.phase5_verify \
  --run-dir "${run_dir}" \
  --confirm-batches 16

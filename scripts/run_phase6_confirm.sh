#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:-}"
eval_config="${2:-}"
if [[ -z "${run_dir}" ]]; then
  echo "usage: $0 <run-dir> [eval-config]" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

cmd=(uv run python -m src.utils.phase6_verify --run-dir "${run_dir}" --confirm-batches 16)
if [[ -n "${eval_config}" ]]; then
  cmd+=(--eval-config "${eval_config}")
fi
"${cmd[@]}"

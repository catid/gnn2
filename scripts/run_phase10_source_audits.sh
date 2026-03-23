#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <run-dir> [split] [eval-config]" >&2
  exit 1
fi

run_dir="$1"
split="${2:-test}"
eval_config="${3:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

cmd=(uv run python -m src.utils.phase9_audit --run-dir "${run_dir}" --split "${split}")
if [[ -n "${eval_config}" ]]; then
  cmd+=(--eval-config "${eval_config}")
fi
"${cmd[@]}"

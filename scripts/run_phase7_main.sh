#!/usr/bin/env bash
set -euo pipefail

config="${1:-}"
results_dir="${2:-}"
resume="${3:-}"
nproc="${4:-1}"
if [[ -z "${config}" || -z "${results_dir}" ]]; then
  echo "usage: $0 <config> <results-dir> [resume] [nproc_per_node]" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export MAKEFLAGS="${MAKEFLAGS:--j16}"

if [[ "${nproc}" -gt 1 ]]; then
  cmd=(uv run torchrun --standalone --nproc_per_node="${nproc}" -m src.train.run --config "${config}" --results-dir "${results_dir}")
else
  cmd=(uv run python -m src.train.run --config "${config}" --results-dir "${results_dir}")
fi

if [[ -n "${resume}" ]]; then
  cmd+=(--resume "${resume}")
fi

"${cmd[@]}"

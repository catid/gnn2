#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 <config> <results-root> <resume> <seed1> [seed2 ...]" >&2
  exit 1
fi

config="$1"
results_root="$2"
resume="$3"
shift 3

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export MAKEFLAGS="${MAKEFLAGS:--j16}"

mkdir -p "${results_root}"

for seed in "$@"; do
  out="${results_root}/seed${seed}"
  uv run python -m src.train.run \
    --config "${config}" \
    --resume "${resume}" \
    --results-dir "${out}" \
    >/tmp/phase7_seed_${seed}.log 2>&1 &
done

wait

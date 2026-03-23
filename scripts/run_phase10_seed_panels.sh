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
  if [[ -f "${out}/summary.json" ]]; then
    echo "skip existing completed run: ${out}"
    continue
  elif [[ -d "${out}" ]] && find "${out}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "refusing non-empty output dir: ${out}" >&2
    exit 1
  fi
  cmd=(uv run python -m src.train.run --config "${config}" --results-dir "${out}")
  if [[ -n "${resume}" ]]; then
    cmd+=(--resume "${resume}")
  fi
  "${cmd[@]}" >/tmp/phase10_seed_${seed}.log 2>&1 &
done

wait

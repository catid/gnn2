#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:-}"
if [[ -z "${run_dir}" ]]; then
  echo "usage: $0 <run-dir> [extra-eval-config ...]" >&2
  exit 1
fi
shift || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

cmd=(uv run python -m src.utils.phase9_verify --run-dir "${run_dir}" --confirm-batches 16)
if [[ "$#" -eq 0 ]]; then
  set -- \
    configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_full_locked.yaml \
    configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_finalqueryheavy.yaml \
    configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_longdistance.yaml
fi
for eval_config in "$@"; do
  cmd+=(--eval-config "${eval_config}")
done
"${cmd[@]}"

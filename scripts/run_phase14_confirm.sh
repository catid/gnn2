#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run-dir> [extra phase14_verify args...]" >&2
  exit 2
fi

run_dir="$1"
shift

uv run python -m src.utils.phase14_verify \
  --run-dir "$run_dir" \
  --eval-config configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_full_locked.yaml \
  --eval-config configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_finalqueryheavy.yaml \
  --eval-config configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_longdistance.yaml \
  "$@"

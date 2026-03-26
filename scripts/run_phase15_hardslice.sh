#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run-dir-a> <run-dir-b> [out-json]" >&2
  exit 2
fi

run_a="$1"
run_b="$2"
out="${3:-artifacts/phase15_hardslice/$(basename "$run_a")__vs__$(basename "$run_b").json}"

uv run python -m src.utils.phase15_hardslice \
  --run-dir "$run_a" \
  --run-dir "$run_b" \
  --name "$(basename "$run_a")" \
  --name "$(basename "$run_b")" \
  --eval-config configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_full_locked.yaml \
  --split confirm \
  --num-batches 32 \
  --batch-size 256 \
  --out "$out"

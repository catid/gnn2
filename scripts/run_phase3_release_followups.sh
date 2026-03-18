#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${1:-phase3_release}"
CKPT="results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  echo "Run the oraclewarm keyed-memory config first." >&2
  exit 1
fi

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase3/dev/hard_st_benchmark_b_v2_keymem_payloadaux_release_maskcurr.yaml \
  --resume "$CKPT" \
  --results-dir "results/phase3_dev/${TAG}_maskcurr_from_oraclewarm" &
pid0=$!

CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run \
  --config configs/phase3/dev/hard_st_benchmark_b_v2_keymem_payloadaux_release_nomask.yaml \
  --resume "$CKPT" \
  --results-dir "results/phase3_dev/${TAG}_nomask_seed710" &
pid1=$!

wait "$pid0"
wait "$pid1"

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase3/dev/hard_st_benchmark_b_v2_keymem_payloadaux_release_nomask_seed712.yaml \
  --resume "$CKPT" \
  --results-dir "results/phase3_dev/${TAG}_nomask_seed712" &
pid2=$!

CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run \
  --config configs/phase3/dev/hard_st_benchmark_b_v2_keymem_payloadaux_release_nomask_seed713.yaml \
  --resume "$CKPT" \
  --results-dir "results/phase3_dev/${TAG}_nomask_seed713" &
pid3=$!

wait "$pid2"
wait "$pid3"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

RESUME_CKPT="results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt"

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase4/main/hard_st_benchmark_b_v2_control_sticky_both_main.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_main/hard_st_b_v2_control_sticky_both_main_seed750

CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run \
  --config configs/phase4/main/hard_st_benchmark_b_v2_control_sticky_both_main_seed752.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_main/hard_st_b_v2_control_sticky_both_main_seed752

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase4/main/hard_st_benchmark_b_v2_control_sticky_both_main_seed753.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_main/hard_st_b_v2_control_sticky_both_main_seed753

CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run \
  --config configs/phase4/main/hard_st_benchmark_b_v2_control_sticky_both_main_seed754.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_main/hard_st_b_v2_control_sticky_both_main_seed754

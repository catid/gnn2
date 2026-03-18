#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

RESUME_CKPT="results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt"

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_aux.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_aux

CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_antiexit.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_antiexit

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_both_dim8.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_both_dim8

CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_setclear_both.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_setclear_both

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_both.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_both

CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_both_highcontrol.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_both_highcontrol

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_both_noroutece.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_both_noroutece

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_aux_router2.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2

CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_aux_router4.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_aux_router4

CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_aux_router2_both.yaml \
  --resume "$RESUME_CKPT" \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2_both

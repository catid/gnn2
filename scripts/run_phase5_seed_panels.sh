#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

uv run python -m src.train.run \
  --config configs/phase4/main/hard_st_benchmark_b_v2_control_sticky_aux_router2_main.yaml \
  --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
  --results-dir results/phase5_anchor/hard_st_b_v2_control_sticky_aux_router2_main_seed770_rerun1

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase4/main/hybrid_es_benchmark_b_v2_control_sticky_aux_router2_resume_seed747.yaml \
  --resume results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/hard_st_best.pt \
  --results-dir results/phase5_anchor/hybrid_es_b_v2_control_sticky_aux_router2_resume_seed747_rerun1

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase5/dev/hybrid_es_benchmark_b_v2_control_router2_setclear_oraclecontrol_resume.yaml \
  --resume results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1/hard_st_best.pt \
  --results-dir results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_rerun1

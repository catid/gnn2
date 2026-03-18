#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

case "${1:-}" in
  medium-adapter)
    uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
      --config configs/phase5/dev/hybrid_es_benchmark_b_v2_control_router2_setclear_oraclecontrol_resume.yaml \
      --resume results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1/hard_st_best.pt \
      --results-dir results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_p1
    ;;
  medium-routeronly)
    uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
      --config configs/phase5/dev/hybrid_es_benchmark_b_v2_control_router2_setclear_oraclecontrol_resume_routeronly.yaml \
      --resume results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1/hard_st_best.pt \
      --results-dir results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_routeronly_from950_seed951_p1
    ;;
  weak-adapter)
    uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
      --config configs/phase5/dev/hybrid_es_benchmark_b_v2_control_router2_setclear_oraclecontrol_resume.yaml \
      --resume results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_seed947_p1/hard_st_best.pt \
      --results-dir results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_p1
    ;;
  weak-routeronly)
    uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
      --config configs/phase5/dev/hybrid_es_benchmark_b_v2_control_router2_setclear_oraclecontrol_resume_routeronly.yaml \
      --resume results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_seed947_p1/hard_st_best.pt \
      --results-dir results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_routeronly_from947_seed951_p1
    ;;
  *)
    echo "usage: $0 {medium-adapter|medium-routeronly|weak-adapter|weak-routeronly}" >&2
    exit 1
    ;;
esac

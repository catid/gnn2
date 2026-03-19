#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

case "${1:-}" in
  b-scout)
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_waitact_setclear_curriculum.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_waitact_setclear_curriculum_seed980_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_controlwaitact_setclear_curriculum.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_controlwaitact_setclear_curriculum_seed981_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_recurrent_waitact_curriculum.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_recurrent_waitact_curriculum_seed982_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_controlwaitact_releasegate_curriculum.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_controlwaitact_releasegate_curriculum_seed983_p1
    ;;
  c-scout)
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_controlsticky_memoryheavy.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_controlsticky_memoryheavy_seed984_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_controlsetclear_memoryheavy.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_controlsetclear_memoryheavy_seed985_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_controlsticky_keepalive.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_controlsticky_keepalive_seed989_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_controlsetclear_waitstate_memoryheavy.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_controlsetclear_waitstate_memoryheavy_seed990_p1
    ;;
  a-scout)
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_weak_es_content_refine_all.yaml \
      --resume results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_p1/hybrid_es_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_weak_es_content_refine_all_seed970_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_weak_es_content_refine_memoryreadout.yaml \
      --resume results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_p1/hybrid_es_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_weak_es_content_refine_memoryreadout_seed971_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_weak_es_content_refine_adapter.yaml \
      --resume results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_p1/hybrid_es_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_weak_es_content_refine_adapter_seed972_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_weak_es_content_refine_sinkcore.yaml \
      --resume results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_p1/hybrid_es_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_p1
    ;;
  d-scout)
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_router2_forceoracle_release.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_router2_forceoracle_release_seed986_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_router2_controlleronly_imitation_release.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_router2_controlleronly_imitation_release_seed987_p1
    uv run python -m src.train.run \
      --config configs/phase6/dev/hard_st_benchmark_b_v2_finalqueryonly_controller_pretrain.yaml \
      --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
      --results-dir results/phase6_dev/hard_st_b_v2_finalqueryonly_controller_pretrain_seed988_p1
    ;;
  *)
    echo "usage: $0 {a-scout|b-scout|c-scout|d-scout}" >&2
    exit 1
    ;;
esac

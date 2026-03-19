#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  anchors)
    CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase7_main.sh \
      configs/phase6/dev/hard_st_benchmark_b_v2_controlsticky_keepalive.yaml \
      results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1 \
      results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt &
    CUDA_VISIBLE_DEVICES=1 ./scripts/run_phase7_main.sh \
      configs/phase6/dev/hard_st_benchmark_b_v2_weak_es_content_refine_sinkcore.yaml \
      results/phase7_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1 \
      results/phase6_anchor/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_rerun1/hybrid_es_best.pt &
    wait
    ;;
  *)
    echo "usage: $0 {anchors}" >&2
    exit 1
    ;;
esac

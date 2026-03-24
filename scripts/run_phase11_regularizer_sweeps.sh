#!/usr/bin/env bash
set -euo pipefail

root="${1:-results/phase11_dev}"
mode="${2:-initial}"

case "${mode}" in
  initial)
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proxyreg_locked01_selectlocked_fqacc_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proxyreg_locked01_selectlocked_fqacc_longer_lowlr_seed11211_p1"
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proxyreg_locked02_selectlocked_fqacc_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proxyreg_locked02_selectlocked_fqacc_longer_lowlr_seed11212_p1"
    ;;
  shifted)
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proxyreg_lockedfqh01_selectlocked_fqacc_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proxyreg_lockedfqh01_selectlocked_fqacc_longer_lowlr_seed11213_p1"
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proxyreg_lockedfqhld01_selectlocked_fqacc_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proxyreg_lockedfqhld01_selectlocked_fqacc_longer_lowlr_seed11214_p1"
    ;;
  *)
    echo "usage: $0 [results-root] [initial|shifted]" >&2
    exit 1
    ;;
esac

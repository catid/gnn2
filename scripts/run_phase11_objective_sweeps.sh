#!/usr/bin/env bash
set -euo pipefail

root="${1:-results/phase11_dev}"
mode="${2:-initial}"

case "${mode}" in
  initial)
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_confirmmix_locked_fq3_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_confirmmix_locked_fq3_longer_lowlr_seed11011_p1"
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_confirmmix_locked_fq5_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_confirmmix_locked_fq5_longer_lowlr_seed11012_p1"
    ;;
  mixed)
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_confirmmix_locked_fqh_fq5_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_confirmmix_locked_fqh_fq5_longer_lowlr_seed11013_p1"
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_confirmmix_locked_fqh_longdistance_fq5_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_confirmmix_locked_fqh_longdistance_fq5_longer_lowlr_seed11014_p1"
    ;;
  *)
    echo "usage: $0 [results-root] [initial|mixed]" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

root="${1:-results/phase11_dev}"
mode="${2:-focal}"

case "${mode}" in
  focal)
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_fqfocal_g1_w025_selectlocked_fqacc_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_fqfocal_g1_w025_selectlocked_fqacc_longer_lowlr_seed11411_p1"
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_fqfocal_g2_w050_selectlocked_fqacc_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_fqfocal_g2_w050_selectlocked_fqacc_longer_lowlr_seed11412_p1"
    ;;
  margin)
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_fqmargin_m02_w025_selectlocked_fqacc_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_fqmargin_m02_w025_selectlocked_fqacc_longer_lowlr_seed11413_p1"
    ./scripts/run_phase11_main.sh configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_fqmargin_m04_w050_selectlocked_fqacc_longer_lowlr.yaml "${root}/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_fqmargin_m04_w050_selectlocked_fqacc_longer_lowlr_seed11414_p1"
    ;;
  *)
    echo "usage: $0 [results-root] [focal|margin]" >&2
    exit 1
    ;;
esac

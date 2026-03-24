#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"

config="configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_refine_multiview_sinkpacketbaseline_querygated_selectproxy_full_locked_fqacc_longer_lowlr.yaml"
main_dir="results/phase11_dev/hard_st_b_v2_teacher1201_refine_multiview_sinkpacketbaseline_querygated_selectproxy_full_locked_fqacc_longer_lowlr_seed11721_p1"
rerun_dir="results/phase11_dev/hard_st_b_v2_teacher1201_refine_multiview_sinkpacketbaseline_querygated_selectproxy_full_locked_fqacc_longer_lowlr_seed11721_rerun1"

case "${mode}" in
  main)
    ./scripts/run_phase11_main.sh "${config}" "${main_dir}"
    ;;
  rerun)
    ./scripts/run_phase11_main.sh "${config}" "${rerun_dir}"
    ;;
  *)
    echo "usage: $0 {main|rerun}" >&2
    exit 1
    ;;
esac

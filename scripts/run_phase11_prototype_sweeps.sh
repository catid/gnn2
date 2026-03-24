#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {round1|round2}" >&2
  exit 1
fi

run() {
  local gpu="$1"
  local config="$2"
  local outdir="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" ./scripts/run_phase11_main.sh "$config" "$outdir"
}

case "$1" in
  round1)
    run 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proto_cosine_s10_selectlocked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proto_cosine_s10_selectlocked_fqacc_longer_lowlr_seed11511_p1 &
    pid0=$!
    run 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proto_cosine_s20_selectlocked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proto_cosine_s20_selectlocked_fqacc_longer_lowlr_seed11512_p1 &
    pid1=$!
    wait "$pid0" "$pid1"
    ;;
  round2)
    run 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proto_cosine_d64_s10_selectlocked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proto_cosine_d64_s10_selectlocked_fqacc_longer_lowlr_seed11513_p1 &
    pid0=$!
    run 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proto_cosine_d64_pull01_selectlocked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_proto_cosine_d64_pull01_selectlocked_fqacc_longer_lowlr_seed11514_p1 &
    pid1=$!
    wait "$pid0" "$pid1"
    ;;
  *)
    echo "unknown round: $1" >&2
    exit 1
    ;;
esac

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
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12713_13111_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12713_13111_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed13211_p1 &
    pid0=$!
    run 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12713_13111_bal90_route10_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12713_13111_bal90_route10_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed13212_p1 &
    pid1=$!
    wait "$pid0" "$pid1"
    ;;
  round2)
    run 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12713_13111_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12713_13111_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed13213_p1 &
    pid0=$!
    run 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12713_13111_bal80_route20_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12713_13111_bal80_route20_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed13214_p1 &
    pid1=$!
    wait "$pid0" "$pid1"
    ;;
  *)
    echo "unknown round: $1" >&2
    exit 1
    ;;
esac

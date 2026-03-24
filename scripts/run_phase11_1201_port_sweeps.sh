#!/usr/bin/env bash
set -euo pipefail

run() {
  local gpu="$1"
  local config="$2"
  local outdir="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" ./scripts/run_phase11_main.sh "$config" "$outdir"
}

run 0 \
  configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_refine_readoutonly_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml \
  results/phase11_dev/hard_st_b_v2_teacher1201_refine_readoutonly_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr_seed11722_p1 &
pid0=$!

run 1 \
  configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_refine_querygated_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml \
  results/phase11_dev/hard_st_b_v2_teacher1201_refine_querygated_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr_seed11723_p1 &
pid1=$!

wait "$pid0" "$pid1"

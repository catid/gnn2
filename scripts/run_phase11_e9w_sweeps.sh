#!/usr/bin/env bash
set -euo pipefail

run() {
  local gpu="$1"
  local config="$2"
  local outdir="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" ./scripts/run_phase11_main.sh "$config" "$outdir"
}

case "${1:-}" in
  round1)
    run 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headinterp_bal45_route55_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headinterp_bal45_route55_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr_seed12511_p1 &
    pid0=$!
    run 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headinterp_bal45_route55_selectlocked_overall_fq4_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headinterp_bal45_route55_selectlocked_overall_fq4_longer_lowlr_seed12512_p1 &
    pid1=$!
    wait "$pid0" "$pid1"
    ;;
  round2)
    run 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headinterp_bal45_route55_selectproxy_full_locked_fqacc_fq4_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headinterp_bal45_route55_selectproxy_full_locked_fqacc_fq4_longer_lowlr_seed12513_p1 &
    pid0=$!
    run 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headinterp_bal45_route55_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headinterp_bal45_route55_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed12514_p1 &
    pid1=$!
    wait "$pid0" "$pid1"
    ;;
  *)
    echo "usage: $0 {round1|round2}" >&2
    exit 1
    ;;
esac

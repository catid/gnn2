#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

run_cfg() {
  local gpu="$1"
  local cfg="$2"
  local out="$3"
  CUDA_VISIBLE_DEVICES="$gpu" ./scripts/run_phase11_main.sh "$cfg" "$out"
}

case "${1:-}" in
  round1)
    run_cfg 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headprefixblend_12713_13111_readout_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_13111_delayed_lw005_stop048_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headprefixblend_12713_13111_readout_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_13111_delayed_lw005_stop048_longer_lowlr_seed13811_p1 &
    run_cfg 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headprefixblend_12713_13111_readout_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_13111_delayed_lw00625_stop048_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headprefixblend_12713_13111_readout_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_13111_delayed_lw00625_stop048_longer_lowlr_seed13812_p1 &
    wait
    ;;
  round2)
    run_cfg 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headprefixblend_12713_13111_readout_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_13111_delayed_lw0075_stop048_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headprefixblend_12713_13111_readout_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_13111_delayed_lw0075_stop048_longer_lowlr_seed13813_p1 &
    run_cfg 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headprefixblend_12713_13111_readout_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_13111_delayed_lw00875_stop048_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headprefixblend_12713_13111_readout_bal95_route05_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_13111_delayed_lw00875_stop048_longer_lowlr_seed13814_p1 &
    wait
    ;;
  *)
    echo "usage: $0 {round1|round2}" >&2
    exit 1
    ;;
esac

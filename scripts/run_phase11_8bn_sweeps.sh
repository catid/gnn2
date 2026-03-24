#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

ROUND="${1:-round1}"

run_cfg() {
  local gpu="$1"
  local cfg="$2"
  local out="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" ./scripts/run_phase11_main.sh "$cfg" "$out" &
}

case "$ROUND" in
  round1)
    run_cfg 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_finalq_lw025_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_finalq_lw025_longer_lowlr_seed12911_p1
    run_cfg 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_finalq_lw050_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_finalq_lw050_longer_lowlr_seed12912_p1
    ;;
  round2)
    run_cfg 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_finalq_lw050_start48_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_finalq_lw050_start48_longer_lowlr_seed12913_p1
    run_cfg 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_delayed_lw050_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_delayed_lw050_longer_lowlr_seed12914_p1
    ;;
  *)
    echo "usage: $0 [round1|round2]" >&2
    exit 1
    ;;
esac

wait

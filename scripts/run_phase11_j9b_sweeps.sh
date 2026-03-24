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
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_finalq_lw0075_s48_llr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_finalq_lw0075_s48_seed14011_p1 &
    run_cfg 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_finalq_lw010_s48_llr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_finalq_lw010_s48_seed14012_p1 &
    wait
    ;;
  round2)
    run_cfg 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_finalq_lw0125_s48_llr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_finalq_lw0125_s48_seed14013_p1 &
    run_cfg 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_finalq_lw015_s48_llr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_finalq_lw015_s48_seed14014_p1 &
    wait
    ;;
  *)
    echo "usage: $0 {round1|round2}" >&2
    exit 1
    ;;
esac

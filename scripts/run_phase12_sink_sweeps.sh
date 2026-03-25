#!/usr/bin/env bash
set -euo pipefail

results_root="${1:-results/phase12_dev}"
mode="${2:-round1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

launch_pair() {
  local gpu_a="$1"
  local cfg_a="$2"
  local out_a="$3"
  local gpu_b="$4"
  local cfg_b="$5"
  local out_b="$6"

  CUDA_VISIBLE_DEVICES="${gpu_a}" nohup ./scripts/run_phase12_main.sh "${cfg_a}" "${out_a}" '' 1 \
    > "logs/phase12/$(basename "${out_a}").log" 2>&1 &
  CUDA_VISIBLE_DEVICES="${gpu_b}" nohup ./scripts/run_phase12_main.sh "${cfg_b}" "${out_b}" '' 1 \
    > "logs/phase12/$(basename "${out_b}").log" 2>&1 &
  wait
}

case "${mode}" in
  round1)
    launch_pair \
      0 \
      configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix2_full.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix2_full_seed15081_p1" \
      1 \
      configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix2_readoutonly.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix2_readoutonly_seed15082_p1"
    ;;
  round2)
    launch_pair \
      0 \
      configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix4_full.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix4_full_seed15083_p1" \
      1 \
      configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix4_readoutonly.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix4_readoutonly_seed15084_p1"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [round1|round2]" >&2
    exit 1
    ;;
esac

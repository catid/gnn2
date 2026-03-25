#!/usr/bin/env bash
set -euo pipefail

results_root="${1:-results/phase12_dev}"
mode="${2:-anchor}"
mkdir -p "${results_root}"

run_bg() {
  local config="$1"
  local out="$2"
  local gpu="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" ./scripts/run_phase12_main.sh "${config}" "${out}" "" 1 \
    >/tmp/"$(basename "${out}")".log 2>&1 &
}

case "${mode}" in
  anchor)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_anchor_multiview_querygated.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_anchor_multiview_querygated_seed15012_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_anchor_queryfilm.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_anchor_queryfilm_seed15013_p1" 1
    wait
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1821_anchor_multiview_querygated_fq5.yaml \
      "${results_root}/hard_st_b_v2_teacher1821_anchor_multiview_querygated_fq5_seed15014_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1201_anchor_readoutprefix_13411.yaml \
      "${results_root}/hard_st_b_v2_teacher1201_anchor_readoutprefix_13411_seed15015_p1" 1
    wait
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1879_anchor_queryreadout.yaml \
      "${results_root}/hard_st_b_v2_teacher1879_anchor_queryreadout_seed15016_p1" 0
    wait
    ;;
  bank)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalbank_sinklate_querygated_finalqweight.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalbank_sinklate_querygated_finalqweight_seed15021_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalbank_sinkreadoutlate_crossattn_routehist_finalqweight.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalbank_sinkreadoutlate_crossattn_routehist_finalqweight_seed15022_p1" 1
    wait
    ;;
  factorized)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_sink_query_finalqweight.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_sink_query_finalqweight_seed15031_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_finalqweight.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_finalqweight_seed15032_p1" 1
    wait
    ;;
  windows)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1201_temporalwindow_contiguous_24_72.yaml \
      "${results_root}/hard_st_b_v2_teacher1201_temporalwindow_contiguous_24_72_seed15041_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1201_temporalwindow_sparse_24_72.yaml \
      "${results_root}/hard_st_b_v2_teacher1201_temporalwindow_sparse_24_72_seed15042_p1" 1
    wait
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [anchor|bank|factorized|windows]" >&2
    exit 1
    ;;
esac

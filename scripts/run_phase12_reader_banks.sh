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
  bank_r2a)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalbank_sinklate_querygated_w12_final.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalbank_sinklate_querygated_w12_final_seed15023_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalbank_sinklate_querygated_exitdelay_w12.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalbank_sinklate_querygated_exitdelay_w12_seed15024_p1" 1
    wait
    ;;
  bank_r2b)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalbank_sinkpacket_queryfilm_exit_routehist.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalbank_sinkpacket_queryfilm_exit_routehist_seed15025_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalbank_readoutlate_querygated_finalquery.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalbank_readoutlate_querygated_finalquery_seed15026_p1" 1
    wait
    ;;
  bank_r2c)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalbank_sinkreadout_crossattn_exit_noroute.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalbank_sinkreadout_crossattn_exit_noroute_seed15027_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalbank_sinklate_queryfilm_delaypeak_entropy.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalbank_sinklate_queryfilm_delaypeak_entropy_seed15028_p1" 1
    wait
    ;;
  factorized)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_sink_query_finalqweight.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_sink_query_finalqweight_seed15031_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_finalqweight.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_finalqweight_seed15032_p1" 1
    wait
    ;;
  factorized_r2)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_sink_query_gated.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_sink_query_gated_seed15033_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_sink_query_bilinear.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_sink_query_bilinear_seed15034_p1" 1
    wait
    ;;
  factorized_r3)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_gated_route.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_gated_route_seed15035_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1821_factorized_temporalbank_query_gated.yaml \
      "${results_root}/hard_st_b_v2_teacher1821_factorized_temporalbank_query_gated_seed15036_p1" 1
    wait
    ;;
  factorized_aux_r2)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_sink_query_gated_payloadaux.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_sink_query_gated_payloadaux_seed15037_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_sink_query_gated_payloadqueryaux.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_sink_query_gated_payloadqueryaux_seed15038_p1" 1
    wait
    ;;
  factorized_aux_r3)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_gated_route_payloadqueryaux.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_gated_route_payloadqueryaux_seed15039_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1821_factorized_temporalbank_query_gated_payloadqueryaux.yaml \
      "${results_root}/hard_st_b_v2_teacher1821_factorized_temporalbank_query_gated_payloadqueryaux_seed15040_p1" 1
    wait
    ;;
  windows)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1201_temporalwindow_contiguous_24_72.yaml \
      "${results_root}/hard_st_b_v2_teacher1201_temporalwindow_contiguous_24_72_seed15041_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1201_temporalwindow_sparse_24_72.yaml \
      "${results_root}/hard_st_b_v2_teacher1201_temporalwindow_sparse_24_72_seed15042_p1" 1
    wait
    ;;
  windows_r2)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1201_temporalwindow_half_24_48.yaml \
      "${results_root}/hard_st_b_v2_teacher1201_temporalwindow_half_24_48_seed15043_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1201_temporalwindow_half_48_72.yaml \
      "${results_root}/hard_st_b_v2_teacher1201_temporalwindow_half_48_72_seed15044_p1" 1
    wait
    ;;
  windows_r3)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1201_temporalwindow_overlap_36_72.yaml \
      "${results_root}/hard_st_b_v2_teacher1201_temporalwindow_overlap_36_72_seed15045_p1" 0
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1874_temporalwindow_contiguous_exit_12.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_temporalwindow_contiguous_exit_12_seed15046_p1" 1
    wait
    ;;
  windows_r4)
    run_bg configs/phase12/dev/hard_st_benchmark_b_v2_teacher1821_temporalwindow_sparse_exit_12.yaml \
      "${results_root}/hard_st_b_v2_teacher1821_temporalwindow_sparse_exit_12_seed15047_p1" 0
    wait
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [anchor|bank|bank_r2a|bank_r2b|bank_r2c|factorized|factorized_r2|factorized_r3|factorized_aux_r2|factorized_aux_r3|windows|windows_r2|windows_r3|windows_r4]" >&2
    exit 1
    ;;
esac

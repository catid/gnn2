#!/usr/bin/env bash
set -euo pipefail

results_root="${1:-results/phase13_dev}"
mode="${2:-anchor}"
mkdir -p "${results_root}"

run_bg() {
  local config="$1"
  local out="$2"
  local gpu="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" ./scripts/run_phase13_main.sh "${config}" "${out}" "" 1 \
    >/tmp/"$(basename "${out}")".log 2>&1 &
}

case "${mode}" in
  anchor)
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_anchor_temporalbank_bilinear_exit_routehist.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_anchor_temporalbank_bilinear_exit_routehist_seed16011_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_anchor_factorized_temporalbank_query_bilinear_noroute.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_anchor_factorized_temporalbank_query_bilinear_noroute_seed16012_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_anchor_factorized_temporalbank_query_bilinear_noroute.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_anchor_factorized_temporalbank_query_bilinear_noroute_seed16012_rerun1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1821_anchor_factorized_temporalbank_query_bilinear_noroute.yaml \
      "${results_root}/hard_st_b_v2_teacher1821_anchor_factorized_temporalbank_query_bilinear_noroute_seed16013_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1879_anchor_queryreadout.yaml \
      "${results_root}/hard_st_b_v2_teacher1879_anchor_queryreadout_seed16014_p1" 0
    wait
    ;;
  a_r1)
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectproxy_locked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectproxy_locked_seed16021_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectlexi_lockedroute_lockedfqacc.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectlexi_lockedroute_lockedfqacc_seed16022_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxlockedfqhld_fq5.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxlockedfqhld_fq5_seed16023_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_teacher15057_delayedkl015_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_teacher15057_delayedkl015_selectlocked_seed16024_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_teacher15051_delayedkl015_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_teacher15051_delayedkl015_selectlocked_seed16025_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_exitmask_trigger_until096_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_exitmask_trigger_until096_selectlocked_seed16026_p1" 1
    wait
    ;;
  a_r2)
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_antiexit_fqinclusive_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_antiexit_fqinclusive_selectlocked_seed16027_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_waitloss_fqonly_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_waitloss_fqonly_selectlocked_seed16028_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_resume15057_teacher15057_delayedkl015_selectlocked_lowlr.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_resume15057_teacher15057_delayedkl015_selectlocked_lowlr_seed16029_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_teacher15057_delayedkl030_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_teacher15057_delayedkl030_selectlocked_seed16030_p1" 1
    wait
    ;;
  a_r3)
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_paramanchor15057_w0005_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_paramanchor15057_w0005_selectlocked_seed16031_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_paramanchor15057_w0010_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_paramanchor15057_w0010_selectlocked_seed16032_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_teacher15057_delayedkl015_paramanchor15057_w0005_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_teacher15057_delayedkl015_paramanchor15057_w0005_selectlocked_seed16033_p1" 0
    wait
    ;;
  b_r1)
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readout_15057extras_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readout_15057extras_selectlocked_seed16041_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend70_15051readout_15057extras_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend70_15051readout_15057extras_selectlocked_seed16042_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readoutonly_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readoutonly_selectlocked_seed16043_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readout_15057extras_teacher15051_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readout_15057extras_teacher15051_selectlocked_seed16044_p1" 1
    wait
    ;;
  b_r2)
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend85_15051readout_15057extras_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend85_15051readout_15057extras_selectlocked_seed16045_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend50_15051readout_15057extras_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend50_15051readout_15057extras_selectlocked_seed16046_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readout_15057extras_teacher15057_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readout_15057extras_teacher15057_selectlocked_seed16047_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readout_15057extras_paramanchor15051_selectlocked.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_15051readout_15057extras_paramanchor15051_selectlocked_seed16048_p1" 1
    wait
    ;;
  c_r1)
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectgeo_val_locked_route.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectgeo_val_locked_route_seed16051_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectlexi_lockedexit_lockedfqacc_fqhacc.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectlexi_lockedexit_lockedfqacc_fqhacc_seed16052_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxfqhld_fq6_selectgeo.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxfqhld_fq6_selectgeo_seed16053_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxlockedfqhld_teacher15057_selectgeo.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxlockedfqhld_teacher15057_selectgeo_seed16054_p1" 1
    wait
    ;;
  c_r2)
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectsum_val_locked_exit_route.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectsum_val_locked_exit_route_seed16055_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectlexi_lockedroute_lockedexit_fqhacc_ldacc.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_selectlexi_lockedroute_lockedexit_fqhacc_ldacc_seed16056_p1" 1
    wait
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxfulllocked_fq8_selectgeo.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxfulllocked_fq8_selectgeo_seed16057_p1" 0
    run_bg configs/phase13/dev/hard_st_benchmark_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxlockedlong_teacher15051_selectlexi.yaml \
      "${results_root}/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_auxlockedlong_teacher15051_selectlexi_seed16058_p1" 1
    wait
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [anchor|a_r1|a_r2|a_r3|b_r1|b_r2|c_r1|c_r2]" >&2
    exit 1
    ;;
esac

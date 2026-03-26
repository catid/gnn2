#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
results_root="${1:-results/phase15_dev}"
mode="${2:-anchor}"

mkdir -p "${results_root}"

run_bg() {
  local config="$1"
  local out="$2"
  local gpu="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "$root/scripts/run_phase15_main.sh" "$config" --results-dir "$out" \
    >/tmp/"$(basename "${out}")".log 2>&1 &
}

case "${mode}" in
  anchor)
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_anchor16045_selectlocked.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_anchor16045_selectlocked_seed18011_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_anchor16081_selectlocked.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_anchor16081_selectlocked_seed18012_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_anchor17024_selectlocked.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_anchor17024_selectlocked_seed18013_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_anchor17031_selectlocked.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_anchor17031_selectlocked_seed18014_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1821_anchor17101_selectlexi_hardslice.yaml" \
      "${results_root}/hard_st_b_v2_teacher1821_anchor17101_selectlexi_hardslice_seed18015_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1879_anchor17103_selectlexi_hardslice.yaml" \
      "${results_root}/hard_st_b_v2_teacher1879_anchor17103_selectlexi_hardslice_seed18016_p1" 1
    wait
    ;;
  a_r1)
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_base_seed18021_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_base_seed18031_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_teacher16081_contentmse010_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_teacher16081_contentmse010_fqonly_seed18024_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_teacher16081_contentmse010_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_teacher16081_contentmse010_fqonly_seed18032_p1" 1
    wait
    ;;
  a_r2)
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_shared_mean_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_shared_mean_base_seed18023_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot4_shared_mean_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot4_shared_mean_base_seed18026_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_mean_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_mean_base_seed18027_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_shared_attention_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_shared_attention_base_seed18028_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_learnedinit_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_learnedinit_base_seed18029_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot4_independent_learnedinit_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot4_independent_learnedinit_base_seed18030_p1" 1
    wait
    ;;
  b_r1)
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv8_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv8_base_seed18033_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_contenthidden_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_contenthidden_base_seed18034_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_finalsink_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_finalsink_base_seed18035_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_trajectorybank_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_trajectorybank_base_seed18036_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv8_trajectorybank_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv8_trajectorybank_base_seed18037_p1" 0
    wait
    ;;
  c_r1)
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_payloadaux040_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_payloadaux040_fqonly_seed18038_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_queryaux025_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_queryaux025_fqonly_seed18039_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_protopull010_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_protopull010_fqonly_seed18040_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_payloadaux040_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_payloadaux040_fqonly_seed18045_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_queryaux025_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_queryaux025_fqonly_seed18046_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_protopull010_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_protopull010_fqonly_seed18047_p1" 1
    wait
    ;;
  d_r1)
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_base_seed18043_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_base.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_base_seed18053_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_payloadaux040_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_payloadaux040_fqonly_seed18055_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_queryaux025_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_queryaux025_fqonly_seed18056_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_payloadaux040_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_payloadaux040_fqonly_seed18057_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_queryaux025_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_queryaux025_fqonly_seed18058_p1" 1
    wait
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [anchor|a_r1|a_r2|b_r1|c_r1|d_r1]" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
anchor_root="${1:-results/phase15_anchor}"
dev_root="${2:-results/phase15_dev}"

mkdir -p "$anchor_root" "$dev_root"

wait_for_file() {
  local path="$1"
  until [[ -f "$path" ]]; do
    sleep 60
  done
}

mode_done() {
  local mode="$1"
  case "$mode" in
    a_r1)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_teacher16081_contentmse010_fqonly_seed18024_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_teacher16081_contentmse010_fqonly_seed18032_p1/summary.json" ]]
      ;;
    a_r2)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_shared_mean_base_seed18023_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot4_shared_mean_base_seed18026_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_mean_base_seed18027_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_shared_attention_base_seed18028_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_learnedinit_base_seed18029_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot4_independent_learnedinit_base_seed18030_p1/summary.json" ]]
      ;;
    b_r1)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv8_base_seed18033_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_contenthidden_base_seed18034_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_finalsink_base_seed18035_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_trajectorybank_base_seed18036_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv8_trajectorybank_base_seed18037_p1/summary.json" ]]
      ;;
    c_r1)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_payloadaux040_fqonly_seed18038_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_queryaux025_fqonly_seed18039_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_protopull010_fqonly_seed18040_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_payloadaux040_fqonly_seed18045_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_queryaux025_fqonly_seed18046_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_protopull010_fqonly_seed18047_p1/summary.json" ]]
      ;;
    d_r1)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_base_seed18043_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_base_seed18053_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_payloadaux040_fqonly_seed18055_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_dualanchor16081_queryaux025_fqonly_seed18056_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_payloadaux040_fqonly_seed18057_p1/summary.json" &&
         -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_queryaux025_fqonly_seed18058_p1/summary.json" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

run_mode_if_needed() {
  local mode="$1"
  if mode_done "$mode"; then
    echo "skip ${mode}: sentinel summary already present"
    return 0
  fi
  echo "run ${mode}"
  "$root/scripts/run_phase15_cluster_scouts.sh" "$dev_root" "$mode"
}

wait_for_file "${anchor_root}/hard_st_b_v2_teacher1821_anchor17101_selectlexi_hardslice_seed18015_p1/summary.json"
wait_for_file "${anchor_root}/hard_st_b_v2_teacher1879_anchor17103_selectlexi_hardslice_seed18016_p1/summary.json"

for mode in a_r1 a_r2 b_r1 c_r1 d_r1; do
  run_mode_if_needed "$mode"
done

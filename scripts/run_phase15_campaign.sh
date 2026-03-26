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
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_teacher16081_contentmse010_fqonly_seed18032_p1/summary.json" ]]
      ;;
    a_r2)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot4_independent_learnedinit_base_seed18030_p1/summary.json" ]]
      ;;
    b_r1)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv8_trajectorybank_base_seed18037_p1/summary.json" ]]
      ;;
    c_r1)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_protopull010_fqonly_seed18047_p1/summary.json" ]]
      ;;
    d_r1)
      [[ -f "${dev_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_dualanchor16081_queryaux025_fqonly_seed18058_p1/summary.json" ]]
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

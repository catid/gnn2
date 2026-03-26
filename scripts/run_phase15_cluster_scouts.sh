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
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_base_seed18022_p1" 1
    wait
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_teacher16081_contentmse010_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_multislot2_independent_teacher16081_contentmse010_fqonly_seed18023_p1" 0
    run_bg \
      "$root/configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_teacher16081_contentmse010_fqonly.yaml" \
      "${results_root}/hard_st_b_v2_teacher1874_contentpath_resume16045_sidecarkv4_teacher16081_contentmse010_fqonly_seed18024_p1" 1
    wait
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [anchor|a_r1]" >&2
    exit 1
    ;;
esac

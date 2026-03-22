#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export MAKEFLAGS="${MAKEFLAGS:--j16}"

root="${1:-results/phase8_dev}"
cluster="${2:-explore}"
mkdir -p "${root}"

declare -a configs
case "${cluster}" in
  explore)
    configs=(
      "hard_st_benchmark_b_v2_controlsticky_keepalive_detachprefix96.yaml"
      "hard_st_benchmark_b_v2_controlsticky_keepalive_latewindow64.yaml"
      "hard_st_benchmark_b_v2_controlsticky_keepalive_latewindow32.yaml"
      "hard_st_benchmark_b_v2_controlsticky_keepalive_detachprefix96_latewindow64.yaml"
      "hard_st_benchmark_b_v2_controlsticky_keepalive_trunc32_detachprefix96.yaml"
      "hard_st_benchmark_b_v2_controlsticky_keepalive_detachprefix64_latewindow32.yaml"
      "hard_st_benchmark_b_v2_monotone_wait_direct_phase8.yaml"
      "hard_st_benchmark_b_v2_controlsetclear_memoryheavy_exitselect_phase8.yaml"
      "hard_st_benchmark_b_v2_controlsetclear_waitstate_memoryheavy_phase8.yaml"
      "hard_st_benchmark_b_v2_recurrent_waitact_curriculum_phase8.yaml"
      "hard_st_benchmark_b_v2_controlwaitact_releaseaux_phase8.yaml"
      "hard_st_benchmark_b_v2_controlwaitact_unlocklate_phase8.yaml"
      "hard_st_benchmark_b_v2_controlsticky_keepalive_setclearhybrid_phase8.yaml"
      "reinforce_benchmark_b_v2_controlsticky_keepalive_base_phase8.yaml"
      "reinforce_benchmark_b_v2_controlsticky_keepalive_entropyhigh_phase8.yaml"
      "reinforce_benchmark_b_v2_controlsticky_keepalive_controlaux_phase8.yaml"
      "reinforce_benchmark_b_v2_controlsticky_keepalive_exitmask_phase8.yaml"
      "reinforce_benchmark_b_v2_controlsticky_keepalive_exitmask_controlaux_phase8.yaml"
    )
    ;;
  recover)
    configs=(
      "hard_st_benchmark_b_v2_teacher1802_refine_memoryonly.yaml"
      "hard_st_benchmark_b_v2_teacher1802_refine_memoryreadout.yaml"
      "hard_st_benchmark_b_v2_teacher1802_refine_sinkcore.yaml"
      "hard_st_benchmark_b_v2_teacher1811_refine_memoryonly.yaml"
      "hard_st_benchmark_b_v2_teacher1811_refine_memoryreadout.yaml"
      "hard_st_benchmark_b_v2_teacher1811_refine_sinkcore.yaml"
    )
    ;;
  *)
    echo "unknown cluster mode: ${cluster}" >&2
    echo "usage: $0 [results-root] [explore|recover]" >&2
    exit 1
    ;;
esac

run_pair() {
  local cfg_a="$1"
  local cfg_b="$2"
  local stem_a="${cfg_a%.yaml}"
  local stem_b="${cfg_b%.yaml}"
  CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase8_main.sh "configs/phase8/dev/${cfg_a}" "${root}/${stem_a}" &
  local pid_a=$!
  CUDA_VISIBLE_DEVICES=1 ./scripts/run_phase8_main.sh "configs/phase8/dev/${cfg_b}" "${root}/${stem_b}" &
  local pid_b=$!
  wait "${pid_a}" "${pid_b}"
}

for ((i = 0; i < ${#configs[@]}; i += 2)); do
  if (( i + 1 < ${#configs[@]} )); then
    run_pair "${configs[i]}" "${configs[i+1]}"
  else
    stem="${configs[i]%.yaml}"
    CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase8_main.sh "configs/phase8/dev/${configs[i]}" "${root}/${stem}"
  fi
done

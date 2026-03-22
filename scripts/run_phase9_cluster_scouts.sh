#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export MAKEFLAGS="${MAKEFLAGS:--j16}"

root="${1:-results/phase9_dev}"
cluster="${2:-audit}"
mkdir -p "${root}"

declare -a configs
case "${cluster}" in
  audit)
    configs=(
      "hard_st_benchmark_b_v2_teacher1879_refine_readoutonly_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1879_refine_sinkreadout_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1821_refine_memoryreadout_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874.yaml"
    )
    ;;
  fragile)
    configs=(
      "hard_st_benchmark_b_v2_teacher1879_refine_readoutonly_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1879_refine_sinkreadout_longer_lowlr.yaml"
    )
    ;;
  medium)
    configs=(
      "hard_st_benchmark_b_v2_teacher1821_refine_memoryreadout_longer_lowlr.yaml"
    )
    ;;
  *)
    echo "unknown cluster mode: ${cluster}" >&2
    echo "usage: $0 [results-root] [audit|fragile|medium]" >&2
    exit 1
    ;;
esac

run_pair() {
  local cfg_a="$1"
  local cfg_b="$2"
  local stem_a="${cfg_a%.yaml}"
  local stem_b="${cfg_b%.yaml}"
  CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase9_main.sh "configs/phase8/dev/${cfg_a}" "${root}/${stem_a}" &
  local pid_a=$!
  CUDA_VISIBLE_DEVICES=1 ./scripts/run_phase9_main.sh "configs/phase8/dev/${cfg_b}" "${root}/${stem_b}" &
  local pid_b=$!
  wait "${pid_a}" "${pid_b}"
}

for ((i = 0; i < ${#configs[@]}; i += 2)); do
  if (( i + 1 < ${#configs[@]} )); then
    run_pair "${configs[i]}" "${configs[i+1]}"
  else
    stem="${configs[i]%.yaml}"
    CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase9_main.sh "configs/phase8/dev/${configs[i]}" "${root}/${stem}"
  fi
done

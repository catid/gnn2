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
      "hard_st_benchmark_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectroute_seed1879.yaml"
      "hard_st_benchmark_b_v2_teacher1879_refine_readoutonly_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1821_refine_queryreadout_contentdistill_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_queryreadout_finalqweight_longer_lowlr.yaml"
    )
    ;;
  fragile)
    configs=(
      "hard_st_benchmark_b_v2_teacher1879_refine_queryreadout_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1879_refine_queryreadout_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1879_refine_queryreadout_contentdistill_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1879_refine_sinkqueryreadout_contentdistill_longer_lowlr.yaml"
    )
    ;;
  medium)
    configs=(
      "hard_st_benchmark_b_v2_teacher1821_refine_queryreadout_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1821_refine_queryreadout_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1821_refine_queryreadout_contentdistill_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1821_refine_sinkqueryreadout_contentdistill_longer_lowlr.yaml"
    )
    ;;
  reader_explore)
    configs=(
      "hard_st_benchmark_b_v2_teacher1874_refine_queryfilm_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_querygated_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1821_refine_queryfilm_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1821_refine_querygated_finalqweight_longer_lowlr.yaml"
    )
    ;;
  head_es)
    configs=(
      "hybrid_es_benchmark_b_v2_teacher1874_headonly_queryreadout_resume_from9122.yaml"
      "hybrid_es_benchmark_b_v2_teacher1874_headonly_sinkqueryreadout_resume_from9122.yaml"
      "hybrid_es_benchmark_b_v2_teacher1821_headonly_queryreadout_resume_from9116.yaml"
      "hybrid_es_benchmark_b_v2_teacher1821_headonly_sinkqueryreadout_resume_from9116.yaml"
    )
    ;;
  *)
    echo "unknown cluster mode: ${cluster}" >&2
    echo "usage: $0 [results-root] [audit|fragile|medium|reader_explore|head_es]" >&2
    exit 1
    ;;
esac

seed_for_config() {
  awk '/^[[:space:]]*seed:/ {print $2; exit}' "$1"
}

results_dir_for_config() {
  local cfg_path="$1"
  local stem="${cfg_path%.yaml}"
  local seed
  seed="$(seed_for_config "${cfg_path}")"
  local base="${stem/hard_st_benchmark_b_v2/hard_st_b_v2}"
  base="${base/hybrid_es_benchmark_b_v2/hybrid_es_b_v2}"
  echo "${root}/${base}_seed${seed}_p1"
}

run_pair() {
  local cfg_a="$1"
  local cfg_b="$2"
  local path_a="configs/phase9/dev/${cfg_a}"
  local path_b="configs/phase9/dev/${cfg_b}"
  local out_a
  local out_b
  out_a="$(results_dir_for_config "${path_a}")"
  out_b="$(results_dir_for_config "${path_b}")"
  CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase9_main.sh "${path_a}" "${out_a}" &
  local pid_a=$!
  CUDA_VISIBLE_DEVICES=1 ./scripts/run_phase9_main.sh "${path_b}" "${out_b}" &
  local pid_b=$!
  wait "${pid_a}" "${pid_b}"
}

for ((i = 0; i < ${#configs[@]}; i += 2)); do
  if (( i + 1 < ${#configs[@]} )); then
    run_pair "${configs[i]}" "${configs[i+1]}"
  else
    path="configs/phase9/dev/${configs[i]}"
    out="$(results_dir_for_config "${path}")"
    CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase9_main.sh "${path}" "${out}"
  fi
done

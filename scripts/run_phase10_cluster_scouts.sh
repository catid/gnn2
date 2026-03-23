#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export MAKEFLAGS="${MAKEFLAGS:--j16}"

root="${1:-results/phase10_dev}"
cluster="${2:-anchor}"
mkdir -p "${root}"

declare -a configs
case "${cluster}" in
  anchor)
    configs=(
      "hard_st_benchmark_b_v2_teacher1874_anchor_querygated_finalqweight.yaml"
      "hard_st_benchmark_b_v2_teacher1874_anchor_queryfilm.yaml"
      "hard_st_benchmark_b_v2_teacher1821_anchor_readoutonly.yaml"
      "hard_st_benchmark_b_v2_teacher1879_anchor_queryreadout_finalqweight.yaml"
    )
    ;;
  multiview)
    configs=(
      "hard_st_benchmark_b_v2_teacher1874_refine_multiview_sink_only_concat_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_multiview_packet_only_concat_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacket_concat_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_queryfilm_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_crossattn_finalqweight_longer_lowlr.yaml"
    )
    ;;
  adapter)
    configs=(
      "hard_st_benchmark_b_v2_teacher1874_refine_querygated_adapter_affine_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_querygated_adapter_lowrank_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_querygated_adapter_residual_finalqweight_longer_lowlr.yaml"
      "hard_st_benchmark_b_v2_teacher1874_refine_queryfilm_adapter_lowrank_longer_lowlr.yaml"
    )
    ;;
  *)
    echo "unknown cluster mode: ${cluster}" >&2
    echo "usage: $0 [results-root] [anchor|multiview|adapter]" >&2
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
  local path_a="configs/phase10/dev/${cfg_a}"
  local path_b="configs/phase10/dev/${cfg_b}"
  local out_a
  local out_b
  out_a="$(results_dir_for_config "${path_a}")"
  out_b="$(results_dir_for_config "${path_b}")"
  CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase10_main.sh "${path_a}" "${out_a}" &
  local pid_a=$!
  CUDA_VISIBLE_DEVICES=1 ./scripts/run_phase10_main.sh "${path_b}" "${out_b}" &
  local pid_b=$!
  wait "${pid_a}" "${pid_b}"
}

for ((i = 0; i < ${#configs[@]}; i += 2)); do
  if (( i + 1 < ${#configs[@]} )); then
    run_pair "${configs[i]}" "${configs[i+1]}"
  else
    path="configs/phase10/dev/${configs[i]}"
    out="$(results_dir_for_config "${path}")"
    CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase10_main.sh "${path}" "${out}"
  fi
done

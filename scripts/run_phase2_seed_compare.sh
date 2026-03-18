#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-results/phase2_final}"

export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export MAKEFLAGS="${MAKEFLAGS:--j16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

run_one() {
  local gpu="$1"
  local config="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    uv run python -m src.train.run \
    --config "${config}" \
    --results-dir "${out_dir}" \
    >"${out_dir}/launcher.log" 2>&1
}

run_pair() {
  local config_a="$1"
  local out_a="$2"
  local config_b="$3"
  local out_b="$4"

  run_one 0 "${config_a}" "${out_a}" &
  local pid_a=$!
  run_one 1 "${config_b}" "${out_b}" &
  local pid_b=$!

  wait "${pid_a}"
  wait "${pid_b}"
}

run_pair \
  "configs/phase2/final/hard_st_benchmark_b_v2_gatedblend_seed601.yaml" \
  "${OUT_ROOT}/hard_st_b_v2_gatedblend_seed601" \
  "configs/phase2/final/hard_st_benchmark_b_v2_maskcurr_seed611.yaml" \
  "${OUT_ROOT}/hard_st_b_v2_maskcurr_seed611"

run_pair \
  "configs/phase2/final/hard_st_benchmark_b_v2_gatedblend_seed602.yaml" \
  "${OUT_ROOT}/hard_st_b_v2_gatedblend_seed602" \
  "configs/phase2/final/hard_st_benchmark_b_v2_maskcurr_seed612.yaml" \
  "${OUT_ROOT}/hard_st_b_v2_maskcurr_seed612"

run_pair \
  "configs/phase2/final/hard_st_benchmark_b_v2_gatedblend_seed603.yaml" \
  "${OUT_ROOT}/hard_st_b_v2_gatedblend_seed603" \
  "configs/phase2/final/hard_st_benchmark_b_v2_maskcurr_seed613.yaml" \
  "${OUT_ROOT}/hard_st_b_v2_maskcurr_seed613"

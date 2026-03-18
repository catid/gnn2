#!/usr/bin/env bash
set -euo pipefail

export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export MAKEFLAGS=-j16
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

TIER="${1:-dev}"
ROOT="${2:-results/${TIER}_suite}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
RUN_SCALING="${RUN_SCALING:-1}"

mkdir -p "$ROOT"

run_single() {
  local config="$1"
  local name="$2"
  uv run python -m src.train.run --config "$config" --results-dir "$ROOT/$name"
}

run_multi() {
  local config="$1"
  local name="$2"
  uv run torchrun --standalone --nproc_per_node=2 -m src.train.run --config "$config" --results-dir "$ROOT/$name"
}

if [[ "$TIER" == "dev" ]]; then
  run_single configs/dev/soft_benchmark_a.yaml soft_benchmark_a
  run_single configs/dev/hard_st_benchmark_a.yaml hard_st_benchmark_a
  run_multi configs/dev/hybrid_es_benchmark_a.yaml hybrid_es_benchmark_a

  run_single configs/dev/soft_benchmark_b.yaml soft_benchmark_b
  run_single configs/dev/hard_st_benchmark_b.yaml hard_st_benchmark_b
  run_multi configs/dev/hybrid_es_benchmark_b.yaml hybrid_es_benchmark_b

  if [[ "$RUN_ABLATIONS" == "1" ]]; then
    run_multi configs/dev/hybrid_es_benchmark_a_rank1.yaml hybrid_es_benchmark_a_rank1
    run_multi configs/dev/hybrid_es_benchmark_a_rank8.yaml hybrid_es_benchmark_a_rank8
    run_multi configs/dev/hybrid_es_benchmark_a_pop16.yaml hybrid_es_benchmark_a_pop16
    run_multi configs/dev/hybrid_es_benchmark_a_nowarm.yaml hybrid_es_benchmark_a_nowarm
    run_multi configs/dev/hybrid_es_benchmark_a_adapters.yaml hybrid_es_benchmark_a_adapters
  fi

  if [[ "$RUN_SCALING" == "1" ]]; then
    run_single configs/dev/soft_benchmark_b_h128.yaml soft_benchmark_b_h128
    run_single configs/dev/hard_st_benchmark_b_h128.yaml hard_st_benchmark_b_h128
    run_multi configs/dev/hybrid_es_benchmark_b_h128.yaml hybrid_es_benchmark_b_h128
    run_multi configs/dev/hybrid_es_benchmark_b_h256.yaml hybrid_es_benchmark_b_h256
  fi
elif [[ "$TIER" == "main" ]]; then
  run_single configs/main/soft_benchmark_a.yaml soft_benchmark_a
  run_single configs/main/hard_st_benchmark_a.yaml hard_st_benchmark_a
  run_multi configs/main/hybrid_es_benchmark_a.yaml hybrid_es_benchmark_a

  run_single configs/main/soft_benchmark_b.yaml soft_benchmark_b
  run_single configs/main/hard_st_benchmark_b.yaml hard_st_benchmark_b
  run_multi configs/main/hybrid_es_benchmark_b.yaml hybrid_es_benchmark_b
else
  echo "Unknown tier: $TIER" >&2
  exit 1
fi

uv run python -m src.utils.report --results-dir "$ROOT" --out docs/experiment_report.md

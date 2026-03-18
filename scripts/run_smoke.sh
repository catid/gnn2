#!/usr/bin/env bash
set -euo pipefail

export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export MAKEFLAGS=-j16
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

ROOT="${1:-results/smoke_suite}"
mkdir -p "$ROOT"

uv run python -m src.train.run --config configs/smoke/soft.yaml --results-dir "$ROOT/soft"
uv run python -m src.train.run --config configs/smoke/hard_st.yaml --results-dir "$ROOT/hard_st"
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run --config configs/smoke/hybrid_es.yaml --results-dir "$ROOT/hybrid_es"
uv run python -m src.utils.report --results-dir "$ROOT" --out docs/experiment_report.md

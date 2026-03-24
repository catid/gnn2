#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_cfg() {
  local gpu=$1
  local cfg=$2
  CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.train.run --config "$cfg"
}

run_cfg 0 configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_sinknative_multiview_sinkbaseline_querygated_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml &
pid0=$!
run_cfg 1 configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_sinknative_multiview_sinkbaseline_queryfilm_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml &
pid1=$!
wait "$pid0" "$pid1"

run_cfg 0 configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_sinkfresh_multiview_sinkbaseline_querygated_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml &
pid2=$!
run_cfg 1 configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_sinkfresh_multiview_sinkbaseline_queryfilm_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml &
pid3=$!
wait "$pid2" "$pid3"

#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export MAKEFLAGS="${MAKEFLAGS:--j16}"

run_a() {
  CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase11_main.sh \
    configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_sinkfrozen_readoutonly_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml \
    results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_sinkfrozen_readoutonly_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr_seed11731_p1
}

run_b() {
  CUDA_VISIBLE_DEVICES=1 ./scripts/run_phase11_main.sh \
    configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_sinkreadout_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr.yaml \
    results/phase11_dev/hard_st_b_v2_teacher1201_partialinit_coremem_sinkreadout_selectproxy_full_locked_fqacc_finalqweight_longer_lowlr_seed11732_p1
}

run_a &
pid_a=$!
run_b &
pid_b=$!

wait "$pid_a"
wait "$pid_b"

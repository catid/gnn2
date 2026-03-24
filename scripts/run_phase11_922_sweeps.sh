#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-round1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"

run_cfg() {
  local gpu="$1"
  local cfg="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m src.train.run --config "${cfg}" &
}

case "${ROUND}" in
  round1)
    run_cfg 0 configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_delayed_lw010_longer_lowlr.yaml
    run_cfg 1 configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_delayed_lw015_longer_lowlr.yaml
    ;;
  round2)
    run_cfg 0 configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_delayed_lw020_longer_lowlr.yaml
    run_cfg 1 configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_probeviews_sinkfinal_mlp_headblend_12514_12424_bal85_route15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_srcdistill_1874_delayed_lw015_start48_longer_lowlr.yaml
    ;;
  *)
    echo "Usage: $0 [round1|round2]" >&2
    exit 1
    ;;
esac

wait

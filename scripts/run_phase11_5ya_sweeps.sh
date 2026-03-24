#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <round1|round2>" >&2
  exit 1
fi

ROUND="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/scripts/run_phase11_main.sh"

run_cfg() {
  local gpu="$1"
  local cfg="$2"
  local out="$3"
  echo "[launch][gpu${gpu}] $(basename "$cfg") -> $(basename "$out")"
  CUDA_VISIBLE_DEVICES="$gpu" "$RUNNER" "$cfg" "$out" &
}

case "$ROUND" in
  round1)
    run_cfg 0 \
      "$ROOT/configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_delayed_lw010_start024_stop072_llr.yaml" \
      "$ROOT/results/phase11_dev/hard_st_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_delayed_lw010_start024_stop072_seed14211_p1"
    run_cfg 1 \
      "$ROOT/configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_delayed_lw010_start048_stop096_llr.yaml" \
      "$ROOT/results/phase11_dev/hard_st_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_delayed_lw010_start048_stop096_seed14212_p1"
    ;;
  round2)
    run_cfg 0 \
      "$ROOT/configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_delayed_lw015_start024_stop072_llr.yaml" \
      "$ROOT/results/phase11_dev/hard_st_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_delayed_lw015_start024_stop072_seed14213_p1"
    run_cfg 1 \
      "$ROOT/configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_delayed_lw015_start048_stop096_llr.yaml" \
      "$ROOT/results/phase11_dev/hard_st_b_v2_teacher1201_pinit_prefix_12713_13111_readout9505_cmix_fq4_selectproxy_src13111_delayed_lw015_start048_stop096_seed14214_p1"
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 1
    ;;
esac

wait

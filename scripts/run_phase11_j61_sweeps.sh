#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

ROUND="${1:-round1}"

build_init() {
  local out_dir="$1"
  local head_mode="$2"
  local head_weight="$3"
  local scaffold_weight="$4"
  local include_sink="$5"

  mkdir -p "$out_dir"
  local out_ckpt="$out_dir/init.pt"
  if [[ -f "$out_ckpt" ]]; then
    echo "exists: $out_ckpt"
    return
  fi

  uv run python - <<'PY' "$out_ckpt" "$head_mode" "$head_weight" "$scaffold_weight" "$include_sink"
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

out_ckpt = Path(sys.argv[1])
head_mode = sys.argv[2]
head_weight = float(sys.argv[3])
scaffold_weight = float(sys.argv[4])
include_sink = sys.argv[5].lower() == "true"

scaffold_path = Path(
    "results/phase11_dev/"
    "20260324_061546_hard_st_benchmark_b_v2_teacher1201_partialinit_coremem_"
    "sinknative_multiview_sinkbaseline_querygated_selectproxy_full_locked_"
    "fqacc_finalqweight_longer_lowlr/hard_st_best.pt"
)
donor_path = Path(
    "results/phase11_dev/"
    "hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_"
    "confirmmix_locked_fq3_longer_lowlr_seed11011_p1/hard_st_best.pt"
)

scaffold_payload = torch.load(scaffold_path, map_location="cpu")
donor_payload = torch.load(donor_path, map_location="cpu")
scaffold_state = scaffold_payload["model"]
donor_state = donor_payload["model"]

result_state = copy.deepcopy(scaffold_state)
prefixes = [
    "query_readout_proj.",
    "multiview_fusion.",
    "multiview_query_proj.",
    "readout.",
]
if include_sink:
    prefixes = ["sink_proj."] + prefixes

for name, tensor in list(result_state.items()):
    if not any(name.startswith(prefix) for prefix in prefixes):
        continue
    donor_tensor = donor_state.get(name)
    scaffold_tensor = scaffold_state.get(name)
    if donor_tensor is None or donor_tensor.shape != tensor.shape:
        continue
    if head_mode == "copy":
        result_state[name] = donor_tensor.clone()
    elif head_mode == "blend":
        result_state[name] = donor_tensor.to(dtype=tensor.dtype) * head_weight + scaffold_tensor.to(dtype=tensor.dtype) * scaffold_weight
    else:
        raise ValueError(f"unknown head_mode: {head_mode}")

payload = {
    "model": result_state,
    "step": 0,
    "extra": {
        "phase": "phase11_j61_synthetic_init",
        "scaffold_checkpoint": str(scaffold_path),
        "donor_checkpoint": str(donor_path),
        "head_mode": head_mode,
        "head_weight": head_weight,
        "scaffold_weight": scaffold_weight,
        "include_sink_proj": include_sink,
    },
}
torch.save(payload, out_ckpt)
print(out_ckpt)
PY
}

mkdir -p results/phase11_init
build_init "results/phase11_init/j61_teacher1201core_native11811sink_11011head_full" "copy" "1.0" "0.0" "false"
build_init "results/phase11_init/j61_teacher1201core_11011fullhead" "copy" "1.0" "0.0" "true"
build_init "results/phase11_init/j61_teacher1201core_native11811sink_11011head85_11811head15" "blend" "0.85" "0.15" "false"
build_init "results/phase11_init/j61_teacher1201core_native11811sink_11011head70_11811head30" "blend" "0.70" "0.30" "false"

run_cfg() {
  local gpu="$1"
  local cfg="$2"
  local out="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" ./scripts/run_phase11_main.sh "$cfg" "$out" &
}

case "$ROUND" in
  round1)
    run_cfg 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_crosssource_querygated_11011head_native11811sink_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_crosssource_querygated_11011head_native11811sink_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed12811_p1
    run_cfg 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_crosssource_querygated_11011fullhead_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_crosssource_querygated_11011fullhead_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed12812_p1
    ;;
  round2)
    run_cfg 0 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_crosssource_querygated_11011head85_11811head15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_crosssource_querygated_11011head85_11811head15_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed12813_p1
    run_cfg 1 \
      configs/phase11/dev/hard_st_benchmark_b_v2_teacher1201_crosssource_querygated_11011head70_11811head30_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr.yaml \
      results/phase11_dev/hard_st_b_v2_teacher1201_crosssource_querygated_11011head70_11811head30_confirmmix_lockedfqhld_fq4_selectproxy_full_locked_fqacc_longer_lowlr_seed12814_p1
    ;;
  *)
    echo "usage: $0 [round1|round2]" >&2
    exit 1
    ;;
esac

wait

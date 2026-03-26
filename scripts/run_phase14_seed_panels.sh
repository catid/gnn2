#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 <config> <seed1> <seed2> <seed3> <seed4> <seed5>" >&2
  exit 2
fi

config="$1"
shift
seeds=("$@")

gpu=0
for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.train.run --config "$config" --seed "$seed" &
  gpu=$((1 - gpu))
done
wait

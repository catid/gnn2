#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu-id> <config> [extra train args...]" >&2
  exit 2
fi

gpu="$1"
config="$2"
shift 2

export CUDA_VISIBLE_DEVICES="$gpu"
exec "$(dirname "$0")/run_phase14_main.sh" "$config" "$@"

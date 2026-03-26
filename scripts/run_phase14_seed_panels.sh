#!/usr/bin/env bash
set -euo pipefail

results_root=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root)
      results_root="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 6 ]]; then
  echo "usage: $0 [--results-root <dir>] <config> <seed1> <seed2> <seed3> <seed4> <seed5>" >&2
  exit 2
fi

config="$1"
shift
seeds=("$@")

if [[ -n "$results_root" ]]; then
  mkdir -p "$results_root"
fi

gpu=0
for seed in "${seeds[@]}"; do
  extra_args=(--seed "$seed")
  if [[ -n "$results_root" ]]; then
    extra_args+=(--results-dir "${results_root}/seed${seed}")
  fi
  "$(dirname "$0")/run_phase14_content_branch.sh" "$gpu" "$config" "${extra_args[@]}" &
  gpu=$((1 - gpu))
done
wait

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

for ((batch_start=0; batch_start<${#seeds[@]}; batch_start+=2)); do
  pids=()
  for gpu in 0 1; do
    idx=$((batch_start + gpu))
    if [[ $idx -ge ${#seeds[@]} ]]; then
      continue
    fi
    seed="${seeds[$idx]}"
    extra_args=(--seed "$seed")
    if [[ -n "$results_root" ]]; then
      seed_results_dir="${results_root}/seed${seed}"
      if [[ -e "$seed_results_dir" ]]; then
        echo "skipping seed ${seed}: existing results dir ${seed_results_dir}" >&2
        continue
      fi
      extra_args+=(--results-dir "$seed_results_dir")
    fi
    "$(dirname "$0")/run_phase14_content_branch.sh" "$gpu" "$config" "${extra_args[@]}" &
    pids+=("$!")
  done
  wait "${pids[@]}"
done

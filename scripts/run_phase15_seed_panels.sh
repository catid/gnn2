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
  batch_seeds=()
  batch_result_dirs=()
  for offset in 0 1; do
    idx=$((batch_start + offset))
    if [[ $idx -ge ${#seeds[@]} ]]; then
      continue
    fi
    seed="${seeds[$idx]}"
    result_dir=""
    if [[ -n "$results_root" ]]; then
      result_dir="${results_root}/seed${seed}"
      if [[ -e "$result_dir" ]]; then
        echo "skipping seed ${seed}: existing results dir ${result_dir}" >&2
        continue
      fi
    fi
    batch_seeds+=("$seed")
    batch_result_dirs+=("$result_dir")
  done
  for gpu in 0 1; do
    if [[ $gpu -ge ${#batch_seeds[@]} ]]; then
      continue
    fi
    seed="${batch_seeds[$gpu]}"
    extra_args=(--seed "$seed")
    result_dir="${batch_result_dirs[$gpu]}"
    if [[ -n "$result_dir" ]]; then
      extra_args+=(--results-dir "$result_dir")
    fi
    "$(dirname "$0")/run_phase15_content_path.sh" "$gpu" "$config" "${extra_args[@]}" &
    pids+=("$!")
  done
  wait "${pids[@]}"
done

#!/usr/bin/env bash
set -euo pipefail

results_root="${1:-results/phase13_dev}"
mode="${2:-anchor}"

case "${mode}" in
  anchor|a_r1|a_r2|b_r1|c_r1)
    ./scripts/run_phase13_stability_sweeps.sh "${results_root}" "${mode}"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [anchor|a_r1|a_r2|b_r1|c_r1]" >&2
    exit 1
    ;;
esac

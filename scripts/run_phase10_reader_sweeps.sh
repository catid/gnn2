#!/usr/bin/env bash
set -euo pipefail

root="${1:-results/phase10_dev}"
mode="${2:-tuned}"

case "${mode}" in
  initial)
    ./scripts/run_phase10_cluster_scouts.sh "${root}" multiview
    ;;
  adapter)
    ./scripts/run_phase10_cluster_scouts.sh "${root}" adapter
    ;;
  tuned)
    ./scripts/run_phase10_cluster_scouts.sh "${root}" multiview
    ./scripts/run_phase10_cluster_scouts.sh "${root}" adapter
    ;;
  *)
    echo "usage: $0 [results-root] [initial|adapter|tuned]" >&2
    exit 1
    ;;
esac

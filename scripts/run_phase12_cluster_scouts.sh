#!/usr/bin/env bash
set -euo pipefail

results_root="${1:-results/phase12_dev}"
mode="${2:-anchor}"

case "${mode}" in
  anchor)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" anchor
    ;;
  bank)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" bank
    ;;
  factorized)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" factorized
    ;;
  windows)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" windows
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [anchor|bank|factorized|windows]" >&2
    exit 1
    ;;
esac

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
  bank_r2a)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" bank_r2a
    ;;
  bank_r2b)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" bank_r2b
    ;;
  bank_r2c)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" bank_r2c
    ;;
  factorized)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" factorized
    ;;
  factorized_r2)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" factorized_r2
    ;;
  factorized_r3)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" factorized_r3
    ;;
  factorized_aux_r2)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" factorized_aux_r2
    ;;
  factorized_aux_r3)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" factorized_aux_r3
    ;;
  windows)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" windows
    ;;
  windows_r2)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" windows_r2
    ;;
  windows_r3)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" windows_r3
    ;;
  windows_r4)
    ./scripts/run_phase12_reader_banks.sh "${results_root}" windows_r4
    ;;
  probe_adapters_r1)
    ./scripts/run_phase12_probe_adapters.sh "${results_root}" round1
    ;;
  probe_adapters_r2)
    ./scripts/run_phase12_probe_adapters.sh "${results_root}" round2
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    echo "usage: $0 [results-root] [anchor|bank|bank_r2a|bank_r2b|bank_r2c|factorized|factorized_r2|factorized_r3|factorized_aux_r2|factorized_aux_r3|windows|windows_r2|windows_r3|windows_r4|probe_adapters_r1|probe_adapters_r2]" >&2
    exit 1
    ;;
esac

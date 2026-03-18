#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-results}"
OUT="${2:-docs/experiment_report.md}"

uv run python -m src.utils.report --results-dir "$ROOT" --out "$OUT"

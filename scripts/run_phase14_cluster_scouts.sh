#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"

"$root/scripts/run_phase14_content_branch.sh" 0 \
  "$root/configs/phase14/dev/hard_st_benchmark_b_v2_teacher1874_anchor16045_selectlocked.yaml" &
pid0=$!

"$root/scripts/run_phase14_content_branch.sh" 1 \
  "$root/configs/phase14/dev/hard_st_benchmark_b_v2_teacher1874_anchor16081_selectlocked.yaml" &
pid1=$!

wait "$pid0"
wait "$pid1"

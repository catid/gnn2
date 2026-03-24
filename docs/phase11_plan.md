# Phase 11 Plan

## Goal

Phase 11 follows directly from the phase-10 result:

- strong frozen `1874` readers already decode almost perfectly on base,
- held confirmations still stall near `full_locked fq_acc ~= 0.30`,
- the next disciplined move is therefore a confirmation-aware objective rather
  than another broad reader-family or adapter sweep.

The active issue is `gnn2-e7s`.

## Main Hypothesis

The best frozen `1874` reader is already strong enough, but the current
objective overfits the base generator. Mixing held-confirm-style benchmark
variants directly into training should improve held-confirm content without
touching routing, memory, or the reader architecture.

## Initial Experiment Block

Start from the phase-10 strong-source multiview query-gated baseline and keep
`memory/router/control` frozen.

Initial configs:

- `11011`: base reader + `full_locked` auxiliary benchmark mix
- `11012`: same, but stronger final-query weighting
- `11013`: `full_locked + finalqueryheavy` auxiliary mix
- `11014`: `full_locked + finalqueryheavy + longdistance` auxiliary mix

## Implementation Notes

- add `training.auxiliary_train_benchmarks` to sample extra batches from
  alternate benchmark configs during supervised training
- keep all auxiliary loss downstream of the frozen routing/memory path
- compare directly against phase-10 `10012` and phase-9 `9142`
- only expand beyond `1874` if one of the initial confirmation-aware runs
  materially improves held confirms

## Commands

```bash
./scripts/run_phase11_main.sh <config> <results-dir> [resume] [nproc_per_node]
./scripts/run_phase11_confirm.sh <run-dir> [extra-eval-config ...]
./scripts/run_phase11_seed_panels.sh <config> <results-root> <resume> <seed1> [seed2 ...]
./scripts/run_phase11_objective_sweeps.sh [results-root] [initial|mixed]
```

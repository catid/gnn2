# Phase 13 Report

## Starting Point

Phase 13 starts from the phase-12 map:

- `15051` was the clean exact-rerun-safe late-route baseline on `1874`,
- `15057` was the highest-upside factorized temporal reader family on base
  behavior,
- but the phase-12 stress story suggested that the high-upside path could drift
  into an easier shortcut basin,
- and the next plausible lever was therefore stabilization rather than more
  generic reader capacity, route-trace features, or broader sink changes.

The central phase-13 question is:

- can the good late-route `1874` reader basin be made optimization-stable
  enough to survive reruns and panels without collapsing into shortcut behavior,
  and if so, does held-confirm content rise above the old ceiling?

## Campaign Coverage

The live ledger is
[phase13_run_matrix.csv](/home/catid/gnn2/docs/phase13_run_matrix.csv).

This report is being filled in incrementally while the remaining verification
panels and reruns are still running. The final version will summarize:

- total substantive runs,
- anchors,
- reruns,
- confirmations,
- seed panels,
- source coverage,
- and the final positive vs strong-mapping exit decision.

## Current Headline Findings

### 1. Conservative continuation clearly stabilizes the late-route basin

The strongest current stable family is still the `16045` bridge from stable
`15051` readout weights into the stronger `15057`-style factorized reader. Its
completed five-seed panel shows:

- base delay-to-final-query mean `0.9967 / 0.9464 / 121.99`
- selected `full_locked` mean `0.9965 / 0.9460 / 121.86`

for `fq_acc / fq_route / fq_exit`.

The important result is qualitative:

- all five seeds stayed in the late-route regime,
- none reproduced the old catastrophic shortcut collapse,
- and the continuation path is therefore materially more stable than the
  original direct `15057` training story.

The newer `16022` and `16066` exact reruns now reinforce the same conclusion:

- `16022_rerun1` cleanly reproduced the intended selection-only late-route
  basin,
- `16066_rerun1` cleanly reproduced the sustained-anchor late-route basin,
- and the first live seeds from both follow-on five-seed panels are also clean
  late-route seeds rather than shortcut relapses.

### 2. The held-confirm ceiling still appears separate from stability

The best verified phase-13 stability recipes so far, including `16022`, `16041`,
`16045`, `16064`, `16066`, and `16072`, still confirm near:

- `full_locked fq_acc ~= 0.312-0.315`
- `full_locked fq_route ~= 0.876-0.877`
- `full_locked fq_exit ~= 115.2-115.5`

So the current phase-13 answer is not yet a positive exit. Stability helps, but
the held-confirm ceiling remains.

### 3. Hard-case weighting and basin-aware selection are not enough on their own

Cluster C is now a fair mapped negative.

The strongest examples, `16055` and `16061_rerun1`, improved summary-time
locked-route slices but independently verified back to the same held-confirm
frontier. That means checkpoint selection alone is not the missing ingredient.

## Live Work

At the time of this draft:

- Cluster D is settled around `16081` as the best refinement result,
- Cluster E is closed as a bounded portability / negative-control sanity map,
- two new five-seed `1874` panels are live on `16022` and `16066`,
- and the next queued verification tranche is the remaining rerun / panel
  coverage on `16041` and `16081` needed to clear the phase-13 verification
  floor.

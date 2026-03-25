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

The newer `16022` and `16066` exact reruns and completed five-seed panels now
reinforce the same conclusion:

- `16022_rerun1` cleanly reproduced the intended selection-only late-route
  basin,
- `16066_rerun1` cleanly reproduced the sustained-anchor late-route basin,
- the completed `16022` panel finished at locked-slice mean
  `0.8530 / 0.9399 / 121.29`,
- and the completed `16066` panel finished at locked-slice mean
  `0.9557 / 0.9391 / 121.34`.

The completed panel stress is more nuanced than the reruns alone:

- `16066` completed with four strong late-route seeds and one weaker but still
  late-route content dip,
- `16022` completed with two strong seeds, one middling recovery seed, one mild
  late-route dip, and one severe late-route content-collapse seed,
- none of those seeds reproduced the old catastrophic shortcut basin,
- so the remaining instability question is no longer only “shortcut or not”.
  It also includes softer non-shortcut content-collapse basins, with `16022`
  showing the harsher version of that failure mode and `16066` showing the
  cleaner current compromise.

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
- the `16022` and `16066` five-seed `1874` panels are now complete,
- `16066` is the cleaner current Cluster A leader while `16022` remains a real
  but high-variance selection-only path,
- the rerun floor is now cleared by completed `16041_rerun1` and
  `16081_rerun1`,
- the extra `16041` bridge panel is now complete at locked mean
  `0.9985 / 0.9399 / 121.24`,
- and the only remaining live work is the final five-seed `16081` refinement
  panel needed to close the phase-13 verification floor cleanly.

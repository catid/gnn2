# Phase 13 Cluster Scorecards

This document is the final phase-13 scorecard. The detailed ledger is in
[phase13_run_matrix.csv](/home/catid/gnn2/docs/phase13_run_matrix.csv).

| Cluster | Half | Best Representative | Current Verdict |
| --- | --- | --- | --- |
| A | fruitful | `16022`, `16064`, `16066` | Strong mapping result. Anti-shortcut stabilization can keep the late-route regime and improve seed stability, but the verified held-confirm frontier still sits near `full_locked fq_acc ~= 0.313`. The completed panel contrast is now sharp: `16022` is high-variance at locked mean `0.8530 / 0.9399 / 121.29`, while `16066` is the cleaner sustained-anchor family at `0.9557 / 0.9391 / 121.34`. |
| B | fruitful | `16041`, `16043`, `16045`, `16072` | Strong mapping result. Conservative continuation from `15051` into the stronger `15057`-style reader clearly stabilizes the late-route basin on `1874`. `16045` is the best fully paneled stable control at locked mean `0.9965 / 0.9460 / 121.86`, and `16041` provides a second stable bridge panel at `0.9985 / 0.9399 / 121.24`, but verified held-confirm content still falls back to the same ceiling. |
| C | fruitful | `16055`, `16061_rerun1`, `16062` | Fair negative. Hard-case weighting and basin-aware checkpoint selection improve summary-time late-route slices, but the gains do not survive independent confirm. |
| D | fruitful | `16081` | Settled. The best post-stability refinement is real on base behavior; its completed panel finished at `0.9998 / 0.9410 / 121.44`, but it still confirms back to the same held-confirm ceiling and does not displace `16045` as the cleaner stable control. |
| E | exploration | `16091`-`16094` | Closed. The `1821` carryover did not survive confirm and the `1879` lines stayed proper negatives. |
| F | exploration gated | not entered | Not yet justified. No stable held-confirm breakthrough exists that would warrant ES polish or a new upstream touch. |

## Cluster Notes

### A. Anti-Shortcut Stability Regularization

- The main positive is scientific, not metric-breaking.
- `16022` showed that lexicographic late-route selection is an honest stability
  improvement over the raw `15057` family.
- Full-window anchoring plus selection (`16066`) kept the late-route regime
  cleanly, but still confirmed at the old held-confirm ceiling.
- The completed panel work sharpens this further:
  `16022` finished with two strong seeds, one middling recovery seed, one mild
  late-route dip, and one severe content-collapse seed, while `16066` finished
  with four strong late-route seeds plus one weaker content dip.
  So Cluster A contains two different stories:
  a high-variance selection-only path and a cleaner sustained-anchor path with
  milder residual variance.

### B. Stable-To-Upside Continuation Bridge

- This is the clearest phase-13 positive so far.
- The `15051 -> 15057` continuation bridge stabilizes the late-route basin
  enough to survive a five-seed panel without recreating the phase-12 shortcut
  collapse.
- The best current bridge is `16045`, which remains the working stable control
  for phase-13 refinement.

### C. Hard-Case Mining And Basin-Aware Selection

- This cluster fairly tested whether the remaining gap was mostly a data-mining
  or checkpoint-selection problem.
- The answer so far is no.
- `16055` and `16061_rerun1` looked stronger on summary-time locked slices but
  both independently confirmed back to the same `full_locked` ceiling.

### D. Post-Stability Reader Refinement

- Settled.
- `16081` is the best refinement result and a real stable base-side win on top
  of `16045`.
- Its completed five-seed panel finished at locked mean
  `0.9998 / 0.9410 / 121.44`: content stayed saturated across all five seeds,
  while route and exit softened mildly relative to `16045`.
- But even `16081` confirms back to the same held-confirm plateau, so Cluster D
  did not justify a new frontier beyond the already paneled stable bridge.

### E. Secondary-Source Sanity

- Closed.
- `16092` was the strongest bounded `1821` selector carryover and looked good
  on summary slices, but confirm fell back to the old medium-source held-
  confirm regime.
- `16093` and `16094` stayed in the expected `1879` bad-source regime with no
  false-positive basin.

# Phase 13 Cluster Scorecards

This document is the live phase-13 scorecard. It will be updated as Cluster D
refinement and Cluster E sanity runs finish. The detailed ledger is in
[phase13_run_matrix.csv](/home/catid/gnn2/docs/phase13_run_matrix.csv).

| Cluster | Half | Best Representative | Current Verdict |
| --- | --- | --- | --- |
| A | fruitful | `16022`, `16064`, `16066` | Strong mapping result. Anti-shortcut stabilization can keep the late-route regime and improve seed stability, but the verified held-confirm frontier still sits near `full_locked fq_acc ~= 0.313`. |
| B | fruitful | `16041`, `16043`, `16045`, `16072` | Strong mapping result. Conservative continuation from `15051` into the stronger `15057`-style reader clearly stabilizes the late-route basin on `1874`, but verified held-confirm content still falls back to the same ceiling. |
| C | fruitful | `16055`, `16061_rerun1`, `16062` | Fair negative. Hard-case weighting and basin-aware checkpoint selection improve summary-time late-route slices, but the gains do not survive independent confirm. |
| D | fruitful | `16081`, `16082` live; `16083`, `16084` queued | In progress. Post-stability refinements are now running on top of the strongest stable bridge boundary `16045`. |
| E | exploration | `16091`-`16094` queued | In progress. Secondary-source sanity runs are queued behind the live Cluster D deck. |
| F | exploration gated | not entered | Not yet justified. No stable held-confirm breakthrough exists that would warrant ES polish or a new upstream touch. |

## Cluster Notes

### A. Anti-Shortcut Stability Regularization

- The main positive is scientific, not metric-breaking.
- `16022` showed that lexicographic late-route selection is an honest stability
  improvement over the raw `15057` family.
- Full-window anchoring plus selection (`16066`) kept the late-route regime
  cleanly, but still confirmed at the old held-confirm ceiling.

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

- Live.
- The current tests are deliberately narrow:
  low-LR delayed-only teacher continuation and low-LR held-confirm-weighted
  continuation on top of `16045`.

### E. Secondary-Source Sanity

- Queued.
- The first four runs port the strongest source-agnostic stability selectors to
  `1821` and the `1879` negative control.

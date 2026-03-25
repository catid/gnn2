# Phase 12 Cluster Scorecards

Phase 12 focused on trajectory-aware frozen-state readers, factorized
content/query reading, portability, stress, and a gated minimal sink change.
The detailed ledger is in
[phase12_run_matrix.csv](/home/catid/gnn2/docs/phase12_run_matrix.csv).

| Cluster | Half | Best Representative | Final Verdict |
| --- | --- | --- | --- |
| A | fruitful | `15051`, `15054`, `15056` | Strong mapping result. Temporal-bank readers are real base-side improvements, but route traces are not the missing ingredient and held confirms stay on the old ceiling. |
| B | fruitful | `15057`, `15038`, `15060` | Strong mapping result. Factorized readers are the strongest base-side family, but they do not break the held-confirm ceiling; the best `1874` family also shows rerun instability into an early-exit shortcut. |
| C | fruitful | `15042`, `15060`, `15063`, `15261-15264`, `15361-15364` | Fair negative on the main hypothesis. Contiguous windows are not a general advantage, and portability to `1821`/`1842` collapses into a weaker earlier-exit regime under panels. |
| D | fruitful | `15065-15068` with `15067_rerun1` | Fair negative. Narrow objective-side distillation on the best reader stayed in the wrong early-exit regime and did not improve held confirms. |
| E | exploration | `15073` with `15073_rerun1` | Serious negative. Probe-guided downstream adapters collapsed content while preserving route and never approached the useful reader regime. |
| F | exploration gated | `15083` with `15083_rerun1` | Strong negative. Minimal keyed sinks are not inert, but they produce stable off-regime shortcuts rather than rescuing held-confirm content. |
| G | exploration optional | not entered | Correctly skipped. No competitive reader frontier justified spending budget on ES polish. |
| H | exploration | `15057_rerun1`, `15060_rerun1`, `15083` confirm, `1821/1842` panels | Strong infrastructure positive. Stress and rerun work changed the phase conclusion by exposing shortcut basins and portability collapse. |

## Cluster Notes

### A. Route-Trace-Conditioned Temporal-Bank Readers

- The core negative is not that temporal banks do nothing.
- The real answer is subtler: temporal banks help, but route traces do not.
- The best stable Cluster A readers are route-blind or nearly so.

### B. Factorized Content / Query Readers

- This cluster produced the strongest base-side `1874` family.
- It also showed that factorization is not enough by itself.
- The critical stress lesson came later: the best family can drift into an
  early-exit shortcut under rerun.

### C. Contiguous-Window Transfer And Portability

- The phase-11 `1201` contiguous-window clue did not become a general phase-12
  reader win.
- Sparse windows often matched or beat contiguous ones.
- Secondary-source portability is weaker and earlier-exit than the strongest
  single seeds suggested.

### D. Best-Reader Objective / Data-Mix Tuning

- Once the better reader existed, the narrow distillation controls were fair to
  test.
- They did not rescue held confirms.
- They mainly reinforced the shortcut-basin story.

### E. Probe-Guided Adapters

- The cleanest negative in the phase.
- Probe-guided initialization was not enough to make tiny downstream adapters
  useful on the best reader boundary.

### F. Minimal Upstream Sink Change

- A worthwhile gated test.
- The answer is not “no effect.”
- The answer is “strong effect in the wrong regime.”

### G. Narrow ES

- Skipped correctly.
- There was no justified competitive frontier that made a head-only ES polish
  worth budget in this phase.

### H. Stress / Confirmation / Generalization

- One of the most valuable artifacts of the phase.
- Without it, phase 12 would have overclaimed both the best `1874` branch and
  the secondary-source portability story.

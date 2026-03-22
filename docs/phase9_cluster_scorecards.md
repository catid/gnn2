# Phase 9 Cluster Scorecards

Phase 9 finished with a narrow frozen-state focus: 108 substantive entries, 75
fruitful and 33 exploration, with cluster counts tracked in
[phase9_run_matrix.csv](/home/catid/gnn2/docs/phase9_run_matrix.csv).

| Cluster | Half | Best Representative | Final Verdict |
| --- | --- | --- | --- |
| A | fruitful | `1874` / `1842` / `1201` audits | Strong positive mapping. Frozen content is highly decodable on strong and medium sources, weak on fragile `1879`. |
| B | fruitful | `9102_rerun1` plus fragile five-seed panel | Fair retired negative. Route and exit timing survive; content stays near chance because the source is weak-content. |
| C | fruitful | `9111` family | Positive but ceiling-limited. Medium teacher-shaped sources support stable head-only recovery, but held-confirm `fq_acc` stays low. |
| D | fruitful | `9150-9154` content-distill family | Competitive but not cleanly dominant. Content-only distillation helps, but it does not decisively beat the best reader family and its lead rerun drifted. |
| E | fruitful | `9301` minimal-safe read-path touch | Retired negative. Tiny touches preserve route on fragile `1879` but do not unlock content. |
| F | exploration | `9132` query-FILM family | Strong positive exploration result. Best alternative-reader family in the phase and near-tied with the strongest aggregate strong-source branch on held confirms. |
| G | exploration | `9202` head-only ES | Mixed negative. ES can preserve route in the reduced head space, but it does not beat the better gradient readers on content. |
| H | exploration | locked-confirm + finalquery-heavy + longdistance suite | Strong positive infrastructure result. This suite is what separated real reader gains from base-only overfit. |

## Cluster Notes

### A. Frozen-State Content Audit

- This cluster changed the whole phase.
- `1874`, `1842`, and `1201` are clear go signals for frozen-head recovery.
- `1879` is the no-go family for optimistic head-only content recovery.

### B. Fragile Direct-Entry Head-Only Recovery

- The fragile five-seed panel is now the cleanest route-faithful / content-poor
  boundary in the repo.
- It should be treated as a source-quality negative, not a head-optimization
  negative.

### C. Medium Teacher-Shaped Head-Only Recovery

- The medium family is stable enough to support repeated head-only work.
- The late plain-readout `9111` family became the cleanest confirmed medium
  branch.
- Its base content is materially better than the older medium distill line, but
  the confirm suite still says the family is generalization-limited rather than
  route-limited.

### D. Content Transfer

- Content-only distillation is real and worth keeping in the map.
- It is not yet the final winner because the cleanest branch remains the
  alternative-reader family, not the distillation family.

### E. Minimal-Safe Partial Touch

- This cluster was justified only because Cluster A said `1879` lacked
  recoverable frozen content.
- The result is still negative: the safest read-path touches are not enough to
  solve fragile weak-content basins.

### F. Alternative Readers

- Query-FILM is the strongest exploration-side result of the phase.
- It is the clearest evidence that reader design, not just objective weighting,
  is a first-class lever on decodable frozen sources.

### G. Head-Only ES

- `9201` is the clean route-collapse negative.
- `9202` is the useful boundary: route can survive head-only ES, but content
  still lags the better gradient readers.

### H. Stress / Confirmation

- This cluster is one of the most valuable artifacts from phase 9.
- Without it, several strong-source branches would have looked much better than
  they really were.

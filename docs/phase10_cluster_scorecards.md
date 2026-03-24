# Phase 10 Cluster Scorecards

Phase 10 focused on read-path architecture and route-preserving adaptation on
already-decodable frozen sources. The detailed ledger is in
[phase10_run_matrix.csv](/home/catid/gnn2/docs/phase10_run_matrix.csv).

| Cluster | Half | Best Representative | Final Verdict |
| --- | --- | --- | --- |
| A | fruitful | `10012` multiview query-gated, `10015` sink-only, `10014` cross-attn | Strong mapping result. Multi-view readers greatly improve base fit, but held confirms still saturate near the same ceiling. |
| B | fruitful | `10024` query-FILM low-rank adapter | Fair negative overall. Most adapters do not beat strict frozen heads, and the strongest `10024` outlier did not survive five-seed verification. |
| C | fruitful | `10042`, `10043`, `10064`, `10065` | Fair negative. Stronger final-query weighting and content distill can change base fit but do not solve held confirms on their own. |
| D | fruitful | `10152` on `1821`, `10142` on `1842` | Positive portability, limited ceiling shift. The best strong-source reader ideas transfer on base behavior, but held confirms remain near the old regime. |
| E | exploration | `10052` iterative cross-attn | Fair negative. Tiny iterative readers preserve route and still do not beat the held-confirm ceiling. |
| F | exploration gated | not entered seriously | Correctly gated off. Phase-10 readers and read-path adapters were already enough to show the ceiling more sharply, so broader touches were not justified. |
| G | exploration | `10521-10525` head-level ES suite | Fair negative. ES does not beat the stronger gradient readers once routing is frozen. |
| H | exploration | locked-confirm, finalquery-heavy, and longdistance confirmations | Strong infrastructure positive. This suite is what separated huge base-fit gains from real held-confirm progress. |

## Cluster Notes

### A. Multi-View Frozen-State Reader Architecture

- This cluster answered the view question cleanly.
- `final_sink_state` is the crucial content view.
- Extra views mainly buy base fit.

### B. Route-Preserving Read-Path Adapter

- Most tiny read-path adapters are not enough.
- The phase-10 decision point was the `10024` query-FILM low-rank adapter.
- Its exact rerun was clean, but its full panel fell back to the old
  held-confirm regime.
- That makes the adapter story mostly a fair negative.

### C. Objective / Data-Mix / Content Distillation

- Final-query weighting remains useful.
- Content-only distillation can help base fit.
- Neither changes the held-confirm limit in a decisive way.

### D. Portability Across Decodable Families

- This cluster matters because it shows the strong-source map is not entirely
  special-case.
- But it also shows that the strong-source held-confirm limit is not just an
  `1874` quirk.

### E. Iterative Readers

- A worthwhile test.
- A clean negative.

### F. Minimal-Safe Partial Unfreeze

- Phase 10 did not justify entering this cluster beyond planning-level gating.
- The strict read-path map was already sharp enough to say broader touches were
  not the next disciplined move.

### G. Head-Level ES

- A cleaner negative than earlier phases because the search space was already
  reduced to the read path.
- That makes the conclusion stronger: ES is not rescuing this regime.

### H. Stress / Confirmation

- This cluster is one of the most valuable phase-10 artifacts.
- Without it, several near-perfect base readers would have looked like much
  stronger scientific results than they really are.

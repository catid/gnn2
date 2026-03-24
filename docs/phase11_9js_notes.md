## gnn2-9js

Question: after `uup` and `2bj` established `13711` as the best delayed-only
`stop@48` source-distill boundary, can a route-aware composite checkpoint
selector keep that route gain while recovering base content or held-confirm
behavior?

Baseline references:

- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`
- `13711` base confirm: `0.6392 / 0.3705 / 0.8024 / 124.42`
- `13711` full_locked: `0.1306 / 0.2539 / 0.9876 / 126.43`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Composite-selector sweep:

- `13911` lexicographic, route floor `0.79`:
  `0.6348 / 0.3555 / 0.7756 / 123.86`
- `13912` lexicographic, route floor `0.80`:
  `0.6382 / 0.3416 / 0.7944 / 124.14`
- `13913` weighted geomean, val fq_acc + val route + locked route:
  `0.6387 / 0.3605 / 0.7786 / 123.98`
- `13914` weighted sum, val fq_acc + locked fq_acc + locked route:
  `0.6548 / 0.3774 / 0.7567 / 124.35`

Round-1 interpretation:

- the hard route-floor selectors did what they were supposed to do on the
  selector, but they gave back too much base final-query accuracy
- `13911` and `13912` were therefore clear negatives

Round-2 interpretation:

- `13914` was also clearly worse than `13711`
- `13913` was the only borderline challenger because its proxy
  `full_locked fq_acc` ticked up slightly without collapsing route

Rerun:

- `13913_rerun1` matched exactly at selected step `335`
- rerun val: `0.6387 / 0.3605 / 0.7786 / 123.98`
- rerun proxy `full_locked`: `0.6450 / 0.3968 / 0.7927 / 124.28`

Confirm:

- `13913` base confirm: `0.6370 / 0.3646 / 0.7646 / 123.96`
- `13913` full_locked: `0.1306 / 0.2539 / 0.9876 / 126.43`
- `13913` finalquery_heavy: `0.1980 / 0.2465 / 0.9860 / 126.44`
- `13913` longdistance: `0.1846 / 0.2635 / 0.9843 / 157.96`

Conclusion:

- route-aware selector tuning is now a fair rerun-backed negative on this
  `13711` family
- hard route floors improved proxy-route discipline but cost too much base
  final-query accuracy
- the softer `13913` composite is stable, but confirm stayed identical to
  `13711` on held splits and worse on base confirm
- the next experiment should change the training signal again rather than spend
  more budget on selector variants

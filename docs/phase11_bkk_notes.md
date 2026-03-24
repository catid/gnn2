## gnn2-bkk

Question: after `gnn2-j9b` showed that `final_query_only` source distill in the
early `0..48` window is a fair local negative, can shifting that same signal
later recover content without disrupting the route regime learned by `13711`?

Baseline references:

- `13411` selected val: `0.6523 / 0.3833 / 0.7766 / 124.08`
- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Late-start `final_query_only` sweep:

- `14111` `lw=0.10`, `start=48`, `stop=96`: `0.6318 / 0.3406 / 0.7587 / 123.49`
- `14112` `lw=0.10`, `start=96`, `stop=144`: `0.6484 / 0.3684 / 0.7994 / 124.68`
- `14113` `lw=0.15`, `start=48`, `stop=96`: `0.6680 / 0.3972 / 0.7716 / 124.25`
- `14114` `lw=0.15`, `start=96`, `stop=144`: `0.6592 / 0.3863 / 0.7805 / 124.44`

Proxy `full_locked` view at the selected checkpoints:

- `14111`: `0.6304 / 0.3707 / 0.8021 / 124.38`
- `14112`: `0.6343 / 0.3800 / 0.7918 / 124.14`
- `14113`: `0.6353 / 0.3754 / 0.7862 / 124.31`
- `14114`: `0.6426 / 0.3856 / 0.7750 / 123.40`

Exact rerun:

- `14113_rerun1` matched `14113` bit-for-bit:
  `0.6680 / 0.3972 / 0.7716 / 124.25`

Interpretation:

- `14111` was simply too weak and too disruptive: it lost both content and
  route relative to `13711`
- `14112` preserved route closest to `13711`, but its final-query content
  stayed too low to count as a better boundary
- `14113` was the only real challenger on overall fit and base final-query
  accuracy, but the exact rerun confirmed that this gain came with a real route
  drop and no compensating locked-proxy improvement
- `14114` moved the teacher window later, but it still failed to beat `13711`
  on the main content-route tradeoff

Conclusion:

- shifting `final_query_only` source distill later does not rescue the line
- both the early-window (`j9b`) and late-window (`bkk`) versions are fair local
  negatives against the `13711` delayed-only `stop@48` boundary
- on this frozen-`1201` branch, the useful teacher signal still looks
  `delayed_only`, not `final_query_only`

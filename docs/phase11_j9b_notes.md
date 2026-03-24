## gnn2-j9b

Question: after selector tuning failed on the `13711` delayed-only `stop@48`
boundary, can making the same `13111` source-logit transfer apply only on
final-query-needed examples recover content without giving back route?

Baseline references:

- `13411` selected val: `0.6523 / 0.3833 / 0.7766 / 124.08`
- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Final-query-only `stop@48` sweep:

- `14011` `lw=0.075`: `0.6484 / 0.3605 / 0.7885 / 125.22`
- `14012` `lw=0.10`: `0.6636 / 0.3744 / 0.7825 / 124.13`
- `14013` `lw=0.125`: `0.6509 / 0.3744 / 0.7915 / 124.22`
- `14014` `lw=0.15`: `0.6548 / 0.3704 / 0.7895 / 124.03`

Proxy `full_locked` view at the selected checkpoints:

- `14011`: `0.6411 / 0.3800 / 0.7862 / 124.92`
- `14012`: `0.6509 / 0.3847 / 0.7983 / 124.38`
- `14013`: `0.6353 / 0.3847 / 0.8049 / 124.10`
- `14014`: `0.6523 / 0.3922 / 0.7638 / 124.31`

Interpretation:

- `14011` was simply too weak: it lost both content and route relative to
  `13711`
- `14012` improved overall fit, but it clearly gave back too much base
  final-query accuracy and route to count as a better checkpoint than `13711`
- `14013` preserved route a bit better than `14012`, but content still stayed
  below both `13411` and `13711`
- `14014` looked tempting on proxy `full_locked fq_acc`, but that happened only
  by giving back too much route, so it is not a real challenger either

Conclusion:

- switching the `13711` source-distill signal from `delayed_only` to
  `final_query_only` from step `0` is a fair local negative
- none of the four variants beat the `13711` content-route tradeoff on the main
  selected split
- the sharper next question is timing, not scope alone:
  `final_query_only` may still be useful if it starts after the early
  route-shaping window instead of interfering from the beginning

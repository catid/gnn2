## gnn2-0iq

Question: after `e42` showed that neither isolated half nor overlap windows could
match the `14313` delayed-only optimum, is the useful effect actually a
midpoint-only phenomenon? If so, a short central pulse should retain the useful
content-route tradeoff without needing the full contiguous `24..72` span.

Baseline references:

- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`
- `14313` selected val: `0.6626 / 0.4002 / 0.7994 / 124.49`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Central-pulse sweep, same source family and selector:

- `14711` central narrow `36..60`, `lw=0.1375`: `0.6509 / 0.3734 / 0.7875 / 124.20`
- `14712` central broad `30..66`, `lw=0.1375`: `0.6479 / 0.3674 / 0.7825 / 123.27`

Interpretation:

- the narrow midpoint pulse `36..60` underperformed `14313` on both content and
  route while also falling below `13711` on final-query accuracy
- the broader midpoint bridge `30..66` was even worse and gave back additional
  route without recovering content
- because both same-weight midpoint variants were clearly below the existing
  boundaries, there was no reason to spend more GPU time on the stronger
  mass-compensated second round

Conclusion:

- midpoint coverage alone is not enough on this frozen-`1201` delayed-only
  branch
- the `14313` effect really does depend on the full contiguous `24..72` window,
  not just the center of that window
- `gnn2-0iq` closes as a fair local negative, and the timing-family map is now
  sharp enough that this subline can be retired

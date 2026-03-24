## gnn2-2bj

Question: after `uup` showed that `lw=0.10` with delayed-only source
distillation and `stop@48` is a real rerun-backed schedule effect, is there a
better local point below `0.10` that keeps the route gain without overshooting
confirmed base content?

Baseline references:

- `13411` selected val: `0.6523 / 0.3833 / 0.7766 / 124.08`
- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Lighter `stop@48` sweep:

- `13811` `lw=0.05`: `0.6489 / 0.3754 / 0.7587 / 124.00`
- `13812` `lw=0.0625`: `0.6528 / 0.3694 / 0.7905 / 124.13`
- `13813` `lw=0.075`: `0.6562 / 0.3843 / 0.7815 / 124.14`
- `13814` `lw=0.0875`: `0.6489 / 0.3644 / 0.7954 / 124.22`

Interpretation:

- every lighter weight gave back final-query accuracy relative to `13711`
- `0.05` also lost route materially and is clearly worse than both `13411` and
  `13711`
- `0.0625` and `0.0875` kept some of the route gain, but both were too weak on
  content to justify promotion
- `0.075` was the closest challenger on overall behavior, but it still stayed
  below `13711` on both final-query accuracy and route

Conclusion:

- the useful local optimum in this delayed-only `stop@48` family is still at
  `lw=0.10`
- the whole lighter-weight band `0.05 .. 0.0875` is a fair local negative
- `2bj` closes without rerun or confirm because no challenger beat the existing
  `13711` boundary on the selector tradeoff

# Phase 11 `gnn2-8te` Notes

## Scope

Starting from `by8`, the probe-initialized two-view frozen-`1201` reader
(`11914`) recovered a real final-query regime and reran exactly, but at a very
bad overall tradeoff. This follow-up keeps the same frozen core and the same
probe-initialized decisive-view reader, and changes only the objective balance
and checkpoint-selection rule.

## Variants

- `12011`: selector-only change to `selection_eval.full_locked.accuracy`
- `12012`: weight-only rebalance to `final_query_weight=2.0`,
  `non_final_query_weight=1.0`
- `12013`: combined selector + weight rebalance
- `12014`: combined selector + weight rebalance with light
  `full_locked + finalqueryheavy + longdistance` auxiliary mixing
- `best_rerun`: exact rerun of the strongest scout, if competitive

## Result

The family is a rerun-backed negative.

Selected test metrics:

- `11732` baseline: `0.5238 / 0.2620 / 0.6197 / 120.08`
- `11914` probe-init boundary: `0.2598 / 0.3015 / 0.6317 / 107.91`
- `12011` selector-only: `0.5264 / 0.3189 / 0.3964 / 122.12`
- `12011_rerun1`: exact match to `12011`
- `12012` weight-only: `0.6217 / 0.8997 / 0.0074 / 126.60`
- `12013` selector + weight: `0.5293 / 0.2627 / 0.3730 / 117.98`
- `12014` selector + weight + confirm-mix:
  `0.5505 / 0.3155 / 0.0000 / 25.73`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Interpretation

- selector-only rebalance `12011` is the least-bad move:
  it restores overall behavior and even improves final-query accuracy, but it
  gives back too much route fidelity to count as a real fix
- weight-only rebalance `12012` is a clear wrong-basin failure:
  it drives late exits and high answer accuracy while final-query route match
  collapses to essentially zero
- combined selector + weight rebalance `12013` does not rescue that tradeoff:
  it stays close to `12011` overall while route remains far below `11732`
- light confirm-mix `12014` is actively harmful here:
  it destroys route and exits much earlier than either `11732` or `11914`

## Conclusion

`gnn2-8te` is closed as a rerun-backed negative.

Objective rebalance alone is not enough on the probe-initialized frozen-`1201`
reader:

- selecting for overall behavior can recover non-final-query fit
- but it does so by giving back the recovered route structure
- stronger weight changes or light confirm-mix do not preserve both

So the next issue is no longer "try another simple weighting tweak". The sharper
follow-up is a route-aware composite selector or objective that can explicitly
keep the `11914` route regime while improving base behavior.

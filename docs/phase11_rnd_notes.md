# Phase 11 `gnn2-rnd` Notes

## Scope

Keep the strong `1201` core frozen, but stop porting the `1874` read-path
geometry. Instead, train a compatible zero-init or source-native head on top of
the frozen `1201` core features, using the same `full_locked`
final-query-accuracy selector discipline as `qiz`.

## Variants

- `11731`: keep the `1201`-initialized `sink_proj`, train only a fresh
  readout-only head
- `11732`: freeze the `1201` core and train a fresh `sink_proj + readout`
  pair

Both runs started from the phase-10 `1201` anchor via partial init, while
leaving routing, memory, control, wait, and release frozen.

## Summary

- `11731` best val: `overall 0.5093`, `fq_acc 0.2532`, `fq_route 0.2781`
- `11731` `full_locked` selector: `overall 0.4922`, `fq_acc 0.2493`,
  `fq_route 0.2932`
- `11731` test: `overall 0.5137`, `fq_acc 0.2540`, `fq_route 0.2955`

- `11732` best val: `overall 0.5249`, `fq_acc 0.2651`, `fq_route 0.6147`
- `11732` `full_locked` selector: `overall 0.4917`, `fq_acc 0.2549`,
  `fq_route 0.5976`
- `11732` test: `overall 0.5238`, `fq_acc 0.2620`, `fq_route 0.6197`

Metric order is `overall / fq_acc / fq_route`.

## Interpretation

This family did improve materially over the broken `qiz` ports:

- the runs no longer collapsed into `fq_route = 0.0`
- the `11732` variant recovered a real delayed-route regime
- the issue is therefore not "any frozen `1201` head instantly dies"

But the family still stayed far below any viable strong-source regime:

- final-query accuracy remained near chance
- even the better `11732` route stayed far below the strong frozen-source
  ceiling
- there was no sign of a promotion-worthy or rerun-worthy candidate

## Conclusion

The `1201` portability failure is not fully solved by simply dropping the
`1874` geometry and training a source-native linear read path. Strong `1201`
content is present and partially recoverable, but a very small frozen
`sink_proj + readout` head is still too weak to exploit it well.

That makes the next reasonable follow-up narrower:

- keep the `1201` core frozen
- use only the sink/readout-side views that the audit showed are decisively
  decodable
- add a source-native query-conditioned reader instead of another plain linear
  head

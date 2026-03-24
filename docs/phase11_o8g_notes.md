# Phase 11 `gnn2-o8g` Notes

## Scope

Start from the `1201` portability boundary established by `qiz`, `t2d`, and
`rnd`:

- direct `1874`-style frozen reader ports collapse immediately
- the exact frozen `1201` anchor is perfectly decodable at sink/readout views
- source-native linear heads improve over the broken ports, but remain weak

This follow-up asked whether a source-native query-conditioned reader could
finally exploit those decisive frozen `1201` views.

## Variants

- `11811`: native `sink_proj`, `multiview_query_gated`,
  `sink_state_query + baseline_readout_input`
- `11812`: native `sink_proj`, `multiview_query_film`,
  `sink_state_query + baseline_readout_input`
- `11813`: fresh `sink_proj`, `multiview_query_gated`,
  `sink_state_query + baseline_readout_input`
- `11814`: fresh `sink_proj`, `multiview_query_film`,
  `sink_state_query + baseline_readout_input`

All four variants kept routing, memory, control, wait, and release frozen and
used the same `full_locked` final-query-accuracy selector discipline as the
earlier `1201` portability slices.

## Best Observed Validation

- `11811`: `overall 0.4819`, `fq_acc 0.2840`, `fq_route 0.0606`
- `11812`: `overall 0.4507`, `fq_acc 0.2483`, `fq_route 0.0804`
- `11813`: `overall 0.4639`, `fq_acc 0.2572`, `fq_route 0.0000`
- `11814`: `overall 0.5010`, `fq_acc 0.2701`, `fq_route 0.0000`

Metric order is `overall / fq_acc / fq_route`.

For `11813`, the run was stopped once it remained route-dead through step 239.
Its best observed validation point had already plateaued near chance content and
zero final-query route, so there was no promotion or rerun case left.

## Interpretation

This family is a clean negative.

- native-sink query-conditioned readers did not rescue the frozen `1201`
  portability problem
- fresh-sink query-conditioned readers optimized smoothly, but they still
  converged to route-dead or near-route-dead final-query behavior
- none of the four variants beat even the weaker linear `11732` baseline

So the issue is not just "use a query-conditioned head instead of a linear
head". Even with the decisive views only, online frozen-reader training still
fails to realize the decodability that the offline audit sees.

## Conclusion

The next promising step is no longer another architecture swap on the same
online training recipe. The sharper follow-up is to test whether the offline
probe can be turned into a useful initialization or supervision target for a
frozen `1201` reader, since the decodable content is clearly present but current
online heads fail to reach it.

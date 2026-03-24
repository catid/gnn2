# Phase 11 `gnn2-by8` Notes

## Scope

Start from the `1201` boundary established by `t2d`, `rnd`, and `o8g`:

- the exact frozen `1201` anchor is perfectly decodable at the sink and readout
  views that matter
- direct online frozen-reader ports still fail badly
- the best source-native linear baseline `11732` recovers a useful final-query
  regime, but remains weak overall

This follow-up asked whether the missing ingredient was mostly
optimization-initialization rather than reader expressivity:

- keep the strong `1201` core frozen
- use only the decisive frozen views
- initialize the online MLP reader from the same query-conditioned probe family
  that already solves the offline frozen decoding problem

## Variants

- `11911`: `final_sink_state` only with a fresh MLP readout
- `11912`: `final_sink_state` only with probe warmstart
- `11913`: `final_sink_state + sink_state_query` with a fresh MLP readout
- `11914`: `final_sink_state + sink_state_query` with probe warmstart
- `11914_rerun1`: exact rerun of the best `11914` variant

All variants kept routing, memory, control, wait, and release frozen and used
the same `full_locked` final-query-accuracy selection rule as the earlier
`1201` follow-ups.

## Selected Test Results

- `11732` baseline: `overall 0.5238`, `fq_acc 0.2620`, `fq_route 0.6197`,
  `fq_exit 120.08`
- `11911`: `0.1895 / 0.2701 / 0.0000 / 13.60`
- `11912`: `0.4206 / 0.2868 / 0.0662 / 108.56`
- `11913`: `0.4551 / 0.2360 / 0.0000 / 4.87`
- `11914`: `0.2598 / 0.3015 / 0.6317 / 107.91`
- `11914_rerun1`: exact match to `11914`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The offline warmstart itself was nontrivial rather than degenerate:

- `11912` warmstart train accuracy: `0.3966`
- `11914` warmstart train accuracy: `0.4605`

## Interpretation

This family is a useful negative rather than a dead branch.

- probe warmstart clearly changed the frozen `1201` optimization regime
- single-view warmstart `11912` moved from near-immediate exit to a genuinely
  late-exit regime, but route fidelity stayed poor
- two-view warmstart `11914` finally matched and slightly exceeded `11732` on
  final-query accuracy and route match
- but `11914` achieved that by sacrificing overall behavior badly and exiting
  earlier than `11732`

So the remaining gap is not simply "the online head starts in the wrong part of
parameter space". Probe-style initialization helps a lot, but it still does not
produce a balanced frozen-reader solution on `1201`.

## Conclusion

`gnn2-by8` is closed as a rerun-backed negative for the claim that
probe-initialized decisive-view readers are enough by themselves.

The sharp next step is no longer another architecture swap on the same setup.
The remaining issue is objective balance:

- preserve the `11914` final-query recovery
- stop the severe collapse in non-final-query behavior
- do that without unfreezing the strong `1201` core

# Phase 12 Report

## Starting Point

Phase 12 started from the phase-10 and phase-11 map:

- on strong decodable `1874`, the best confirmed frozen-reader families could
  push base decoding very high while held-confirm content stayed near the old
  ceiling around `full_locked fq_acc ~= 0.31`,
- `final_sink_state` had already emerged as the decisive frozen content view,
- route-trace retention and basin entry were no longer the main unknowns on the
  good source families,
- on frozen `1201`, readout-side transfer was real on base behavior but still
  collapsed onto the same held-confirm plateau,
- and the next plausible levers were therefore trajectory-aware reading,
  factorized content/query reading, and only after those a minimal sink-side
  change.

The phase-12 question was:

- can a route-trace-conditioned temporal-bank reader or a factorized
  content/query reader materially improve held-confirm content on already
  decodable frozen sources without harming late-route fidelity?

## Campaign Coverage

The full ledger is
[phase12_run_matrix.csv](/home/catid/gnn2/docs/phase12_run_matrix.csv).

Phase 12 finished as a narrow architecture campaign with:

- `82` substantive rows,
- `72` fruitful and `10` exploration runs,
- `5` reproduced anchors,
- `13` locked-confirm evaluations,
- `11` same-seed rerun rows,
- `12` seed-panel rows,
- `5` source families represented: `1874`, `1821`, `1842`, `1201`, and the
  `1879` negative control.

The rerun ladder was scientifically useful rather than cosmetically clean:

- `6` reruns matched exactly,
- `3` more reproduced the selected checkpoint metrics exactly,
- `2` reruns were meaningful mismatches that materially changed the phase map.

This is a **strong mapping exit**, not a positive exit. No phase-12 family
cleared the requested positive bar of `full_locked fq_acc >= 0.40` with
`fq_route >= 0.88`, `fq_exit >= 115`, and a strong five-seed base gain.

## Headline Findings

### 1. Trajectory-aware readers improve base behavior, not held confirms

Cluster A asked the main question directly on frozen `1874`.

The strongest exact-rerun-clean trajectory-aware readers were:

- [15051 bilinear mixed-bank routehist](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_temporalbank_sinkreadout_bilinear_exit_routehist_seed15051_p1)
- [15054 latent-pool no-route](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_temporalbank_sinkreadout_latentpool_exit_noroute_seed15054_p1)

Their verified confirms are:

| Run | Base | Full-Locked | FinalQuery-Heavy | Longdistance |
| --- | --- | --- | --- | --- |
| `15051` | `0.9850 / 0.9699 / 0.9505 / 122.40` | `0.6494 / 0.3159 / 0.8771 / 115.49` | `0.4482 / 0.3122 / 0.8801 / 115.84` | `0.5112 / 0.3022 / 0.8843 / 145.39` |
| `15054` | `0.9528 / 0.9726 / 0.9492 / 122.38` | `0.6487 / 0.3144 / 0.8771 / 115.49` | `0.4470 / 0.3107 / 0.8801 / 115.84` | `0.5103 / 0.3008 / 0.8843 / 145.39` |

So the phase-12 answer on pure trajectory-bank readers is:

- temporal banks can fit base behavior extremely well,
- extra temporal capacity is real and reproducible,
- but held confirms still sit on the old strong-source ceiling.

### 2. Route-trace conditioning is not the missing ingredient

Phase 12 explicitly tested route-conditioned readers against no-route controls.

Representative controls:

- [15027 no-route cross-attn](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_temporalbank_sinkreadout_crossattn_exit_noroute_seed15027_p1)
- [15049 no-route query-FILM](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_temporalbank_sinkreadout_queryfilm_exit_noroute_seed15049_p1)
- [15052 no-route bilinear](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_temporalbank_sinkreadout_bilinear_exit_noroute_seed15052_p1)

The clean pattern is:

- route traces were usually neutral to mildly harmful,
- no-route controls usually preserved the same or better content fit,
- route-trace features never opened a new held-confirm regime.

That means the main phase-12 hypothesis narrows sharply:

- the remaining limit is not failure to condition on route trace summaries,
- it is somewhere deeper in reader stability or in the training basin itself.

### 3. Factorized content/query readers are the strongest base-side family, but still ceiling-limited

Cluster B separated content retrieval from query interpretation.

The strongest phase-12 `1874` family was:

- [15057 no-route temporal bilinear factorized reader](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_seed15057_p1)

Its locked confirm was:

- base `0.9980 / 0.9973 / 0.9378 / 120.98`
- full_locked `0.6462 / 0.3097 / 0.8771 / 115.49`
- finalquery_heavy `0.4465 / 0.3101 / 0.8801 / 115.84`
- longdistance `0.5093 / 0.2994 / 0.8843 / 145.39`

Its five-seed base panel mean was:

- `0.9982 / 0.9975 / 0.9457 / 121.86`

So factorization absolutely matters for base behavior. It produced the best
phase-12 `1874` family on raw base fit. But even this family did not lift the
locked-confirm ceiling.

Payload/query auxiliaries also failed to move the held frontier:

- [15038 factorized sink-query with payload/query auxiliaries](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_sink_query_gated_payloadqueryaux_seed15038_p1)
  confirmed at base `0.9880 / 0.9753 / 0.9392 / 121.25` and full_locked
  `0.6572 / 0.2961 / 0.8797 / 116.14`.

So the factorized answer is:

- separating content and query helped the easy/base regime,
- narrow auxiliary supervision helped fit,
- neither changed held-confirm content recovery.

### 4. Stress work showed the strongest 1874 boundary is not exact-stable

Cluster H ended up carrying much of the real scientific weight.

The crucial result was:

- [15057_rerun1](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_seed15057_rerun1)

That rerun did **not** preserve the original late-route solution. Instead it
fell into an earlier-exit shortcut regime:

- val `0.9990 / fq_route 0.9712 / fq_exit 68.02`
- test `0.9980 / fq_route 0.9691 / fq_exit 67.00`

This matters more than a cosmetic rerun miss:

- it explains why several later phase-12 branches could look extremely strong
  on base accuracy while no longer behaving like the original late-route
  reader,
- it shows the best `1874` boundary is not merely ceiling-limited, but also
  fragile to optimization drift toward an early-exit shortcut basin.

So phase 12 did not just say “trajectory readers fail.” It said:

- the best route-blind factorized temporal reader has real upside,
- but its good late-route solution is not yet training-stable enough.

### 5. Contiguous windows do not transfer as a general advantage

Cluster C started from the real phase-11 `1201` clue that a full contiguous
`24..72` span mattered for delayed-only source distillation.

But the phase-12 direct reader tests did **not** generalize that into a broad
contiguous-window win:

- [15041 contiguous 1201 slab](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1201_temporalwindow_contiguous_24_72_seed15041_p1)
  underperformed
- [15042 sparse 1201 slab](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1201_temporalwindow_sparse_24_72_seed15042_p1)
  on the same budget and parameter count

Later portability work sharpened the point further.

The completed five-seed portability families clustered around:

- `1821`: base overall `0.9646`, base route `0.8274`, base exit `80.43`
- `1842`: base overall `0.9462`, base route `0.8253`, base exit `80.54`

Those are not stable transfers of the strong `1874` late-route regime. They are
weaker earlier-exit regimes.

So the real conclusion from Cluster C is:

- contiguous slabs are not a general reader advantage,
- the old `1201` contiguous-window clue was specific to that earlier
  transfer/distillation story,
- and strong-source portability from `1874` to `1821`/`1842` is not robust.

### 6. Probe-guided adapters were a serious negative

Cluster E asked whether the good reader boundary simply needed a better
initialized tiny downstream adapter.

Representative runs:

- [15071 random low-rank](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_adapter_lowrank_r8_random_seed15071_p1)
- [15072 final-readout probe](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_adapter_lowrank_r8_probe_finalreadout_seed15072_p1)
- [15073 content-probe low-rank](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_adapter_lowrank_r8_probe_content_seed15073_p1)
- [15074 affine probe](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_adapter_affine_probe_finalreadout_seed15074_p1)

The best of them, `15073`, only reached:

- val `0.2949 / route 0.9688 / exit 67.80`
- test `0.2881 / route 0.9775 / exit 68.01`

with an exact rerun match on top.

So probe-guided downstream adapters are now a clean negative, not an
underexplored positive.

### 7. Minimal keyed sinks change behavior, but in the wrong direction

Cluster F was the only justified upstream touch.

The decisive runs were:

- [15081 keyed sink 2-slot full retrain](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix2_full_seed15081_p1)
- [15083 keyed sink 4-slot full retrain](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix4_full_seed15083_p1)
- [15083_rerun1](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_sinkmix4_full_seed15083_rerun1)

`15083` looked spectacular on dev score but was actually a stable shortcut:

- val `1.0000 / fq_route 0.9668 / fq_exit 67.54`
- confirm full_locked `0.2734 / 0.9040 / 95.00`
- finalquery_heavy `0.3083 / 0.9437 / 64.69`
- longdistance `0.2493 / 0.9199 / 108.66`

The exact rerun matched bit-for-bit.

So the keyed-sink lesson is very strong:

- minimal sink changes can absolutely alter the readout regime,
- but the first thing they find is a stable early-exit shortcut,
- not a better held-confirm content solution.

That means sink compression is **not yet** the first justified rescue path.

## Main Answer

Phase 12 gives a cleaner answer than phase 10 or phase 11.

On strong decodable frozen `1874`:

- trajectory-aware readers improve base behavior,
- factorized content/query readers improve base behavior even more,
- route-trace conditioning does not help,
- payload/query auxiliaries do not help on held confirms,
- and the strong-source held-confirm ceiling still does not move.

On secondary sources:

- the best `1874` family does not transfer robustly,
- `1821` and `1842` both collapse into a weaker earlier-exit regime under
  real panels.

On sink changes:

- keyed sinks are not inert,
- but their first stable effect is an off-regime shortcut rather than a rescue.

And the most important extra phase-12 lesson is about stability:

- the strongest `1874` family is not only ceiling-limited,
- it is also vulnerable to drifting into the same early-exit shortcut basin
  under rerun.

## Exit

Phase 12 exits as a **strong mapping result**.

The sharpened map is:

- route-trace conditioning is not the missing ingredient,
- temporal banks help base behavior but not held confirms,
- factorized content/query reading helps base behavior but not held confirms,
- contiguous windows are not a general portability win,
- keyed sinks are not yet the right upstream fix,
- and the remaining frontier is now as much about training stability against
  shortcut basins as about reader expressivity itself.

## Next Experiment

The single next experiment should be narrower than phase 12:

- start from the best route-blind factorized temporal reader family on `1874`,
- explicitly regularize against the early-exit shortcut basin,
- preserve the late-route solution rather than adding more route traces or more
  generic reader capacity,
- and avoid reopening routing, memory, or broader sink families.

The real next question is no longer “can the reader read more?” It is:

- can the good late-route reader solution be made optimization-stable without
  drifting into the easy early-exit shortcut?

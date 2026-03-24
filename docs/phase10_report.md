# Phase 10 Report

## Starting Point

Phase 10 started from the phase-9 boundary:

- frozen-state content on strong teacher-free source `1874`, secondary source
  `1842`, medium source `1821`, and ES upper-bound `1201` is decodable,
- fragile `1879` is route-faithful but content-poor,
- the best phase-9 strong-source frozen-head family still plateaued on held
  confirms near `full_locked fq_acc ~= 0.32`,
- the next plausible lever was therefore read-path architecture and strictly
  downstream read-path adaptation rather than more routing work.

The phase-10 question was:

- can a better multi-view reader or a tightly route-preserving read-path
  adapter materially improve held-confirm content on already-decodable frozen
  sources without harming late route fidelity?

## Campaign Coverage

The full ledger is
[phase10_run_matrix.csv](/home/catid/gnn2/docs/phase10_run_matrix.csv).

Phase 10 finished with a read-path-heavy budget split and full anchor
reproduction before broad claims. The final ledger contains `84` substantive
rows: `53` fruitful and `31` exploration, with `5` confirmed anchors,
`13` exact reruns, and `36` seed-panel rows. That cleanly hits the nominal
phase target rather than only the hard floor, and it completes the required
anchor block, reruns, panels, and locked confirmations.

This is a **strong mapping exit**, not a positive exit. No phase-10 family
cleared the positive-exit bar of `full_locked fq_acc >= 0.40` with
`fq_route >= 0.88`, `fq_exit >= 115`, and a strong five-seed base content gain.

## Headline Findings

### 1. The strong-source held-confirm ceiling is real

Phase 10 first asked whether phase-9 frozen-head limits were just artifacts of
one reader family. They were not.

The main strong-source baselines all reproduced cleanly:

- [9142 anchor](/home/catid/gnn2/results/phase10_anchor/hard_st_b_v2_teacher1874_anchor_querygated_finalqweight_seed9142_p1)
- [9132 anchor](/home/catid/gnn2/results/phase10_anchor/hard_st_b_v2_teacher1874_anchor_queryfilm_seed9132_p1)

Their follow-on five-seed families show the same qualitative result:

| Family | Base Overall | Base FQ Acc | Base FQ Route | Full-Locked FQ Acc |
| --- | ---: | ---: | ---: | ---: |
| multiview query-gated | 0.9891 | 0.9775 | 0.9445 | 0.2996 |
| one-shot cross-attn | 0.9809 | 0.9711 | 0.9499 | 0.2991 |
| query-FILM baseline | 0.8237 | 0.6392 | 0.9525 | 0.3060 |
| sink-only | 0.8931 | 0.7811 | 0.9398 | 0.2981 |

So phase 10 confirms that:

- the strong-source base fit can become almost perfect,
- late route and late exits remain stable,
- but strict frozen readers alone still plateau near the same held-confirm
  content ceiling.

### 2. `final_sink_state` is the decisive frozen view

The view map is much sharper now.

Representative strong-source runs:

- [10015 sink-only](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_multiview_sink_only_concat_finalqweight_longer_lowlr_seed10015_p1)
- [10016 packet-only](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_multiview_packet_only_concat_finalqweight_longer_lowlr_seed10016_p1)
- [10011 sink+packet](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_multiview_sinkpacket_concat_finalqweight_longer_lowlr_seed10011_p1)
- [10012 sink+packet+baseline query-gated](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_querygated_finalqweight_longer_lowlr_seed10012_p1)

The qualitative answer is stable across confirms and panels:

- `final_sink_state` alone already reaches the full held-confirm plateau,
- `packet_state_query` alone is content-weak,
- adding packet and baseline readout views improves base fit dramatically,
- but those extra views do not materially move the held-confirm ceiling.

That means the remaining strong-source limit is not “we forgot the right frozen
view.” It is downstream of that.

### 3. Iterative readers also saturate at the same ceiling

Cluster E tested whether content was present but required a small amount of
iterative inference to decode.

Representative runs:

- [10052 iter2](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_crossattn_iter2_finalqweight_longer_lowlr_seed10052_p1)
- [10053 iter4](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_multiview_sinkpacketbaseline_crossattn_iter4_finalqweight_longer_lowlr_seed10053_p1)
- [10552-10555 panel](/home/catid/gnn2/results/phase10_panel/teacher1874_crossattn_iter2_panel)

The result is a fair negative:

- iterative readers preserve route,
- iterative readers fit base extremely well,
- iterative readers still land at the same held-confirm regime as one-shot
  fusion.

So phase 10 can retire the “needs a tiny recurrent decoder” hypothesis as the
main missing ingredient.

### 4. Most read-path adapters were flat, but one low-rank FILM adapter finally moved the needle

Most route-preserving adapters on `1874` preserved route but failed to beat the
best frozen-head baselines on held confirms:

- [10021 affine](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_querygated_adapter_affine_finalqweight_longer_lowlr_seed10021_p1)
- [10023 residual](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_querygated_adapter_residual_finalqweight_longer_lowlr_seed10023_p1)
- [10032 low-rank query-gated rerun](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_querygated_adapter_lowrank_finalqweight_longer_lowlr_seed10032_rerun1)

The one exception is the query-FILM low-rank adapter:

- [10024 original](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_queryfilm_adapter_lowrank_longer_lowlr_seed10024_p1)
- [10024 rerun](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1874_refine_queryfilm_adapter_lowrank_longer_lowlr_seed10024_rerun1)

Its exact rerun is bit-for-bit clean through the saved best checkpoint. But the
five-seed panel says the single-seed held-confirm bump does not hold:

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.9067 | 0.8095 | 0.9461 | 121.72 |
| full_locked | 0.6568 | 0.2957 | 0.8840 | 116.21 |
| finalquery_heavy | 0.4512 | 0.3080 | 0.8788 | 115.64 |
| longdistance | 0.5312 | 0.3376 | 0.8855 | 145.51 |

Compared with the best phase-9 strong-source baseline
(`full_locked fq_acc = 0.3237`) and the phase-10 query-FILM anchor panel
(`full_locked fq_acc = 0.3060`), this is not a robust improvement.

So the adapter lesson is now clear:

- `10024` was a real and exactly reproducible single-seed outlier,
- but the family does not clear the verification ladder,
- and strict read-path-only adaptation still mostly saturates.

### 5. Portability is real on base behavior, but still confirmation-limited

Phase 10 also checked whether the strongest strong-source reader ideas transfer
to secondary decodable families.

Representative confirmed runs:

- [10142 on 1842](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1842_refine_multiview_sinkpacketbaseline_querygated_fq5_longer_lowlr_seed10142_p1)
- [10143 on 1842 with content distill](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1842_refine_multiview_sinkpacketbaseline_querygated_contentdistill_esanchor_finalqweight_longer_lowlr_seed10143_p1)
- [10152 on 1821](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1821_refine_multiview_sinkpacketbaseline_querygated_fq5_longer_lowlr_seed10152_p1)
- [10162 adapter on 1842](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1842_refine_multiview_sinkpacketbaseline_querygated_adapter_affine_fq5_longer_lowlr_seed10162_p1)
- [10163 adapter on 1821](/home/catid/gnn2/results/phase10_dev/hard_st_b_v2_teacher1821_refine_multiview_sinkpacketbaseline_querygated_adapter_residual_fq5_longer_lowlr_seed10163_p1)

Five-seed panel means for the strongest portability lines:

| Source | Base Overall | Base FQ Acc | Base FQ Route | Full-Locked FQ Acc |
| --- | ---: | ---: | ---: | ---: |
| `1821` multiview query-gated fq5 | 0.8920 | 0.8027 | 0.9258 | 0.2557 |
| `1842` multiview query-gated fq5 | 0.7642 | 0.5424 | 0.9131 | 0.2537 |

This is a real portability result for base behavior, but not a breakthrough on
held confirms. The secondary-source families remain close to their earlier
confirmation ceiling.

### 6. `1879` stays a clean negative control

The fragile family remained negative in phase 10 and was used only as a sanity
check source family, not a main tuning target.

Representative family:

- [1879 negative panel](/home/catid/gnn2/results/phase10_panel/teacher1879_queryreadout_negative_panel)

Its multi-seed result still says the same thing phase 9 already established:

- route can remain excellent,
- exits can remain late,
- content stays near chance.

That justifies the phase-10 decision not to spend meaningful budget on
fragile-source tuning.

### 7. Head-level ES is a fair negative

Cluster G asked a much narrower ES question than earlier phases:

- can black-box polish help once routing is already frozen and the remaining
  problem is content readout?

Representative runs:

- [10521](/home/catid/gnn2/results/phase10_dev/hybrid_es_b_v2_teacher1874_multiview_querygated_resume_from10012_seed10521_p1)
- [10522](/home/catid/gnn2/results/phase10_dev/hybrid_es_b_v2_teacher1874_multiview_queryfilm_resume_from10013_seed10522_p1)
- [10523](/home/catid/gnn2/results/phase10_dev/hybrid_es_b_v2_teacher1874_multiview_crossattn_resume_from10014_seed10523_p1)
- [10524](/home/catid/gnn2/results/phase10_dev/hybrid_es_b_v2_teacher1821_multiview_querygated_fq5_resume_from10152_seed10524_p1)
- [10525](/home/catid/gnn2/results/phase10_dev/hybrid_es_b_v2_teacher1874_adapter_affine_resume_from10021_seed10525_p1)

The answer is still no:

- ES can preserve route in some cases,
- ES often collapses content or route,
- ES never beat the stronger gradient-trained readers or adapters on
  held-confirm content.

So phase 10 retires ES as a primary read-path optimizer for this problem.

## Main Answer

Phase 10 gives a sharper answer than phase 9.

On strong decodable frozen sources:

- the crucial frozen view is `final_sink_state`,
- multi-view readers mainly improve base fit rather than held-confirm content,
- iterative readers do not solve the ceiling,
- most strict read-path adapters are flat,
- and even the strongest adapter outlier (`10024`) falls back to the same
  held-confirm regime once it faces a real five-seed panel.

On secondary decodable sources:

- the same reader ideas port well enough to improve base behavior,
- but held confirms remain close to the old ceiling,
- which points away from a purely source-specific failure.

On fragile sources:

- phase 10 reaffirms that route fidelity alone is not enough,
- and `1879` remains a source-quality negative, not a reader-optimization
  negative.

## Exit

Phase 10 exits as a **strong mapping result**.

The sharpened map is:

- the decisive frozen view is `final_sink_state`,
- multi-view and iterative readers do not materially beat the held-confirm
  ceiling,
- portability exists on base behavior but not on held confirms,
- head-level ES is not the missing optimizer,
- strict downstream read-path adaptation is mostly flat,
- and the best adapter outlier did not survive full panel verification.

## Next Experiment

The single next experiment should stop searching broadly over reader families
and instead test a narrow confirmation-aware objective on the strongest frozen
strong-source baseline.

The best target is still the `1874` family, but the next move should be:

- start from the strongest frozen-head baseline,
- train with an explicit confirmation-mixed or consistency-style objective,
- keep routing frozen and keep the read path simple,
- and avoid spending more budget on generic multi-view capacity, ES, or
  secondary fragile-source tuning.

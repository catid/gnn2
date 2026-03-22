# Phase 9 Report

## Starting Point

Phase 9 started from the phase-8 boundary:

- teacher-guided wait/release-only supervision can create a real teacher-free,
  route-faithful basin-entry effect,
- route retention is no longer the missing ingredient on those sources,
- reopening `memory_` on fragile teacher-free basins destroys the basin,
- head-only reopenings preserve late route but leave content weak.

The phase-9 question was therefore narrower:

- do frozen route-faithful states already contain enough answer content to
  justify strict head-only recovery,
- if yes, which reader/objective family works best,
- and only if no, what is the smallest safe touch beyond frozen heads.

## Campaign Coverage

The full ledger is [phase9_run_matrix.csv](/home/catid/gnn2/docs/phase9_run_matrix.csv).

Phase 9 finished with:

- `108` substantive entries
- `75 / 33` fruitful / exploration split
- `18` exact same-seed reruns
- `7` full five-seed panels
- `11` confirmation rows plus additional locked-confirm evaluations inside the
  headline families
- `5` anchor reproductions

This is a strong-mapping exit, not a positive exit. No head-only or minimal-safe
pipeline reached the positive-exit bar of confirmation-clean
`fq_acc >= 0.45` with `fq_route >= 0.80` and `fq_exit >= 115` across a full
five-seed family.

## Headline Findings

### 1. Frozen-state decodability is the key split

Cluster A answered the first phase-9 question directly.

Strong and medium route-faithful sources are highly decodable even with frozen
state:

- [1874 base audit](/home/catid/gnn2/results/phase9_dev/audit_teacher1874_base_test/audit_summary.json):
  `go_signal=true`
- [1874 full_locked audit](/home/catid/gnn2/results/phase9_dev/audit_teacher1874_full_locked/audit_summary.json):
  best probe accuracy `0.9925`
- [1842 base audit](/home/catid/gnn2/results/phase9_dev/audit_teacher1821_memoryreadout1842_base_test/audit_summary.json):
  `go_signal=true`
- [1842 full_locked audit](/home/catid/gnn2/results/phase9_dev/audit_teacher1821_memoryreadout1842_full_locked/audit_summary.json):
  best probe accuracy `0.9104`
- [1201 full_locked audit](/home/catid/gnn2/results/phase9_dev/audit_anchor1201_es_full_locked/audit_summary.json):
  best probe accuracy `1.0000`

Fragile direct-entry `1879` is qualitatively different:

- [1879 base audit](/home/catid/gnn2/results/phase9_dev/audit_teacher1879_base_test/audit_summary.json)
- [1879 full_locked audit](/home/catid/gnn2/results/phase9_dev/audit_teacher1879_full_locked/audit_summary.json)
- [1879 finalqueryheavy audit](/home/catid/gnn2/results/phase9_dev/audit_teacher1879_finalqueryheavy/audit_summary.json)

Its best frozen probes stay around `0.30` rather than the `0.91+` to `0.99+`
regime reached by `1842`, `1874`, and `1201`.

That is the main phase-9 split:

- `1874`, `1842`, and `1201` are reader/objective/generalization problems
- `1879` is a weak-content source even though route fidelity is strong

### 2. Fragile direct-entry head-only recovery is a fair negative

The fragile `1879` family is now a confirmation-clean five-seed negative under
strictly frozen memory/router/control.

Five-seed means:

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.2491 | 0.2500 | 0.8353 | 122.11 |
| full_locked | 0.2498 | 0.2396 | 0.9887 | 126.97 |
| finalquery_heavy | 0.2403 | 0.2460 | 0.9869 | 126.96 |
| longdistance | 0.2583 | 0.2614 | 0.9857 | 159.00 |

Representative exact rerun:
[9102_rerun1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1879_refine_queryreadout_finalqweight_longer_lowlr_seed9102_rerun1)

So the fragile family does exactly what phase 8 suggested:

- late route and late exits survive head-only shaping,
- content stays near chance,
- and the audit now explains why: the source itself is weakly decodable.

### 3. Head-only shaping is genuinely viable on decodable strong sources

The strongest aggregate confirmed strong-source family is now the
query-gated + final-query-weighted branch over
`9142 / 9146 / 9147 / 9148 / 9149`.

Five-seed means:

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.8488 | 0.7004 | 0.9416 | 121.49 |
| full_locked | 0.6534 | 0.3237 | 0.8771 | 115.49 |
| finalquery_heavy | 0.4529 | 0.3181 | 0.8801 | 115.84 |
| longdistance | 0.5137 | 0.3057 | 0.8843 | 145.39 |

Representative exact-rerun-clean member:
[9142_p1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1874_refine_querygated_finalqweight_longer_lowlr_seed9142_p1)
plus
[9142_rerun1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1874_refine_querygated_finalqweight_longer_lowlr_seed9142_rerun1)

This is a real frozen-head content result. It materially improves base content
without touching memory and keeps late route intact under stress. The remaining
ceiling is held-confirm final-query accuracy, not route.

### 4. Query-FILM is the strongest alternative-reader result

The best alternative-reader family is the query-FILM branch over
`9132 / 9331 / 9332 / 9333 / 9334`.

Five-seed means:

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.8231 | 0.6506 | 0.9454 | 121.86 |
| full_locked | 0.6532 | 0.3232 | 0.8771 | 115.49 |
| finalquery_heavy | 0.4535 | 0.3188 | 0.8801 | 115.84 |
| longdistance | 0.5153 | 0.3079 | 0.8843 | 145.39 |

Representative member:
[9132_p1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1874_refine_queryfilm_longer_lowlr_seed9132_p1)
with exact rerun
[9132_rerun1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1874_refine_queryfilm_longer_lowlr_seed9132_rerun1)

This matters because it separates head architecture from loss shaping:

- query-gated final-query weighting is the strongest aggregate family,
- query-FILM is the strongest alternative-reader family,
- the two are nearly tied on held-confirm content,
- so reader design is a first-class lever, not just a detail of the loss.

### 5. Content-only distillation is competitive, but not the final winner

The strong-source content-distill family over
`9150 / 9151 / 9152 / 9153 / 9154` is real:

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.8452 | 0.6934 | 0.9401 | 121.39 |
| full_locked | 0.6543 | 0.3255 | 0.8767 | 115.37 |
| finalquery_heavy | 0.4556 | 0.3214 | 0.8795 | 115.75 |
| longdistance | 0.5140 | 0.3061 | 0.8841 | 145.39 |

But the lead rerun
[9153_rerun1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1874_refine_querygated_contentdistill_finalqweight_longer_lowlr_seed9153_rerun1)
did not exactly match the original `9153` summary trajectory. So phase 9 should
treat content-only distillation as competitive rather than dominant.

### 6. Medium teacher-shaped sources support head-only recovery, but stay confirmation-limited

The late plain-readout `1821` family is now the best confirmed medium-source
head-only pipeline:

`9111 / 9351 / 9352 / 9353 / 9354`

Five-seed means:

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.6666 | 0.3498 | 0.9147 | 122.21 |
| full_locked | 0.5936 | 0.2494 | 0.8394 | 115.99 |
| finalquery_heavy | 0.3815 | 0.2400 | 0.8405 | 116.33 |
| longdistance | 0.4478 | 0.2356 | 0.8093 | 143.01 |

Representative member:
[9111_p1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1821_refine_readoutonly_longer_lowlr_seed9111_p1)
with exact rerun
[9111_rerun1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1821_refine_readoutonly_longer_lowlr_seed9111_rerun1)

This improves materially over the earlier medium content-distill family:

| Family | Base Overall | Base FQ Acc | Base FQ Route | Full-Locked FQ Acc |
| --- | ---: | ---: | ---: | ---: |
| medium readout-only | 0.6666 | 0.3498 | 0.9147 | 0.2494 |
| medium content-distill | 0.4539 | 0.3189 | 0.9215 | 0.2408 |

So medium sources are not route-limited. They are stable enough for head-only
recovery, but still plateau on held-confirm content.

### 7. Minimal-safe touches and head-only ES did not solve the remaining problem

Gated minimal-safe touch on fragile `1879` preserved route but did not unlock
content. Best representative:
[9301_p1](/home/catid/gnn2/results/phase9_dev/hard_st_b_v2_teacher1879_refine_queryreadout_meminputbias_finalqweight_shortlowlr_seed9301_p1)

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.2705 | 0.2654 | 0.8108 | 121.33 |
| full_locked | 0.2373 | 0.2366 | 0.9766 | 126.89 |
| finalquery_heavy | 0.2419 | 0.2463 | 0.9815 | 126.93 |
| longdistance | 0.2581 | 0.2599 | 0.9857 | 158.94 |

Head-only ES on strong source also failed to beat the better gradient readers.
Best representative:
[9202_rerun1](/home/catid/gnn2/results/phase9_dev/hybrid_es_b_v2_teacher1874_headonly_sinkqueryreadout_resume_from9122_seed9202_rerun1)

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.5889 | 0.2655 | 0.7935 | 108.27 |
| full_locked | 0.6657 | 0.3142 | 0.7092 | 100.32 |
| finalquery_heavy | 0.4518 | 0.3087 | 0.7011 | 99.48 |
| longdistance | 0.5107 | 0.3086 | 0.7056 | 125.23 |

So phase 9 can answer both follow-up questions:

- minimal-safe partial unfreezing was not needed on decodable sources,
- and the first gated minimal-safe touches were not enough to rescue weak-content
  fragile basins,
- head-only ES is not the missing optimizer.

## Final Answer

Phase 9 is a **strong mapping exit**.

The campaign now supports the following verified answer:

- frozen route-faithful basins can already contain strongly decodable answer
  information,
- strict head-only shaping is therefore fundamentally viable on the right
  source families,
- the strongest aggregate confirmed head-only branch is the strong-source
  query-gated + final-query-weighted family,
- the strongest alternative-reader branch is query-FILM and it is nearly tied
  on held-confirm performance,
- fragile `1879` is weak-content, not just badly read out,
- minimal-safe read-path touches did not change that,
- head-only ES preserved route but did not beat strong gradient readers.

What phase 9 did **not** achieve is robust confirmation-clean content recovery:

- the best strong-source families still stall around `fq_acc ~ 0.32` on locked
  confirms,
- the best medium-source family still stalls around `fq_acc ~ 0.25`,
- fragile `1879` remains near chance even with preserved route and late exits.

So the remaining bottleneck is now narrower than it was at the start of the
phase:

- not basin entry,
- not route retention,
- not whether frozen content exists on strong sources,
- but how to increase read-path capacity or read-path adaptation on decodable
  sources without letting route drift.

## Single Next Experiment

The next experiment should be a single route-regularized read-path adapter on
the strongly decodable `1874` source, starting from the best confirmed frozen
reader baseline rather than from a fragile weak-content source.

Concretely:

- start from the strong-source query-gated final-query-weighted family,
- add one tightly constrained read-path adapter only,
- regularize explicitly against route drift under the locked-confirm suite,
- do not spend more budget on `1879`, new basin-entry tricks, or broad memory
  reopening until that adapter test is resolved.

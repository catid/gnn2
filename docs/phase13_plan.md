# Phase 13 Plan

## Starting Point

Phase 13 starts from the phase-12 strong-mapping exit at commit `4b5c89c`.
The current question is no longer whether route traces, more temporal context,
or a first keyed sink can help. Phase 12 settled that the best remaining
frontier is the stability of the late-route `1874` reader basin.

Current anchor metrics:

| Anchor | Role | Base | Full-Locked |
| --- | --- | --- | --- |
| `15051` | stable trajectory-aware control on `1874` | `0.9850 / 0.9699 / 0.9505 / 122.40` | `0.6494 / 0.3159 / 0.8771 / 115.49` |
| `15057` | high-upside unstable factorized temporal reader on `1874` | `0.9980 / 0.9973 / 0.9378 / 120.98` | `0.6462 / 0.3097 / 0.8771 / 115.49` |
| `15057_rerun1` | shortcut-collapse stress anchor on `1874` | phase-12 stress result: early-exit shortcut drift near exit `67` | not a promoted stable point |
| `15060` | weaker `1821` portability baseline | `0.9587 / 0.9864 / 0.9157 / 122.01` | `0.5952 / 0.2582 / 0.8237 / 115.80` |
| `15016` | `1879` negative control anchor | fragile negative-control head-only family | content-poor sanity family only |

Metric order throughout:

- `overall / fq_acc / fq_route / fq_exit`

## Main Hypothesis

The best `15057`-style route-blind factorized temporal reader already has
enough representational capacity. The remaining blocker is that training can
drift into an easier early-exit shortcut basin. The next gains should come from
stabilization and anti-shortcut training, not from more generic reader
capacity, more route traces, or broader sink work.

Alternative hypothesis:

- even if the late-route basin is stabilized, held-confirm content stays near
  the old `~0.31-0.32` ceiling, which would mean the frontier is no longer
  primarily reader-basin stability.

## Anchor Reproduction

Reproduce these from scratch before broad new claims:

1. `15051` stable `1874` control
2. `15057` high-upside unstable `1874` baseline
3. `15057_rerun1` same-seed rerun that collapses into the shortcut basin
4. `15060` `1821` secondary-source baseline
5. `15016` `1879` negative control

Exact commands:

```bash
./scripts/run_phase13_cluster_scouts.sh results/phase13_anchor anchor
./scripts/run_phase13_confirm.sh results/phase13_anchor/hard_st_b_v2_teacher1874_anchor_temporalbank_bilinear_exit_routehist_seed16011_p1
./scripts/run_phase13_confirm.sh results/phase13_anchor/hard_st_b_v2_teacher1874_anchor_factorized_temporalbank_query_bilinear_noroute_seed16012_p1
./scripts/run_phase13_confirm.sh results/phase13_anchor/hard_st_b_v2_teacher1821_anchor_factorized_temporalbank_query_bilinear_noroute_seed16013_p1
./scripts/run_phase13_confirm.sh results/phase13_anchor/hard_st_b_v2_teacher1879_anchor_queryreadout_seed16014_p1
```

## Budget

Phase-13 substantive run target:

- target: `64+`
- hard floor: `48+`

Budget split:

- fruitful main work: `~85%`
- bounded exploration / sanity: `~15%`

Planned substantive allocation:

| Cluster | Budget | Notes |
| --- | --- | --- |
| A | 24 runs | main anti-shortcut stabilization cluster on `15057` family |
| B | 14 runs | stable-to-upside continuation from `15051` into `15057`-style reader |
| C | 14 runs | hard-case weighting, held-confirm-aware selection, rollback logic |
| D | 6 runs | only after a stable recipe exists |
| E | 6 runs | secondary-source sanity on `1821` or `1842`, plus `1879` control |
| F | 0-3 runs | gated tiny ES or upstream sanity only if justified |

This yields `64-67` planned substantive runs without padding.

## Clusters

### Cluster A

Anti-shortcut stabilization on `15057`-style route-blind factorized temporal
bilinear readers on `1874`.

Required mechanisms:

- held-confirm-aware checkpoint selection
- explicit no-regularizer control
- hard-case auxiliary benchmarks (`full_locked`, `finalquery_heavy`,
  `longdistance`)
- teacher/self-distillation from late-route checkpoints
- conservative resume / low-LR continuation
- honest anti-shortcut routing losses only where they are already implemented

Promotion rule:

- must beat or match `15057` on base without visible shortcut drift
- must not clearly underperform `15051` on locked confirms
- exact same-seed rerun required before promotion

### Cluster B

Stable-to-upside continuation bridge from `15051` into the stronger
factorized/temporal family.

Required mechanisms:

- partial-init or blended init from `15051` and `15057`
- staged continuation via resume / low LR
- distillation from `15051` or `15057`
- route-preserving and locked-aware checkpoint selection

### Cluster C

Hard-case data mining and basin-aware selection.

Required mechanisms:

- `selection_eval_benchmarks` for `full_locked`, `finalquery_heavy`,
  `longdistance`
- lexicographic or composite selection over held-confirm content, route, and
  exit
- disagreement audits for `15051`, `15057`, `15057_rerun1`
- auxiliary hard-case mixes only as part of the stability protocol

### Cluster D

Post-stability refinement only if a stable `15057`-like basin exists.

Allowed scope:

- small reader refinements on top of stabilized recipe
- content-only distillation on stabilized recipe

### Cluster E

Secondary-source sanity and portability.

Apply only the best 1-2 stability recipes to:

- `1821` or `1842`
- `1879` negative control

### Cluster F

Optional, gated, and narrow:

- tiny head-only ES polish on a stabilized reader
- or one tiny upstream sanity touch if evidence strictly justifies it

## Promotion And Retirement Rules

Promote a candidate only if all are true:

1. It is competitive against `15051` or `15057` under comparable budget.
2. It does not obviously drift toward the shortcut basin in the tracked exit
   metrics.
3. Its exact same-seed rerun matches the intended regime.
4. Its locked confirm is not clearly weaker than the stable control.

Retire a family when:

1. at least 4 concrete variants have been tried,
2. at least 1 tuned top variant exists,
3. at least 1 exact rerun exists,
4. failure mode is diagnosed and recorded in the ledger.

## Verification Protocol

Headline results are only confirmed after:

1. exact same-seed rerun from scratch
2. five-seed panel
3. locked confirmation evaluation
4. fair comparison against the strongest relevant baseline
5. independent recomputation with `src/utils/phase13_verify.py`
6. ledger update with confirmation status

## Stop Criteria

Positive exit:

- stable `1874` recipe clears:
  - full_locked `fq_acc >= 0.35`
  - full_locked `fq_route >= 0.88`
  - full_locked `fq_exit >= 115`
  - base `fq_acc >= 0.97`
- and reruns do not collapse into the early-exit shortcut basin

Strong-mapping exit:

- clear verified answer on whether the blocker is shortcut-basin stability or a
  deeper held-confirm ceiling,
- stabilized-vs-unstable bridge mapped fairly,
- hard-case mining and held-confirm-aware selection judged fairly,
- narrow next experiment identified.

## Phase-13 Command Deck

Initial anchor block:

```bash
./scripts/run_phase13_cluster_scouts.sh results/phase13_anchor anchor
```

Initial stabilization rounds:

```bash
./scripts/run_phase13_cluster_scouts.sh results/phase13_dev a_r1
./scripts/run_phase13_cluster_scouts.sh results/phase13_dev b_r1
./scripts/run_phase13_cluster_scouts.sh results/phase13_dev c_r1
```

Verification / panels:

```bash
./scripts/run_phase13_confirm.sh <run-dir>
./scripts/run_phase13_seed_panels.sh <config> <results-root> <resume-ckpt> <seed1> <seed2> <seed3> <seed4> <seed5>
```


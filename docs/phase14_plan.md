# Phase 14 Plan

## Starting Point

Phase 14 starts from the pushed phase-13 closeout at `cb3a0bd`.

Established starting metrics and controls:

| Anchor | Role | Key locked / confirm reading |
| --- | --- | --- |
| `16045` | stable route/exit control | best fully paneled stable bridge; selected full-locked panel mean `0.9965 / 0.9460 / 121.86` for `fq_acc / fq_route / fq_exit` |
| `16081` | best content-side refinement | selected full-locked panel mean `0.9998 / 0.9410 / 121.44`; adds content but softens route/exit |
| `16066` | clean anti-shortcut family | selected full-locked panel mean `0.9557 / 0.9391 / 121.34` |
| `16092` | bounded `1821` sanity baseline | confirm `base 0.9563 / 0.9831 / 0.9201 / 122.64`, `full_locked 0.5935 / 0.2549 / 0.8237 / 115.80` |
| `16093` | `1879` negative control | remains in the expected bad-source regime |

Phase-13 closeout conclusion:

- stabilization worked,
- catastrophic shortcut collapse is no longer the main blocker on `1874`,
- held-confirm content still confirms back near `full_locked 0.648 / 0.313 / 0.877 / 115.49`,
- the next justified move is narrow content-focused supervision on top of the fixed `16045` recipe.

## Main Question

Can content-branch-only supervision on top of the fixed `16045` backbone improve held-confirm content recovery on `1874` while preserving the stabilized late-route regime?

Secondary questions:

- does a dual-anchor route/content contract help more than content-only supervision from labels alone?
- if branch-only supervision still saturates, what is the smallest safe content-path modularization that helps without touching routing?

## Anchor Reproductions

These must run before broad phase-14 claims:

1. `16045` stable control
2. `16081` best refinement
3. `16066` clean anti-shortcut family
4. one `1821` portability baseline
5. one `1879` negative-control baseline

## Content-Failure Hard Slice

Phase 14 uses an explicit hard slice aimed at the remaining failure: late-route but wrong-content.

Evaluation sources:

- `configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_full_locked.yaml`
- `configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_finalqueryheavy.yaml`

Per-sample late-route definitions:

- strict late-route: exact `route_match == 1` on delay-to-final-query samples
- relaxed late-route: delay-to-final-query sample with `exit_time >= query_time - 8`

The promotion slice is the union of:

1. `anchor_gap_late`: `16045` and `16081` are both relaxed-late, but final answers differ
2. `late_wrong_content`: candidate is relaxed-late and wrong on delay-to-final-query
3. `late_wrong_content_strict`: candidate is strict-late and wrong on delay-to-final-query
4. `late_hard_distance`: relaxed-late delay-to-final-query samples in the top retrieval-distance quartile

The hard-slice audit is generated with `scripts/run_phase14_hardslice.sh` and stored in candidate packs.

## Budget

Target run budget: `56-64` substantive runs.

Approximate split:

- mainline `75%`: Clusters A-C, about 48 runs
- gated architecture `15%`: Cluster D, about 8 runs only if A-C plateau
- characterization `10%`: Clusters E-F, about 8 runs

## Clusters

### Cluster A: Content-Branch-Only Auxiliary / Distillation On Fixed `16045`

- fixed route/exit control is `16045`
- train only factorized content branch, content/query combiner, and downstream readout
- no route/control teacher targets
- concrete families:
  - payload/content latent auxiliary on `factorized_content_hidden`
  - content-hidden distillation from `16081`
  - content-logit distillation from `16081` on final-query / delayed scopes
  - contrastive content-branch targets if the saved metadata supports them
  - no-aux control resumed from `16045`

### Cluster B: Dual-Anchor Route / Content Contract

- route/exit anchor stays with `16045`
- content teacher is `16081` unless another source is explicitly justified
- parameter anchoring is allowed only on non-content branches
- concrete families:
  - non-content parameter anchor to `16045` + content-hidden distillation from `16081`
  - non-content parameter anchor to `16045` + content-logit distillation from `16081`
  - hard-slice-only teacher application
  - route/content split contracts with generator-known content auxiliary

### Cluster C: Content-Failure Hard-Slice Mining And Selection

- mine disagreements between `16045` and `16081` on relaxed-late examples
- use content-failure-aware selection and rollback, not broad objective fishing
- concrete families:
  - hardslice-weighted task sampling
  - hardslice-aware lexicographic selection
  - rollback triggers keyed to held-confirm content collapse while route/exit stay late
  - finalquery-heavy / longdistance only where the mined slice justifies it

### Cluster D: Gated Content-Only Sidecar Path

- only after A-C show a verified plateau
- content-only sidecar or dual-channel sink disconnected from route decisions
- no router/control changes

### Cluster E: Compute-Quality Frontier

- compare `16045` against the best promoted phase-14 recipe
- produce held-confirm quality versus exit / compute tables

### Cluster F: Secondary-Source Sanity And Negative Control

- apply the top 1-2 promoted recipes to one of `1821` / `1842`
- keep `1879` as a bounded negative control only

## Promotion Gate

A candidate may only be promoted if all of the following hold:

1. exact same-seed rerun preserves the late-route regime
2. content-failure hard-slice metrics improve or stay clearly better than `16045`
3. locked confirms do not show route or exit collapse
4. route and exit stay in-range for the intended late-route regime
5. `src/utils/phase14_verify.py` passes
6. its candidate pack is complete and marked `pass`

## Verification Protocol

Headline results require:

1. exact same-seed rerun
2. five-seed panel
3. locked confirmation evaluation
4. comparison against the strongest relevant baseline
5. independent metric recomputation with `phase14_verify.py`
6. candidate-pack gate pass

Serious negatives require:

1. at least four concrete variants in-cluster
2. at least one tuned top variant
3. at least one rerun
4. diagnostics showing the failure mode
5. retirement note in the ledger

## Stop Criteria

Positive exit:

- a promoted `1874` recipe clears, across panels and locked confirms,
  `full_locked overall >= 0.67`, `full_locked fq_acc >= 0.36`,
  `full_locked fq_route >= 0.87`, `full_locked fq_exit >= 115`,
  while staying within a small tolerance of the `16045` route/exit regime.

Otherwise strong mapping exit:

- the campaign cleanly settles whether content-only supervision, dual-anchor contracts,
  or hard-slice mining can move the ceiling,
- whether a gated sidecar path is necessary,
- and whether any winning recipe is even weakly portable.

## Exact Commands

Anchor examples:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run --config configs/phase14/dev/hard_st_benchmark_b_v2_teacher1874_anchor16045_selectlocked.yaml
CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run --config configs/phase14/dev/hard_st_benchmark_b_v2_teacher1874_anchor16081_selectlocked.yaml
```

Hard-slice audit:

```bash
scripts/run_phase14_hardslice.sh \
  results/phase13_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend85_15051readout_15057extras_selectlocked_seed16045_p1 \
  results/phase13_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend85_resume16045_teacher15057_delayedkl010_selectlocked_lowlr_seed16081_p1
```

Verification:

```bash
scripts/run_phase14_confirm.sh results/phase14_dev/<run_dir>
```

Panels:

```bash
scripts/run_phase14_seed_panels.sh configs/phase14/dev/<config>.yaml 17157 17158 17159 17160 17161
```

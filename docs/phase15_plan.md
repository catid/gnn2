# Phase 15 Plan

## Starting Point

Phase 15 starts from the pushed phase-14 closeout at `1ae1a3b`.

Established starting controls and ceiling reads:

| Anchor | Role | Key reading |
| --- | --- | --- |
| `16045` | fixed stable route/exit control | selected full-locked five-seed `fq_acc / fq_route / fq_exit = 0.9965 / 0.9460 / 121.86` |
| `16081` | best content teacher candidate | selected full-locked five-seed `0.9998 / 0.9410 / 121.44`; adds content but softens route/exit slightly |
| `17024` | best content-branch-only phase-14 branch | five-seed `full_locked dqf 0.9953 / 0.9406 / 121.29`; locked confirm `0.2995 / 0.8797 / 116.14` |
| `17031` | best dual-anchor phase-14 branch | five-seed `full_locked dqf 0.9974 / 0.9432 / 121.55`; locked confirm `0.2874 / 0.8850 / 116.22` |
| `17101` | bounded `1821` sanity baseline | rerun-clean on summary slices; locked confirm `overall / dqf 0.6136 / 0.2533 / 0.8329 / 116.34` |
| `17103` | `1879` negative control | remains cleanly negative |

Phase-14 closeout conclusion:

- the stabilized `16045` late-route regime is no longer the main blocker,
- content-only supervision, dual-anchor contracts, hard-slice mining, and the first tiny content-only sidecar all preserved late-route summary behavior,
- but every serious branch still collapsed back to the same locked-confirm held-content ceiling,
- so the next justified move is a richer content-only path that still cannot affect routing.

## Main Question

Can a richer content-only path on top of the fixed `16045` route anchor improve held-confirm content recovery on `1874` without changing routing, control, or exit behavior?

Secondary questions:

- does a multi-slot content channel beat the current single-slot factorized content branch?
- does a readout-only sidecar memory beat the multi-slot channel?
- once the richer path exists, which content-only or dual-anchor training contract uses it best?

## Anchor Reproductions

These must run before broad phase-15 claims:

1. `16045` stable control
2. `16081` best refinement
3. `17024` best content-branch-only result
4. `17031` best dual-anchor result
5. one `1821` portability baseline
6. one `1879` negative-control baseline

## Content-Failure Hard Slice

Phase 15 keeps the phase-14 content-failure hard slice and extends it to richer content paths.

Evaluation sources:

- `configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_full_locked.yaml`
- `configs/phase8/confirm/hard_st_benchmark_b_v2_confirm_finalqueryheavy.yaml`

Per-sample late-route definitions:

- strict late-route: exact `route_match == 1` on delay-to-final-query samples
- relaxed late-route: delay-to-final-query sample with `exit_time >= query_time - 8`

Promotion slice is the union of:

1. `anchor_gap_late`: `16045` and `16081` are both relaxed-late, but final answers differ
2. `late_wrong_content`: candidate is relaxed-late and wrong on delay-to-final-query
3. `late_wrong_content_strict`: candidate is strict-late and wrong on delay-to-final-query
4. `late_hard_distance`: relaxed-late delay-to-final-query samples in the top retrieval-distance quartile

The slice is generated with `scripts/run_phase15_hardslice.sh` and stored in candidate packs.

## Route Isolation Contract

Every richer content-path branch must satisfy all of the following:

- routing modules remain frozen by config
- control modules remain frozen by config
- exit behavior remains frozen by default
- the richer content path is only computed after routing has completed
- route-related traces (`router_probs`, `exit_time`, sink accumulation) remain unchanged in dedicated unit tests when the richer path is zero-init and shared weights are copied from a baseline model

## Budget

Target run budget: `52-60` substantive runs.

Approximate split:

- mainline richer content path `70%`: Clusters A-B, about 36-40 runs
- training contracts `20%`: Clusters C-D, about 10-12 runs
- characterization `10%`: Clusters E-F, about 6-8 runs

## Clusters

### Cluster A: Multi-Slot Content Channel Under Fixed `16045`

- fixed route/exit control is `16045`
- routing/control remain frozen
- richer content path may influence final readout only
- concrete families:
  - 2-slot content channel
  - 4-slot content channel
  - shared-slot vs independent-slot content projections
  - zero-init vs learned-init slot paths
  - query-conditioned slot retrieval vs mean pooling

### Cluster B: Content-Only Sidecar Memory Under Frozen Routing

- content-only sidecar written from frozen features and read only by final readout
- no route/control dependence on sidecar state
- concrete families:
  - sidecar KV memory from final sink state
  - sidecar KV memory from factorized content branch
  - query-conditioned retrieval vs simple retrieval
  - tiny temporalized source via trajectory-bank source where justified

### Cluster C: Content-Specific Supervision On The New Path

- content-only supervision only
- no route/control teacher targets
- concrete families:
  - content-hidden distillation from `16081`
  - content-logit distillation from `16081`
  - generator-known payload / relation / answer-factor auxiliaries if available
  - contrastive retrieval loss on slots or sidecar memory
  - no-aux control

### Cluster D: Dual-Anchor Route / Content Contract On The New Path

- route anchor stays with `16045`
- content teacher is `16081` unless another source is explicitly justified
- parameter anchoring allowed only on non-content modules
- concrete families:
  - non-content parameter anchor to `16045` + content-hidden distillation
  - non-content parameter anchor to `16045` + content-logit distillation
  - generator-known content auxiliary on the richer path
  - no-content-teacher control

### Cluster E: Compute-Quality Frontier

- compare `16045` against the best promoted phase-15 recipe
- produce held-confirm quality versus exit / compute tables

### Cluster F: Secondary-Source Sanity And Negative Control

- apply the top 1-2 promoted recipes to one of `1821` / `1842`
- keep `1879` as bounded negative control only

## Promotion Gate

A candidate may only be promoted if all of the following hold:

1. exact same-seed rerun preserves the late-route regime
2. content-failure hard-slice metrics improve or hold relative to `16045`
3. locked confirms do not show route or exit collapse
4. route and exit stay in-range for the intended stable regime
5. `src/utils/phase15_verify.py` passes
6. its candidate pack is complete and marked `pass`

## Verification Protocol

Headline results require:

1. exact same-seed rerun
2. five-seed panel
3. locked confirmation evaluation
4. comparison against the strongest relevant baseline
5. independent metric recomputation with `phase15_verify.py`
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

- the campaign settles whether multi-slot channels help,
- whether readout-only sidecar memory helps,
- whether content-only or dual-anchor supervision can exploit the richer path,
- whether any recipe is weakly portable,
- and whether the remaining ceiling looks like read-path depth or content writing / retrieval failure.

## Exact Commands

Anchor examples:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.run --config configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_anchor16045_selectlocked.yaml
CUDA_VISIBLE_DEVICES=1 uv run python -m src.train.run --config configs/phase15/dev/hard_st_benchmark_b_v2_teacher1874_anchor17024_selectlocked.yaml
```

Hard-slice audit:

```bash
scripts/run_phase15_hardslice.sh \
  results/phase13_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend85_15051readout_15057extras_selectlocked_seed16045_p1 \
  results/phase13_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend85_resume16045_teacher15057_delayedkl010_selectlocked_lowlr_seed16081_p1
```

Verification:

```bash
scripts/run_phase15_confirm.sh results/phase15_dev/<run_dir>
```

Panels:

```bash
scripts/run_phase15_seed_panels.sh configs/phase15/dev/<config>.yaml 18051 18052 18053 18054 18055
```

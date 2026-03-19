# Phase 6 Plan

## Starting Point

Repo baseline at phase-6 start:

- commit: `4814c7a`
- primary reference docs:
  - [phase5_report.md](/home/catid/gnn2/docs/phase5_report.md)
  - [phase5_lessons.md](/home/catid/gnn2/docs/phase5_lessons.md)
  - [phase5_run_matrix.csv](/home/catid/gnn2/docs/phase5_run_matrix.csv)
- active phase-6 issue: `gnn2-4xu`

Phase-5 anchor facts to reproduce first:

| Anchor | Run | Split | Overall Acc | Final-Query Acc | Final-Query Route Match | Final-Query Exit Time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Poor hard-ST | `hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1` | confirm | 0.5442 | 0.2436 | 0.0034 | 41.48 |
| Medium-basin ES rescue | `hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_p1` | confirm | 1.0000 | 1.0000 | 1.0000 | 127.00 |
| Weak-basin ES partial rescue | `hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_p1` | confirm | 0.9131 | 0.8276 | 1.0000 | 127.00 |

Current scientific map:

- Benchmark B v2 is learnable.
- The `needs_final_query` signal is decodable.
- Hard-ST from-scratch discovery still fails across seeds.
- ES is no longer just a late polish trick: on medium basins it can fully rescue the controller, and on weak basins it already fixes routing while leaving a content bottleneck.

## Phase-6 Questions

Primary:

1. Can hard-routing discovery on Benchmark B v2 become robust across seeds with a broader and fairer exploration of controller, memory, objective, and optimization changes?

Secondary:

2. Can weak basins be converted into strong full-task models reliably with staged recovery pipelines, especially ES route rescue followed by gradient content refinement?

Tertiary:

3. Where does hybrid EGGROLL-inspired ES actually help on the improved architecture:
   discovery, rescue, refinement, alternating schedules, or nowhere outside narrow resume settings?

## Hypotheses

1. Phase-5 hard-ST failure is not only a routing-head problem. It is a coupled discovery problem involving:
   weak wait-state initialization, under-shaped retention dynamics, and premature-exit gradients dominating before the wait controller becomes reliable.
2. Weak-basin checkpoints are now mostly separable into:
   - route-bad / content-bad
   - route-good / content-bad
   ES can already move checkpoints into the second regime.
3. A staged pipeline should outperform direct hard-ST discovery:
   ES rescue on routing-control parameters, then short gradient content refinement with routing frozen or nearly frozen.
4. Genuinely different discovery methods may beat hard-ST in the controller-only regime even if they do not beat the best resumed ES result.

## Mandatory Clusters

All clusters get:

- scout suite
- local tuning on the best 1-2 variants
- same-seed rerun for the best tuned config before promotion
- promotion only after rerun is roughly stable

### Cluster A: Weak-Basin Recovery Pipelines

Goal: turn route-rescuable weak checkpoints into strong full-task models.

Initial variants:

1. ES route rescue -> freeze all router/control/release params -> gradient content refinement.
2. ES route rescue -> freeze route head only -> refine packet adapters / content core.
3. ES route rescue -> freeze routing -> refine sink/readout plus content memory heads.
4. ES route rescue -> short grad refine -> short ES refresh.
5. ES route rescue -> teacher distill from medium rescued checkpoint into weak rescued checkpoint.

Local tuning knobs:

- refinement LR
- freeze mask scope
- refinement duration
- adapter-only vs readout+core
- auxiliary loss weights during refinement

### Cluster B: Factorized Wait-Controller Discovery

Goal: make WAIT a first-class persistent decision.

Initial variants:

1. binary `WAIT/ACT` controller with separate `EXIT/FORWARD` act head
2. set/clear latch plus route head
3. hazard-style continue-wait head
4. dwell-time / deadline head
5. separate recurrent control cell with controller-only inputs
6. select-exit / release-gated variants with better initialization and schedules

Local tuning knobs:

- wait bias
- controller LR scale
- temperature schedule
- control write / clear biases
- controller loss weights

### Cluster C: Persistent Control-Memory / Retention Mechanisms

Goal: keep `needs_final_query` alive over long horizons.

Initial variants:

1. dedicated control-memory slot
2. dual-track payload/control memory
3. slower-decay or no-decay control state
4. self-refresh / keepalive write
5. explicit packet metadata wait flag
6. multi-timescale control memory

Local tuning knobs:

- control memory dimension
- refresh strength
- clear rules
- router access scale to control memory

### Cluster D: Alternative Discovery Algorithms

Goal: test whether the discovery problem is fundamentally hard-ST-specific.

Initial methods:

1. controller-only REINFORCE with value/baseline and entropy support
2. lightweight actor-critic / PPO-style controller training
3. DAgger or scheduled imitation-release for wait/control only
4. controller-only CEM / NES over a small factorized controller

Local tuning knobs:

- entropy coefficient
- baseline normalization
- schedule length for imitation release
- population size / elite fraction for CEM

### Cluster E: ES Role Mapping on Improved Architectures

Goal: map exactly where ES helps.

Required sub-studies:

1. resume ES from weak / medium / strong basins
2. router-only vs router+adapter ES
3. alternating `grad -> ES -> grad`
4. renewed ES discovery on a simplified slice or an easier controller architecture
5. same-seed reruns and confirmation evaluations for headline ES claims

Local tuning knobs:

- sigma
- rank
- population
- reward normalization
- evolve-adapters flag
- alternating schedule lengths

### Cluster F: Objective / Curriculum / Supervision

Goal: make discovery and retention trainable without relying on one brittle recipe.

Initial variants:

1. `needs_final_query` auxiliary head
2. temporal consistency loss on control state
3. anti-premature-exit penalty
4. delayed EXIT unlock
5. partial oracle-control imitation then release
6. wait/act-only supervision
7. final-query timing auxiliary
8. entropy support / exploration regularization

Local tuning knobs:

- auxiliary loss weights
- unlock schedule length
- imitation duration
- entropy coefficient

## Fairness And Budget

Budget allocation target:

- 15% anchor reproduction + tooling
- 35% broad scouting across clusters
- 30% local tuning within promising clusters
- 20% confirmations, seed panels, and reporting

Run budget:

- target `>= 60` substantive new runs
- hard floor `>= 50`
- `>= 20` promoted runs
- `>= 6` exact same-seed reruns
- `>= 5` full 5-seed panels
- `>= 6` locked confirmation evaluations

Fairness rule:

- before any cluster consumes more than 25% of the promoted-run budget, every mandatory cluster must have:
  - initial scout suite
  - local tuning on at least one top variant
  - same-seed rerun on the best tuned config

## Verification Ladder

No headline claim is accepted until all five are done:

1. same-seed rerun from scratch
2. 5-seed panel
3. locked confirmation evaluation
4. comparison against the strongest relevant baseline under a similar budget
5. independent verification with a phase-6 verify utility

Negative-result retirement standard:

- at least 4 concrete variants unless the family is inherently narrower and explicitly justified
- at least one tuned top variant
- at least one rerun
- diagnostics showing where it fails
- retirement reason recorded in the phase-6 ledger

## Confirmation Protocol

Main target remains Benchmark B v2.

Phase 6 will add:

1. a locked confirmation evaluation on the standard full benchmark with a new `confirm_seed`
2. a second locked confirmation control with changed generator seed and a targeted final-query-heavy mode mix

If code changes are needed, implement them before promotions. Test will not be used for broad fishing once the phase-6 confirmation configs exist.

## Anchor Reproduction Commands

Poor hard-ST anchor:

```bash
uv run python -m src.train.run \
  --config configs/phase5/dev/hard_st_benchmark_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit.yaml \
  --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
  --results-dir results/phase6_anchor/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_rerun1
```

Medium-basin ES rescue anchor:

```bash
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase5/dev/hybrid_es_benchmark_b_v2_control_router2_setclear_oraclecontrol_resume.yaml \
  --resume results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1/hard_st_best.pt \
  --results-dir results/phase6_anchor/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_rerun1
```

Weak-basin ES partial-rescue anchor:

```bash
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase5/dev/hybrid_es_benchmark_b_v2_control_router2_setclear_oraclecontrol_resume.yaml \
  --resume results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_seed947_p1/hard_st_best.pt \
  --results-dir results/phase6_anchor/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_rerun1
```

Anchor verification:

```bash
uv run python -m src.utils.phase5_verify --run-dir results/phase6_anchor/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_rerun1 --confirm-batches 16
uv run python -m src.utils.phase5_verify --run-dir results/phase6_anchor/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_rerun1 --confirm-batches 16
uv run python -m src.utils.phase5_verify --run-dir results/phase6_anchor/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_rerun1 --confirm-batches 16
```

## Planned New Entry Points

These will be added during phase 6:

```bash
./scripts/run_phase6_cluster_scouts.sh
./scripts/run_phase6_main.sh <cluster> <variant>
./scripts/run_phase6_seed_panels.sh <run-family>
./scripts/run_phase6_confirm.sh <run-dir>

uv run python -m src.utils.phase6_verify --run-dir <run-dir>
uv run python -m src.utils.phase5_audit --run-dir <run-dir> --split test --num-batches 8 --probe-train-batches 8 --probe-test-batches 8
```

## Execution Order

1. Reproduce the three anchors from scratch.
2. Add phase-6 ledger and verification extensions.
3. Add confirmation config support.
4. Run scout suites across all six clusters.
5. Run local tuning sweeps on the best variants in each cluster.
6. Rerun the best tuned configs before promotion.
7. Promote top variants across clusters.
8. Run 5-seed panels and locked confirmation evaluations.
9. Finish ES role mapping on the improved architectures.
10. Write cluster scorecards, report, lessons, and updated README commands.

## Stop Criteria

Positive exit:

- a robust hard-discovery or staged-recovery result beats the phase-5 story on 5 seeds and locked confirmation, and clears the full verification ladder

or

Strong mapping exit:

- all mandatory clusters are explored fairly
- cluster winners are tuned and verified
- the boundaries of hard-ST, staged recovery, and ES are materially clearer than phase 5
- the next recommended experiment is narrower and higher-confidence than before

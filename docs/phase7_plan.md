# Phase 7 Plan

## Starting Point

Repo baseline at phase-7 start:

- commit: `2270040`
- active phase-7 issue: `gnn2-07r`
- prior ledger: [phase6_run_matrix.csv](/home/catid/gnn2/docs/phase6_run_matrix.csv)
- current target benchmark: Benchmark B v2 with locked confirmation configs under
  [configs/phase6/confirm](/home/catid/gnn2/configs/phase6/confirm)

Phase-6 anchor facts to reproduce first:

| Anchor | Run | Split | Overall Acc | Final-Query Acc | Final-Query Route Match | Final-Query Exit |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Direct keepalive discovery | `hard_st_b_v2_controlsticky_keepalive_seed989_p1` | test | 0.5472 | 0.3349 | 0.4472 | 93.57 |
| Weak-basin staged recovery | `hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_p1` | test | 0.9274 | 0.8777 | 0.8369 | 116.85 |
| Strong keepalive memory-only reopen | `hard_st_b_v2_controlsticky_keepalive_refine_memoryonly_seed989_p1` | test | 0.9613 | 0.9826 | 0.9739 | 126.13 |

Current scientific map:

- The benchmark is definitely learnable.
- From-scratch hard-ST discovery is still not robust across seeds.
- The strongest direct discovery result is the keepalive basin, but it is still seed-sensitive.
- Weak-basin ES route rescue plus gradient refinement is already useful.
- Once a real keepalive basin exists, reopening `memory_` alone is enough for near-perfect long-horizon recovery.
- Recovery quality depends strongly on how much late-route structure exists in the source checkpoint.

## Phase-7 Questions

Primary:

1. Can from-scratch hard-routing discovery become robust across seeds by training more directly for the keepalive basin?

Secondary:

2. Can staged recovery become systematic and predictable across weak, medium, and strong source checkpoints?

Tertiary:

3. Where does hybrid EGGROLL-inspired ES help best now:
   discovery, route rescue, content rescue, alternating schedules, or only narrow resume regimes?

## Budget And Fairness

Phase 7 follows a hard 50/50 split:

- Fruitful half: about 48 substantive runs
- Exploration half: about 48 substantive runs

Hard floor:

- at least 80 substantive new runs
- at least 24 promoted runs
- at least 10 exact same-seed reruns
- at least 6 full 5-seed panels
- at least 8 locked confirmation evaluations
- at least 2 exact anchor reproductions from phase 6
- at least 2 fully worked staged-recovery pipelines
- at least 2 serious non-hard-ST discovery attempts

No cluster is retired before:

1. at least 4 concrete variants (unless inherently narrower and explicitly justified)
2. local tuning on at least one top variant
3. an exact same-seed rerun on the best tuned variant
4. diagnostics that explain the failure
5. a retirement note in the phase-7 ledger

## Fruitful Half

### Cluster A: Direct Keepalive-Basin Discovery

Goal: turn the phase-6 best seed into a reproducible basin instead of a lucky seed.

Planned scout variants:

1. keepalive baseline replay
2. keepalive + stronger `control_state_weight` decay schedule
3. keepalive + explicit `set_clear` hybrid control state
4. keepalive + explicit release / exit schedule
5. keepalive + router-visible control-memory scale increase
6. keepalive + tuned write/clear bias pair
7. keepalive + controller LR scale sweep
8. keepalive + control auxiliary sweep

Planned tuning knobs:

- `control_input_scale`
- `control_write_bias`
- `control_clear_bias`
- `control_state_weight_{start,end}`
- `anti_exit_weight_{start,end}`
- `controller_lr_scale`
- `temperature_{start,end}`

Promotion target:

- at least 2 promoted variants
- same-seed rerun for each
- 5-seed panel for the best one if it remains competitive

### Cluster B: Systematic Staged Recovery Pipelines

Goal: make recovery a controlled pipeline rather than a lucky bridge.

Planned scout variants:

1. weak ES rescue -> routing frozen -> `memory_` only refine
2. weak ES rescue -> routing frozen -> `memory_ + readout` refine
3. weak ES rescue -> routing frozen -> `sink_proj + readout` refine
4. weak ES rescue -> routing frozen -> `memory_ + sink_proj + readout`
5. medium rescue -> `memory_` only reopen
6. strong keepalive source -> `memory_` only reopen replay
7. rescue -> short grad refine -> short ES refresh
8. rescue -> teacher-guided content distillation

Planned tuning knobs:

- reopened prefix set
- refine LR
- refine steps (`train_steps_delta`)
- auxiliary loss weights during refine
- whether to freeze full routing vs route head only

Promotion target:

- at least 2 promoted pipelines
- 5-seed panel for the best recovery pipeline
- confirmation eval on weak, medium, and strong source variants

### Cluster C: ES-Assisted Basin Entry And Rescue Mapping

Goal: map whether ES can help before pure late polish.

Planned scout variants:

1. weak-basin resume ES router-only
2. weak-basin resume ES router+adapter
3. medium-basin resume ES router-only replay
4. medium-basin resume ES router+adapter replay
5. strong keepalive resume ES router-only
6. strong keepalive resume ES router+adapter
7. alternating `grad -> ES -> grad`
8. short ES preconditioning before content refinement
9. ES rescue + frozen-route gradient refine
10. ES discovery attempt on simplified final-query-heavy slice

Planned tuning knobs:

- `sigma`
- `rank`
- `population`
- `evolve_adapters`
- reward normalization
- alternating schedule length

Promotion target:

- at least 10 substantive runs
- at least 2 reruns
- at least 2 confirmation evaluations
- 5-seed panel if a pipeline is competitive

## Exploration Half

### Cluster D: Alternative Discovery Algorithms

Goal: test whether discovery is fundamentally a hard-ST optimization issue.

Planned methods:

1. controller-only REINFORCE with entropy support
2. actor-critic style controller-only update
3. controller-only CEM / NES
4. DAgger-style wait/control imitation-release
5. small control-policy search while content remains gradient-trained

Tuning knobs:

- entropy coefficient
- baseline normalization
- imitation horizon
- elite fraction / population
- controller LR

### Cluster E: Different Controller / Memory Families

Goal: try families that are more different from the current path.

Planned variants:

1. monotone wait-then-act controller
2. hazard-style waiting model
3. explicit finite-state wait machine
4. separate recurrent control cell
5. external control-memory track with router visibility
6. explicit packet metadata control bit

Tuning knobs:

- wait-state dimension
- hazard temperature
- finite-state transition biases
- control-router visibility scale

### Cluster F: Supervision / Curriculum / Pretraining Alternatives

Goal: give discovery more learning signal without only increasing route CE.

Planned variants:

1. `needs_final_query` auxiliary prediction
2. temporal consistency loss on control state
3. anti-premature-exit penalty
4. wait/act-only supervision
5. final-query timing auxiliary
6. teacher self-distillation from strong recovered models
7. partial oracle-control imitation then release

Tuning knobs:

- auxiliary weights
- unlock schedule
- imitation duration
- distillation strength

### Cluster G: Transfer / Generalization / Confirmation Stress

Goal: confirm that successful mechanisms are real and not narrow generator artifacts.

Planned controls:

1. locked confirmation full benchmark replay
2. locked confirmation final-query-heavy replay
3. altered trigger distribution
4. altered query distance longer-horizon control
5. altered distractor / noise rate
6. final-query-only evaluation slice
7. source-checkpoint transfer across confirmation settings
8. if needed, Benchmark B v3 alongside B v2, never replacing it

## Verification Ladder

No headline result is called confirmed until all six are done:

1. exact same-seed rerun
2. five-seed panel
3. locked confirmation evaluation
4. comparison against the strongest relevant baseline under similar budget
5. independent verification with a phase-7 verify utility
6. ledger entry updated with confirmation status

Serious negative conclusions require:

1. at least 4 concrete variants in the cluster
2. at least 1 tuned top variant
3. at least 1 same-seed rerun
4. diagnostics showing where it fails
5. a retirement note in the ledger

## Anchor Reproduction Commands

Direct keepalive discovery:

```bash
uv run python -m src.train.run \
  --config configs/phase6/dev/hard_st_benchmark_b_v2_controlsticky_keepalive.yaml \
  --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
  --results-dir results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1
```

Weak-basin staged recovery:

```bash
uv run python -m src.train.run \
  --config configs/phase6/dev/hard_st_benchmark_b_v2_weak_es_content_refine_sinkcore.yaml \
  --resume results/phase6_anchor/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_rerun1/hybrid_es_best.pt \
  --results-dir results/phase7_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1
```

Strong keepalive memory-only reopen:

```bash
uv run python -m src.train.run \
  --config configs/phase6/dev/hard_st_benchmark_b_v2_controlsticky_keepalive_refine_memoryonly.yaml \
  --resume results/phase6_dev/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1/hard_st_best.pt \
  --results-dir results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_refine_memoryonly_seed989_rerun1
```

## Initial Execution Order

1. reproduce the three anchors above
2. create `docs/phase7_run_matrix.csv`
3. create `src/utils/phase7_verify.py` and phase-7 helper scripts
4. run scout suites across Clusters A-G while honoring the 50/50 split
5. run local tuning for the top 1-2 variants in every cluster
6. rerun the best tuned configs
7. promote top configs
8. run 5-seed panels and locked confirmations
9. generate scorecards and reports

## Stop Criteria

Positive exit:

- robust from-scratch discovery improvement across 5 seeds and confirmation, or
- robust staged recovery across weak / medium basins across 5 seeds and confirmation, or
- a broader, verified ES rescue / basin-entry map

Strong mapping exit:

- 50/50 fruitful vs exploration budget honored
- all required clusters explored fairly
- cluster winners tuned and rerun
- major positives and negatives independently verified
- the strongest remaining bottleneck is sharply localized
- the next experiment is narrower and higher-confidence than phase 6

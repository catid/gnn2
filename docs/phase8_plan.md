# Phase 8 Plan

## Starting Point

Phase 8 starts from the verified phase-7 map:

| Item | Run / Panel | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | --- | ---: | ---: | ---: | ---: |
| best direct discovery | [seed989_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1) | 0.5540 | 0.3382 | 0.4358 | 92.86 |
| weak-basin recovery anchor | [seed973_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1) | 0.9173 | 0.8643 | 0.8155 | 115.62 |
| strong-source memory-only reopen | [seed989 memory-only rerun](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_refine_memoryonly_seed989_rerun1) | 0.9564 | 0.9800 | 0.9719 | 125.76 |
| keepalive-anchor ES | [seed1201](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1) | 0.9756 | 0.9500 | 1.0000 | 127.00 |
| medium-source recovery panel | `forceoracle -> memoryreadout` 5 seeds | 0.4400 +/- 0.0202 | 0.2988 +/- 0.0061 | 0.5702 +/- 0.0142 | 96.46 +/- 1.71 |
| strong-source transfer panel | `sinkonlylonger -> sinkreadout` 5 seeds | 0.5143 +/- 0.0077 | 0.2521 +/- 0.0141 | 0.2460 +/- 0.0071 | 86.21 +/- 1.21 |

Working interpretation from phase 7:

- from-scratch hard-ST still fails at reliable basin entry,
- route-faithful source checkpoints matter more than local loss tweaks,
- teacher-shaped sources and adapter ES can preserve late routing,
- strong-source transfer exposed a content bottleneck after route transfer,
- the next step is teacher-seeded direct keepalive basin entry, with teacher removed at inference.

## Repo State Notes

- `README.md` is stale relative to phase 7 and still headlines phase-5 outcomes.
- Phase 8 will update the top-level scientific summary and add phase-8 entrypoints by the end.

## Phase 8 Questions

Primary:

- Can teacher-guided control supervision create a robust, teacher-free basin-entry recipe for from-scratch hard-routing discovery?

Secondary:

- Once the student enters a route-faithful basin, what is the smallest reliable refinement that restores content quality?

Tertiary:

- Which teacher source, channel set, release schedule, and ES stage are actually necessary?

## Budget Split

Target substantive budget: 100+ runs. Hard floor: 84.

Planned split:

- fruitful half: 60 runs
- exploration half: 40 runs

Approximate promoted-run allocation:

- fruitful half: 18 promoted
- exploration half: 10 promoted

Approximate verification allocation:

- 12+ same-seed reruns
- 7+ five-seed panels
- 10+ locked confirmation evaluations

## Cluster Map

### Fruitful Half

#### Cluster A: Teacher-Seeded Direct Keepalive Basin Entry

Goal:

- enter the keepalive basin directly during from-scratch hard-ST training using narrow teacher control supervision only during training

Core variants:

1. teacher wait/release only, early warmup then hard release
2. teacher control-state only
3. teacher wait/release + control-state
4. teacher wait/release + control-state with final-query-only target scope
5. delay-required-only teacher supervision
6. gradual teacher anneal
7. intermittent teacher dropout during warmup
8. teacher-channel noise / dropout
9. teacher-seeded control-memory partial init
10. teacher wait/release + sparse late-step control auxiliary

Local tuning axes:

- teacher weights
- warmup length
- release schedule shape
- control-state weight schedule
- controller LR scale

Promotion target:

- at least 3 reproducible promoted runs

#### Cluster B: Teacher Source / Channel / Release Mapping

Teacher sources:

1. strong route-faithful ES-assisted keepalive source
2. medium `forceoracle -> memoryreadout` source
3. strong transfer-style route-positive / content-weak source
4. one weaker teacher-shaped negative-control source

Channel families:

1. wait/release only
2. control-state only
3. wait/release + control-state
4. full-route upper-bound ablation

Release schedules:

1. abrupt
2. linear anneal
3. dropout/intermittent
4. confidence-gated if practical

Promotion target:

- at least 2 promoted comparisons backed by a source/channel/release table

#### Cluster C: Post-Entry Content Recovery Pipelines

Goal:

- identify the smallest reliable refine step after route-faithful basin entry

Variants:

1. routing frozen -> memory only
2. routing frozen -> memory + readout
3. routing frozen -> sink/readout only
4. routing frozen -> sinkcore
5. refine -> short ES refresh -> refine
6. teacher-seeded basin entry -> medium recovery
7. teacher-seeded basin entry -> weak recovery

Promotion target:

- at least 2 promoted pipelines

#### Cluster D: ES on Teacher-Seeded and Recovery Sources

Goal:

- map where ES helps once teacher-seeded sources exist

Variants:

1. router-only ES on teacher-seeded basin-entry student
2. adapter ES on teacher-seeded basin-entry student
3. ES immediately after teacher release
4. ES after content refinement
5. grad -> ES -> grad
6. short ES designed for basin widening
7. medium teacher-shaped source vs strong teacher-shaped source

Promotion target:

- at least 1 five-seed competitive ES-assisted pipeline

### Exploration Half

#### Cluster E: RSM-Inspired Detached-Warmup Discovery

Goal:

- test whether detached warmup / terminal-only supervision reduces greedy early-exit failure

Variants:

1. terminal-only supervision
2. detached warmup steps
3. late-window-only supervision
4. one-step detached span
5. two-step detached span
6. wait-horizon curriculum + detached warmup

#### Cluster F: Different Control Architectures

Variants:

1. monotone wait-then-act
2. hazard-style wait head
3. finite-state control
4. separate recurrent control cell
5. external control memory track
6. metadata control bit with learned update
7. WAIT/ACT then EXIT/FORWARD decomposition
8. clear-on-final-query structured latch

#### Cluster G: Alternative Discovery Algorithms

Variants:

1. improved REINFORCE baseline
2. PPO-like controller-only surrogate
3. DAgger / scheduled imitation-release on control channels
4. controller-only CEM / NES
5. policy search with frozen content trunk

#### Cluster H: Stress / Confirmation / Generalization

Locked confirmations:

1. existing `full_locked`
2. existing `finalquery_heavy`
3. altered query-distance / distractor shift
4. alternate generator-seed confirmation

Every headline result must be evaluated here before being called robust.

## Promotion Rules

Each cluster follows the same ladder:

1. scout: at least 4 concrete variants
2. local tuning: at least 2 top variants
3. same-seed rerun of the best tuned config
4. promotion only if the rerun roughly matches
5. verification:
   - five-seed panel if competitive
   - locked confirmations
   - independent recomputation with `phase8_verify`

Retirement requires:

- 4+ concrete variants unless explicitly narrower
- 1 tuned top variant
- 1 exact rerun
- diagnostics explaining the failure
- retirement note in the ledger

## Verification Protocol

A result is only `confirmed` when all six are true:

1. exact same-seed rerun
2. five-seed panel
3. locked confirmation evaluation
4. comparison to the strongest relevant baseline
5. independent recomputation via `src/utils/phase8_verify.py`
6. ledger row updated with confirmation status

Teacher honesty rules:

- teacher used only during training
- teacher removed at inference
- report exact teacher source and teacher channels
- main claims use narrow teacher channels by default
- full-route teacher is upper-bound only

## Anchor Reproduction Commands

All anchor reproductions write under `results/phase8_anchor`.

```bash
./scripts/run_phase8_main.sh \
  configs/phase6/dev/hard_st_benchmark_b_v2_controlsticky_keepalive.yaml \
  results/phase8_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1

./scripts/run_phase8_main.sh \
  configs/phase6/dev/hard_st_benchmark_b_v2_weak_es_content_refine_sinkcore.yaml \
  results/phase8_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1

./scripts/run_phase8_main.sh \
  configs/phase6/dev/hard_st_benchmark_b_v2_controlsticky_keepalive_refine_memoryonly.yaml \
  results/phase8_anchor/hard_st_b_v2_controlsticky_keepalive_refine_memoryonly_seed989_rerun1

./scripts/run_phase8_main.sh \
  configs/phase7/dev/hybrid_es_benchmark_b_v2_controlsticky_keepalive_resume.yaml \
  results/phase8_anchor/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1 \
  results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1/hard_st_best.pt \
  2
```

Verification:

```bash
./scripts/run_phase8_confirm.sh results/phase8_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1
./scripts/run_phase8_confirm.sh results/phase8_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1
./scripts/run_phase8_confirm.sh results/phase8_anchor/hard_st_b_v2_controlsticky_keepalive_refine_memoryonly_seed989_rerun1
./scripts/run_phase8_confirm.sh results/phase8_anchor/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1
```

## Initial Phase-8 Sweep Commands

Teacher sweeps will write under `results/phase8_dev`.

```bash
./scripts/run_phase8_teacher_sweeps.sh
./scripts/run_phase8_cluster_scouts.sh
```

Promoted runs will use:

```bash
./scripts/run_phase8_main.sh <config> <results-dir> [resume] [nproc]
./scripts/run_phase8_seed_panels.sh <config> <results-root> <resume> <seed1> [seed2 ...]
./scripts/run_phase8_confirm.sh <run-dir> [extra-eval-config ...]
```

## Planned Outputs

- `docs/phase8_run_matrix.csv`
- `src/utils/phase8_verify.py`
- `scripts/run_phase8_cluster_scouts.sh`
- `scripts/run_phase8_main.sh`
- `scripts/run_phase8_confirm.sh`
- `scripts/run_phase8_seed_panels.sh`
- `scripts/run_phase8_teacher_sweeps.sh`
- `docs/phase8_cluster_scorecards.md`
- `docs/phase8_report.md`
- `docs/phase8_lessons.md`

## Stop Criteria

Positive exit:

- teacher-free basin-entry result beats the old direct-discovery anchor by at least
  - +0.15 FQ accuracy
  - +0.20 FQ route
  - +15 FQ exit
- across a five-seed panel and locked confirmations

Strong mapping exit:

- exact source/channel/release requirements for basin entry are clear
- post-entry content bottleneck boundary is clearer
- ES role before/during/after release is clearer
- clusters were explored fairly under the 60/40 split
- major positives and negatives were independently verified

## Execution Order

1. write phase-8 plan and ledger scaffold
2. reproduce anchors
3. create `phase8_verify` and phase-8 scripts
4. run cluster-A and cluster-B teacher sweeps first
5. run exploration scouts in parallel to preserve the 60/40 split
6. tune, rerun, and promote the strongest variants
7. run five-seed panels and locked confirmations
8. write report, lessons, scorecards, and README update
9. close or update `bd` issues and push

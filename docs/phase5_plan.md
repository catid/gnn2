# Phase 5 Plan: Robust Hard-Routing Discovery And ES Role Mapping

## Scope

Phase 5 is a larger follow-up to phase 4. The goal is not to re-run the same
single-seed controller idea. The goal is to:

1. verify the phase-4 story from scratch on this machine,
2. make hard-routing `delay_to_final_query` discovery more robust across seeds,
3. and map the exact regime where the EGGROLL-inspired hybrid ES method helps.

The main target benchmark remains Benchmark B v2. Benchmark changes are allowed
only as explicit controls or confirmation slices; Benchmark B v2 stays in place.

## Starting Point

Starting metrics are copied from the checked-in phase-4 report and the saved
phase-4 summaries.

### Phase-4 hard-ST story

- Best dev single seed:
  `results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2`
  - overall accuracy: `0.7539`
  - final-query accuracy: `0.5154`
  - final-query exit time: `72.02`
  - final-query route match: `0.3543`
- Larger audit of that same seed:
  - final-query accuracy: `0.5041`
  - final-query exit time: `70.07`
  - final-query route match: `0.3312`
  - probes from packet, memory, router, and control state: `1.000`
- Promoted seed panel for the best robust family
  (`sticky_both_dim8`, phase-4 main seeds `750/752/753/754`):
  - mean overall accuracy: `0.5801`
  - mean final-query accuracy: `0.2600`
  - mean final-query exit time: `31.96`
  - mean final-query route match: `0.0000`
- Phase-4 conclusion:
  the benchmark is learnable and the signal is decodable, but hard-ST discovery
  is not robust across seeds.

### Phase-4 hybrid ES story

- Scratch hybrid on the improved controller still collapsed to early exit.
- Resume-based hybrid ES from a working hard-ST checkpoint succeeded:
  `results/phase4_main/hybrid_es_b_v2_control_sticky_aux_router2_resume_seed747`
  - resumed checkpoint before ES:
    - overall accuracy: `0.8013`
    - final-query accuracy: `0.5930`
    - final-query exit time: `77.34`
    - final-query route match: `0.4480`
  - resumed hybrid ES after polish:
    - overall accuracy: `1.0000`
    - final-query accuracy: `1.0000`
    - final-query exit time: `127.00`
    - final-query route match: `1.0000`

### Current working hypothesis

The current controller still asks a single 3-way router head to preserve a
persistent wait decision through generic latent state. The key phase-4 insight
was that the signal is decodable but the router does not robustly obey it.

Phase-5 working hypothesis:

1. Hard-ST discovery fails because "keep waiting" is not a first-class stable
   decision in the controller.
2. A factorized wait controller, plus better state-retention supervision and
   variance-reducing optimization choices, can improve robustness across seeds.
3. Hybrid ES is more likely to help as rescue / refinement than from-scratch
   discovery, but that needs a broader map than the single phase-4 resume win.

## Anchor Reproductions

These must be rerun from scratch before major new claims:

1. Representative hard-ST anchor:
   `configs/phase4/main/hard_st_benchmark_b_v2_control_sticky_aux_router2_main.yaml`
   with a phase-4 main seed run from the phase-3 oraclewarm checkpoint.
2. Representative resumed hybrid ES anchor:
   `configs/phase4/main/hybrid_es_benchmark_b_v2_control_sticky_aux_router2_resume_seed747.yaml`
   resuming from
   `results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/hard_st_best.pt`.

Anchor reproduction rule:

- run from scratch,
- record metrics in the phase-5 registry,
- compare against the checked-in phase-4 summary,
- do not treat either anchor as reproduced until the rerun is in the same
  qualitative regime.

## Required Infrastructure Work

Before the main family sweeps:

1. add a machine-readable experiment ledger,
2. add stricter evaluation support with confirmation seeds / confirmation split,
3. add an independent verification utility,
4. extend audit tooling for controller traces, probe decodability, and
   seed-to-seed variance summaries.

Planned files:

- `docs/phase5_run_matrix.csv`
- `src/utils/phase5_verify.py`
- `scripts/run_phase5_main.sh`
- `scripts/run_phase5_confirm.sh`
- `scripts/run_phase5_seed_panels.sh`

## Validation / Test / Confirmation Protocol

Phase 5 uses a stricter evaluation protocol than phase 4.

### Splits

- `train`: used for optimization only
- `val`: used for broad sweeps and early stopping
- `test`: used for promoted runs only
- `confirm`: locked confirmation split with distinct generator seeds

### Headline-claim protocol

Every headline result must satisfy all of:

1. pilot run,
2. exact same config + same seed rerun from scratch,
3. promoted 5-seed panel,
4. locked confirmation evaluation,
5. independent recomputation via `src.utils.phase5_verify`,
6. direct comparison against the strongest relevant baseline.

## Promotion Rules

Promote a pilot to a main-tier run only if at least one of these is true on
validation:

- `per_mode.delay_to_final_query.accuracy` improves by `>= 0.10` over the
  promoted phase-4 hard-ST family,
- mean final-query exit time rises above `90`,
- final-query route match is clearly above zero and not a tiny artifact,
- confirmation-style re-eval on a held-out generator seed shows the same
  qualitative behavior.

Reject candidates that only improve easy modes while final-query behavior stays
 at phase-4 levels.

## Intervention Families

### Family A: Factorized Wait Controller

This is the primary family and gets the largest budget.

Required concrete variants:

1. hierarchical router:
   first `WAIT vs ACT`, then if `ACT` choose `EXIT vs FORWARD`
2. binary wait latch + route head:
   explicit persistent wait bit plus separate action head
3. hazard-style continue-wait controller:
   explicit continue probability / stop hazard with a separate act head
4. recurrent control cell:
   dedicated control-state cell separated from payload path

Promotion target:
at least 2 variants beyond pilots and at least 1 five-seed panel.

### Family B: Persistent Control Memory / State Retention

Concrete variants:

1. dedicated control-memory slot separate from payload memory
2. control keepalive / self-refresh update
3. slower-decay or no-decay control memory
4. explicit router access to control-memory readout

Required checks:

- probe decodability over time,
- final-query exit-time shift,
- route-match improvement if any.

### Family C: Objective / Supervision / Curriculum

Concrete variants:

1. per-step `needs_final_query` auxiliary head
2. temporal consistency loss on wait/control state
3. anti-premature-exit loss targeted at final-query mode
4. wait-factor supervision only, without full-route imitation
5. final-query timing auxiliary target

At least one promoted run must come from this family, and at least one variant
must not simply increase route CE weight.

### Family D: Discovery Robustness / Optimization / Initialization

Concrete variants:

1. separate optimizer groups for controller vs rest of model
2. controller-specific learning rate / weight decay / clipping
3. controller initialization bias sweeps
4. temperature / Gumbel / exploration schedule for hard-ST discovery
5. horizon curriculum or batch-size schedule

At least one seed-panel comparison in this family must focus on variance
reduction rather than peak single-seed quality.

### Family E: ES Role Mapping And Stress Tests

Required sub-studies:

1. resume ES polish from weak / medium / strong checkpoints
2. router-only ES vs router+adapter ES on improved architectures
3. alternating gradient/ES schedule
4. at least one simplified discovery attempt on a final-query-only or easier
   slice
5. multi-seed confirmation that any ES gain is not single-seed luck

Phase-5 goal for ES is to map:

- from-scratch discovery,
- mid-training rescue,
- late polish,
- or no useful role.

### Family F: Benchmark Controls / Confirmation Splits

Required controls:

1. locked confirmation split with distinct generator seeds
2. at least one held-out difficulty control:
   altered trigger positions, retrieval distance, or final-query-only slice

Benchmark B v2 stays as the primary benchmark. Any new slice must be additive.

## Diagnostics To Add

Phase 5 will extend the audit tooling with:

- per-mode accuracy tables
- final-query accuracy, route match, and exit-time histogram
- final-query delay-persistence curve
- router-logit trajectories for representative cases
- decodability probes for `needs_final_query` from:
  - packet state
  - control memory/state
  - router input
- memory read/write statistics and similarity traces
- route entropy / router confidence
- seed-to-seed variance plots
- validation vs test vs confirmation tables
- ES reward variance, rank/sigma/population diagnostics

## Run Budget

Minimum phase-5 campaign:

- 2 anchor reproductions
- 30+ substantive new runs
- 10+ promoted runs
- 3 full 5-seed panels
- 2 locked confirmation evaluations
- 2 exact same-seed reruns
- 1 renewed ES-from-resume campaign
- 1 renewed ES discovery or simplified-discovery campaign

Budget allocation target:

- Family A: 10-12 runs
- Family B: 6-8 runs
- Family C: 5-6 runs
- Family D: 5-6 runs
- Family E: 6-8 runs
- Family F controls / confirmations: 4+ runs

Some runs will count toward more than one family, but the ledger will mark the
primary family and hypothesis.

## Planned Execution Order

1. Read the current repo state and write this plan.
2. Create the run ledger and stricter evaluation plumbing.
3. Reproduce the two phase-4 anchor results from scratch.
4. Run the first Family A factorized-controller pilots.
5. Add/extend diagnostics in parallel with the Family A pilots.
6. Promote the best Family A variants and run same-seed reruns.
7. Expand through Families B, C, and D using the best controller family.
8. Run Family E ES mapping on weak / medium / strong checkpoints.
9. Run 5-seed panels and confirmation evaluations for promoted winners.
10. Write `docs/phase5_report.md` and `docs/phase5_lessons.md`.

## Exact Command Templates

Anchor reproduction:

```bash
uv run python -m src.train.run \
  --config configs/phase4/main/hard_st_benchmark_b_v2_control_sticky_aux_router2_main.yaml \
  --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
  --results-dir results/phase5_anchor/<run_name>

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase4/main/hybrid_es_benchmark_b_v2_control_sticky_aux_router2_resume_seed747.yaml \
  --resume results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/hard_st_best.pt \
  --results-dir results/phase5_anchor/<run_name>
```

Phase-5 pilots:

```bash
uv run python -m src.train.run \
  --config configs/phase5/dev/<candidate>.yaml \
  --results-dir results/phase5_dev/<run_name>
```

Promoted hard-ST runs:

```bash
uv run python -m src.train.run \
  --config configs/phase5/main/<candidate>.yaml \
  --results-dir results/phase5_main/<run_name>
```

Promoted hybrid ES runs:

```bash
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase5/main/<candidate>.yaml \
  --results-dir results/phase5_main/<run_name>
```

Independent verification:

```bash
uv run python -m src.utils.phase5_verify \
  --run-dir results/phase5_main/<run_name> \
  --checkpoint <optional checkpoint>
```

## Stop Criteria

### Positive exit

Hard-routing discovery is considered robust only if:

- final-query accuracy is materially above the phase-4 hard-ST story,
- final-query exit timing moves substantially closer to the real final query,
- route match is non-trivial,
- the result holds across 5 seeds,
- confirmation evaluation agrees,
- and independent verification agrees.

### Strong mapping exit

If robust hard-ST discovery still fails, phase 5 must still deliver a much
stronger map than phase 4:

- all six families explored sufficiently,
- exact conditions where hard-ST succeeds or fails,
- exact conditions where ES helps or does not help,
- multiple confirmed positives and negatives,
- and a narrower next experiment than `gnn2-hzo`.

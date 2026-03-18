# Phase 4 Plan: Durable Final-Query Control Retention

## Scope

Phase 4 targets the remaining Benchmark B v2 failure mode from phase 3:
the model can now delay, but it does not durably preserve the latent
`needs_final_query` control signal after the trigger event.

This is a targeted research campaign, not a general cleanup pass.

## Starting Point

Starting metrics are taken from the checked-in phase-3 release note and
result summaries under `results/phase3_dev/`.

### Current phase-3 hard-routing release baseline

- Config family: `configs/phase3/dev/hard_st_benchmark_b_v2_keymem_payloadaux_release_nomask*.yaml`
- 3-seed overall test accuracy: `0.6110 +/- 0.0308`
- 3-seed route match: `0.4851 +/- 0.0399`
- 3-seed delay rate: `0.6922 +/- 0.0145`
- 3-seed early-exit rate: `0.2614 +/- 0.0078`

### Per-mode phase-3 hard-routing release behavior

- `easy_exit`: solved
- `delay_to_trigger_exit`: mean accuracy about `0.919`, route match about `0.891`
- `delay_to_final_query`: mean accuracy about `0.244`, route match about `0.0004`
- `delay_to_final_query` mean exit time: about `25-33`, not near the true final query at `127`

### Key controls already settled

- Mask-curriculum release is not the answer. It collapses back to immediate exit.
- Higher non-masked `oracle_route_weight` alone is not the answer. It nudges
  overall accuracy slightly but still exits around step `33-34`.
- Oracle-routed keyed-memory runs nearly solve Benchmark B v2, which shows the
  content path is good enough when routing/control is correct.

### Hybrid ES starting point

- `configs/phase3/dev/hybrid_es_benchmark_b_v2_keymem_payloadaux_maskcurr_pop64.yaml`
- Result: collapse to early exit under the old mask curriculum
- Conclusion: hybrid ES must be re-evaluated only after the architecture can
  preserve final-query control state without that curriculum

## Exact Failure Hypothesis

The remaining failure is not reluctance to delay in general. The model delays
heavily after the trigger, but it treats `delay_to_trigger_exit` and
`delay_to_final_query` too similarly after the trigger event.

Primary hypothesis:

1. The model can encode payload information well enough.
2. The model does not maintain a durable, decodable `needs_final_query` control
   signal from trigger time to the final query.
3. That signal is either:
   - not being written cleanly,
   - not retained in memory or packet state,
   - not exposed clearly enough to the router,
   - or not made behaviorally valuable enough by the objective.

Corollary:

- A separate persistent control-memory track may be needed in addition to the
  current payload-oriented keyed packet memory.

## Required Audit

Before claiming any intervention works, phase 4 must show where the control
signal disappears.

Audit outputs to generate:

- per-mode accuracy including `delay_to_final_query`
- final-query exit-time histogram
- route-action histogram over time for final-query cases
- router-logit trajectories by mode
- packet age and TTL histograms by mode
- memory read/write statistics and slot-use traces by mode
- probe analysis for `needs_final_query` decodability from:
  - packet state
  - packet memory / control memory
  - router input features
- compute-penalty contribution by mode

## Intervention Families

### Family A: Persistent control-state design

At least two interventions from this family will be implemented and tested.

Planned candidates:

1. Separate control-memory slot or control vector distinct from payload memory
2. Sticky control latch written on trigger and cleared only on final query / exit
3. Router access to explicit control-memory features

### Family B: Objective / supervision / curriculum

At least two interventions from this family will be implemented and tested.

Planned candidates:

1. Auxiliary `needs_final_query` prediction at each step
2. Temporal consistency / persistence loss on control state
3. Anti-premature-exit penalty for final-query mode
4. Final-query-specific release warm start without reverting to the old mask curriculum

### Family C: Hard-routing / hybrid-ES follow-through

This family is required after a better hard-ST architecture exists.

Planned candidates:

1. Hybrid ES retest on the improved architecture without the old mask curriculum
2. Router-only vs router+adapter ES on the improved architecture
3. One small sigma/rank/population sweep if the first hybrid ES rerun is promising

## Promotion Criteria

Promote a dev candidate to main-tier only if it satisfies at least one of:

- `delay_to_final_query` accuracy improves by at least `+0.08` over the current
  phase-3 release baseline on a like-for-like dev comparison
- mean final-query exit time rises above `80`
- final-query route match is clearly above zero and stable
- latent probe decodability improves materially and the behavior follows it

Reject candidates that only increase overall accuracy by exploiting
`easy_exit` or `delay_to_trigger_exit` while leaving final-query behavior unchanged.

## Success / Stop Criteria

### Positive-result exit

Benchmark B v2 must achieve all of:

1. `delay_to_final_query` accuracy `>= 0.40` averaged over 3 seeds, or at least
   `+0.15` absolute over the current phase-3 final-query accuracy
2. evidence the policy actually waits for the final query:
   - mean final-query exit time `>= 100`, or
   - clearly improved final-query route match
3. overall test accuracy competitive with or better than the best phase-3 release
4. reproduction across 3 seeds
5. hybrid ES retested on the improved architecture

### Strong-negative-result exit

After the full campaign, if final-query behavior still does not materially
improve, the report must isolate the bottleneck tightly enough that the next
step is obvious:

- memory architecture insufficiency
- missing control-state supervision
- router readout mismatch
- or hybrid-ES/search mismatch on the revised model

## Run Budget

Minimum campaign budget:

- 1 audit pass on the current phase-3 checkpoint family
- 3 to 5 small correctness pilots
- 8 to 10 dev runs across the intervention families
- 2 promoted main-tier runs
- 3-seed rerun for the best hard-routing candidate
- 1 hybrid-ES retest on the improved architecture

Target total: `15+` substantive new runs.

## Planned Run Order

1. Add audit instrumentation and trace/probe support.
2. Audit current phase-3 checkpoints to localize the control-state failure.
3. Implement first persistent-control intervention.
4. Run small pilots, then a short dev sweep for Family A.
5. Implement objective/supervision interventions from Family B.
6. Run dev sweeps combining the best Family A and Family B candidates.
7. Promote the best 1-2 hard-routing candidates to main-tier runs.
8. Run a 3-seed comparison of the best new hard-routing candidate.
9. Retest hybrid ES on the improved architecture.
10. Write `phase4_report.md` and `phase4_lessons.md`.

## Exact Command Template

Smoke / audit examples:

```bash
uv run python -m src.train.run --config configs/phase4/dev/<candidate>.yaml
uv run python -m src.utils.report --results_dir results/phase4_dev/<run_name> --out docs/phase4_report.md
```

Multi-GPU hybrid ES:

```bash
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run --config configs/phase4/dev/<hybrid_candidate>.yaml
```

Promoted main-tier:

```bash
uv run python -m src.train.run --config configs/phase4/main/<candidate>.yaml
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run --config configs/phase4/main/<hybrid_candidate>.yaml
```

## Expected Deliverables

- `docs/phase4_plan.md`
- `docs/phase4_report.md`
- `docs/phase4_lessons.md`
- `configs/phase4/...`
- `results/phase4_*`
- updated README commands
- updated `bd` issue state

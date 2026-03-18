# Phase 4 Report

This report tracks the phase-4 campaign on Benchmark B v2, focused on the
remaining phase-3 failure mode: durable retention of the `needs_final_query`
control state.

## Starting Point

Phase 4 started from the phase-3 no-mask hard-routing release result and its
remaining bottleneck:

| Prior result | Test acc | Delay rate | Route match | Early-exit rate | Compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase-2 hard-ST baseline, 3 seeds | 0.4446 +/- 0.0048 | 0.0002 +/- 0.0001 | 0.2559 +/- 0.0000 | 0.9997 +/- 0.0003 | 1.0003 +/- 0.0003 |
| Phase-3 no-mask hard-routing release, 3 seeds | 0.6110 +/- 0.0308 | 0.6922 +/- 0.0145 | 0.4851 +/- 0.0399 | 0.2614 +/- 0.0078 | 23.4136 +/- 2.6542 |

Phase-3 improved `delay_to_trigger_exit`, but `delay_to_final_query` still
failed badly.

## Phase-4 Audit

The first phase-4 job was a targeted audit of the old failure mode using
`src.utils.phase4_audit`.

### Phase-3 release audit

- `delay_to_final_query` accuracy: `0.2563`
- mean exit time: `32.81`
- route match: `0.0010`
- premature-exit rate: `0.9990`

The surprising result was that linear probes could already decode
`needs_final_query` almost perfectly from the old model:

- packet state probe: `1.000`
- memory read state probe: `1.000`
- router probability probe: `1.000`

That changed the working hypothesis. The old model was not purely failing to
represent the signal; it was failing to preserve and use that signal in the
hard router.

### Instrumentation added

Phase 4 added:

- explicit control-state tracing in `RoutingForwardOutput`
- optional sticky and set-clear control-state mechanisms
- direct control-to-router logit projections
- `needs_final_query` control supervision and anti-premature-exit supervision
- per-mode final-query metrics, including `premature_exit_rate` and
  `final_query_wait_gap`
- traced audit outputs with final-query exit histograms, route traces, memory
  traces, and probe summaries

## Intervention Families

### Family A: Persistent control-state design

- sticky control state with direct router projection
- set-clear control state with explicit clear gate
- stronger router coupling via `control_router_scale`

### Family B: Objective and behavioral shaping

- explicit `needs_final_query` control supervision
- anti-premature-exit loss on the final-query wait window

### Family C: Hard-routing / ES follow-through

- main-tier seed reruns on the strongest new architecture
- planned hybrid EGGROLL-inspired ES retest on that improved architecture

## Dev Results

Finished dev runs to date:

| Run | Overall acc | Final-query acc | Final-query exit time | Final-query route match | Compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hard_st_b_v2_control_sticky_aux` | 0.6406 | 0.2627 | 33.16 | 0.0000 | 25.27 |
| `hard_st_b_v2_control_sticky_antiexit` | 0.6419 | 0.2714 | 43.00 | 0.0007 | 30.25 |
| `hard_st_b_v2_control_setclear_both` | 0.5312 | 0.2533 | 27.20 | 0.0013 | 19.23 |
| `hard_st_b_v2_control_sticky_both` | 0.6855 | 0.3797 | 107.71 | 0.6297 | 61.50 |
| `hard_st_b_v2_control_sticky_both_dim8` | 0.7152 | 0.4218 | 104.18 | 0.5020 | 60.55 |
| `hard_st_b_v2_control_sticky_aux_router2` | 0.7539 | 0.5154 | 72.02 | 0.3543 | 43.90 |
| `hard_st_b_v2_control_sticky_aux_router2_both` | 0.6084 | 0.2794 | 60.21 | 0.0461 | 37.57 |
| `hard_st_b_v2_control_sticky_aux_router4` | 0.6185 | 0.2393 | 32.81 | 0.0033 | 24.98 |

Interpretation of the dev sweep:

- pure control supervision is not enough
- pure anti-exit pressure is not enough
- set-clear control is actively harmful on this benchmark
- increasing router coupling too far (`router4`) collapses back toward the old
  early-exit failure
- adding anti-exit back on top of the best router-coupled model also hurts
- the best single-seed tradeoff is `sticky_aux_router2`
- the strongest patience signal is `sticky_both_dim8`, but at much higher
  compute

## Audit Comparison Of The Two Best Dev Runs

`hard_st_b_v2_control_sticky_both_dim8`:

- final-query accuracy: `0.4188`
- mean exit time: `103.31`
- route match: `0.5035`
- premature-exit rate: `0.4924`

`hard_st_b_v2_control_sticky_aux_router2`:

- final-query accuracy: `0.5227`
- mean exit time: `73.69`
- route match: `0.3693`
- premature-exit rate: `0.6307`

This was the central dev-stage tradeoff:

- larger control state plus anti-exit makes the model wait much later
- stronger control-to-router coupling yields better final-query accuracy and
  better overall accuracy at lower compute on a favorable seed

## Larger-Batch Audit Of The Best Dev Seed

The best dev run was re-audited with more batches to check whether its apparent
gain was just a small-eval artifact.

`hard_st_b_v2_control_sticky_aux_router2`, larger audit:

- final-query accuracy: `0.5041`
- mean exit time: `70.07`
- route match: `0.3312`
- premature-exit rate: `0.6688`
- probe accuracy from control state, packet state, memory read state, and router
  probabilities: all `1.000`

That matters. The best dev result was real for that seed. The failure of phase 4
is therefore not that the dev sweep hallucinated an improvement; it is that the
improvement did not survive promoted robustness testing.

Artifacts:

- `results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/artifacts/phase4_audit/final_query_exit_hist.png`
- `results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/artifacts/phase4_audit/final_query_action_traces.png`
- `results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/artifacts/phase4_audit/final_query_memory_traces.png`
- `results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/artifacts/phase4_audit/probe_accuracy.png`

## Promoted Hard-Routing Runs

Two architecture families were promoted to main-tier.

### Promoted family 1: `sticky_aux_router2`

This was the best dev-seed configuration, so it was the first thing promoted.

| Run | Overall acc | Final-query acc | Final-query exit time | Final-query route match | Compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| `seed770` | 0.4502 | 0.2575 | 0.50 | 0.0000 | 1.38 |
| `seed771` | 0.6338 | 0.2515 | 32.74 | 0.0000 | 25.02 |
| mean | 0.5420 | 0.2545 | 16.62 | 0.0000 | 13.20 |

This family failed decisively under promotion. One seed collapsed all the way
back to immediate exit and the other still had zero final-query route match.

### Promoted family 2: `sticky_both_dim8`

Because `sticky_aux_router2` was clearly fragile, the dim8 sticky+anti-exit
family was promoted next as the strongest "force the model to wait" design.

| Run | Overall acc | Final-query acc | Final-query exit time | Final-query route match | Compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| `seed750` | 0.6245 | 0.2435 | 39.67 | 0.0000 | 28.38 |
| `seed752` | 0.4836 | 0.2655 | 12.02 | 0.0000 | 9.30 |
| `seed753` | 0.6133 | 0.2595 | 40.66 | 0.0000 | 28.57 |
| `seed754` | 0.5989 | 0.2715 | 35.50 | 0.0000 | 25.16 |
| mean | 0.5801 | 0.2600 | 31.96 | 0.0000 | 22.85 |

For the required 3-seed panel (`seed750`, `seed752`, `seed753`), the result is:

- overall accuracy: `0.5738 +/- 0.0639`
- final-query accuracy: `0.2562 +/- 0.0093`
- final-query exit time: `30.78 +/- 13.28`
- final-query route match: `0.0000`

This family also failed the positive criterion. It delayed a bit more than the
old phase-3 release, but it still exited far before the real final query and
never learned the correct final-query route on promoted seeds.

## Robustness Interpretation

Phase 4 therefore supports a stronger and more precise negative result:

- the benchmark is not obviously flawed, because oracle-routed runs and the best
  dev seed both show that final-query waiting is learnable
- the model can represent the `needs_final_query` signal, because probes remain
  perfect
- explicit sticky control helps on favorable seeds
- but the current training recipe is not robust enough to make that behavior
  reliable across promoted seeds
- the failure mode is no longer "can the model ever wait?" but "can the model
  reliably make the hard router obey a decodable control state?"

## Hybrid ES Retest

The first hybrid rerun on the improved architecture exposed a systems issue
rather than a scientific result:

- only rank 0 executes the warmstart phase
- rank 1 waits at a distributed barrier
- the original code used the default 10-minute process-group timeout
- the longer phase-4 warmstart exceeded that timeout and the job failed before
  ES actually started

Phase 4 fixed this by adding an explicit distributed timeout in
`src/train/run.py` and restarted the hybrid rerun into:

- `results/phase4_main/hybrid_es_b_v2_control_sticky_both_main_timeoutfix`

That timeout-fixed scratch hybrid rerun still relearned the old failure mode
during warmstart. By late warmstart validation:

- final-query accuracy stayed around `0.237` to `0.251`
- final-query exit time stayed at `0.0`
- final-query route match stayed at `0.0`
- premature-exit rate stayed effectively `1.0`

So hybrid ES from scratch is still not a discovery mechanism for this problem.

### Resume-based hybrid ES on a working controller

The more informative family-C test was to start ES from the best improved
hard-routing checkpoint instead of asking hybrid ES to rediscover the controller
from scratch.

Phase 4 added explicit resume support to `run_hybrid_es`, then launched:

- `results/phase4_main/hybrid_es_b_v2_control_sticky_aux_router2_resume_seed747`

using the working checkpoint:

- `results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/hard_st_best.pt`

Before ES, that resumed hard-ST checkpoint was reevaluated under the current
Benchmark B v2 config and evaluation loop:

| Model | Overall acc | Final-query acc | Final-query exit time | Final-query route match | Compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| resumed hard-ST checkpoint | 0.8013 | 0.5930 | 77.34 | 0.4480 | 46.82 |
| resumed hybrid ES | 1.0000 | 1.0000 | 127.00 | 1.0000 | 71.07 |

The resumed hybrid ES run used both GPUs and 16 CPU threads, and it stayed at
perfect validation metrics from generation `3` onward. Final test metrics were:

- overall accuracy: `1.0000`
- final-query accuracy: `1.0000`
- final-query exit time: `127.00`
- final-query route match: `1.0000`
- premature-exit rate: `0.0`
- compute: `71.07`
- ES wall-clock time: `334.3s`

Interpretation:

- hybrid ES is still poor as a from-scratch discovery method here
- but once the architecture and initialization are corrected, hybrid ES can be a
  strong local refinement stage instead of a liability

## Current Conclusion

Phase 4 closes with a mixed but scientifically tighter answer.

Primary question, hard-routing robustness:

- no, the current controller redesign does not yet make `delay_to_final_query`
  reliable in a promoted multi-seed hard-ST comparison
- the strongest hard-ST family still averaged only about `0.256` final-query
  accuracy with zero route match across promoted seeds
- so the primary phase-4 exit is still a strong negative result

Secondary question, hybrid ES on the corrected architecture:

- yes, conditionally
- hybrid ES from scratch still collapses during warmstart and does not solve the
  discovery problem
- but resumed hybrid ES from a working controller checkpoint is no longer
  pathological and can strongly improve that checkpoint on this benchmark

The main bottleneck exposed by phase 4 is therefore:

- not benchmark impossibility
- not missing memory capacity
- not inability to represent the control signal
- but instability in making the hard router obey that signal reliably across
  seeds

That points directly to phase 5: stabilize controller discovery first, then
test whether ES polish remains strong across a real multi-seed panel.

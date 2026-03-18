# Phase 5 Report

## Starting Point

Phase 5 started from commit `d26a071216af59c61085fdcdd1227007724f35a4`,
the phase-4 writeups, and the existing phase-5 controller campaign. The key
starting facts were:

- the benchmark was learnable,
- the `needs_final_query` signal was decodable,
- favorable hard-ST seeds existed,
- but promoted hard-ST seed panels collapsed back to early exit,
- and the only convincing ES success was a single resumed polish run from a
  strong checkpoint.

The strongest hard-ST checkpoint at the start of this continuation was the
`seed950` router2/select-exit run:
[summary.json](/home/catid/gnn2/results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1/summary.json)
with independently verified confirmation metrics in
[verification.json](/home/catid/gnn2/results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1/artifacts/phase5_verify/verification.json).
Its full-benchmark confirmation result was still poor on the final-query mode:
overall accuracy `0.5442`, final-query accuracy `0.2436`, final-query route
match `0.0034`, and final-query exit time `41.48`.

The phase-5 experiment ledger is now regenerated from saved artifacts at
[phase5_run_matrix.csv](/home/catid/gnn2/docs/phase5_run_matrix.csv) and covers
30 phase-5 runs.

## What Changed

This continuation added:

- direct release-gate routing support in
  [packet_routing.py](/home/catid/gnn2/src/models/packet_routing.py),
- broader control-target definitions plus the ES-path fix for release-control
  tensors in [run.py](/home/catid/gnn2/src/train/run.py),
- stronger verification and audit use through
  [phase5_verify.py](/home/catid/gnn2/src/utils/phase5_verify.py) and
  [phase5_audit.py](/home/catid/gnn2/src/utils/phase5_audit.py),
- new phase-5 configs under
  [configs/phase5](/home/catid/gnn2/configs/phase5),
- and new ES role-mapping results under
  [results/phase5_dev](/home/catid/gnn2/results/phase5_dev).

## Headline Results

### 1. Medium-basin ES rescue is now confirmed, not anecdotal

Starting from the medium hard-ST checkpoint
[seed950](/home/catid/gnn2/results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1),
the adapter-enabled ES resume run
[seed951_p1](/home/catid/gnn2/results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_p1)
reached perfect test and confirmation performance, and the exact same-seed rerun
[seed951_rerun1](/home/catid/gnn2/results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_rerun1)
matched it exactly.

| Run | Split | Overall Acc | Final-Query Acc | Final-Query Route | Final-Query Exit |
| --- | --- | ---: | ---: | ---: | ---: |
| Hard-ST `seed950` | test | 0.5661 | 0.2527 | 0.0047 | 40.82 |
| Hard-ST `seed950` | confirm | 0.5442 | 0.2436 | 0.0034 | 41.48 |
| ES resume from `seed950` | test | 1.0000 | 1.0000 | 1.0000 | 127.00 |
| ES resume from `seed950` | confirm | 1.0000 | 1.0000 | 1.0000 | 127.00 |
| ES resume rerun from `seed950` | test | 1.0000 | 1.0000 | 1.0000 | 127.00 |
| ES resume rerun from `seed950` | confirm | 1.0000 | 1.0000 | 1.0000 | 127.00 |

This is the strongest verified result of the phase.

### 2. Medium-basin rescue does not need adapters

The router-only counterpart
[routeronly_from950_seed951_p1](/home/catid/gnn2/results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_routeronly_from950_seed951_p1)
lands in essentially the same regime:

- test overall accuracy `0.9995`
- test final-query accuracy `0.9990`
- confirmation overall accuracy `0.9993`
- confirmation final-query accuracy `0.9985`
- route match remains `1.0`
- final-query exit remains `127.0`

So the medium-basin rescue is fundamentally a controller-search result, not an
adapter-only representation rewrite.

### 3. Weak-basin ES can fix routing without fully fixing task quality

Resuming ES from the weaker hard-ST checkpoint
[seed947](/home/catid/gnn2/results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_seed947_p1)
produced a very different pattern:

| Run | Split | Overall Acc | Final-Query Acc | Final-Query Route | Final-Query Exit |
| --- | --- | ---: | ---: | ---: | ---: |
| ES adapter+ from `seed947` | test | 0.9226 | 0.8415 | 1.0000 | 127.00 |
| ES adapter+ from `seed947` | confirm | 0.9131 | 0.8276 | 1.0000 | 127.00 |
| ES router-only from `seed947` | test | 0.8469 | 0.6865 | 1.0000 | 127.00 |
| ES router-only from `seed947` | confirm | 0.8320 | 0.6668 | 1.0000 | 127.00 |

Interpretation:

- ES can still repair the controller policy from this weaker basin.
- The route is perfect: exit timing is correct and premature exit is zero.
- But task accuracy lags because the underlying content representation is weaker.
- In this weak regime, adapters help materially.

This is the cleanest evidence so far that the remaining bottleneck has split in
two:

1. controller discovery,
2. content quality once the controller is already right.

### 4. Hard-ST controller discovery is still not fixed

The direct release-gate family improved exit timing on the full benchmark but
did not recover final-query routing:

- [seed940](/home/catid/gnn2/results/phase5_dev/hard_st_b_v2_waitact_waitloss_controlsetclear_lightaux_waitbias_releasegate_all_seed940_p1):
  final-query accuracy `0.2265`, route match `0.0010`, exit `24.45`
- [seed941](/home/catid/gnn2/results/phase5_dev/hard_st_b_v2_waitact_waitloss_controlsetclear_lightaux_waitbias_releasegate_finalquery_neg8_seed941_p1):
  final-query accuracy `0.2570`, route match `0.0000`, exit `33.87`

The simplified `delay_to_final_query`-only control run
[seed952_p1](/home/catid/gnn2/results/phase5_dev/hard_st_b_v2_waitact_waitloss_controlsetclear_lightaux_waitbias_releasegate_finalquery_neg8_finalqueryonly_seed952_p1)
was retired early. Its logged validation at step `99` stayed at chance:

- overall accuracy `0.2496`
- final-query accuracy `0.2496`
- final-query route match `0.0000`
- final-query exit `19.49`
- premature exit `1.0`

So simplifying the slice was not enough to make this hard-ST controller branch
discover the correct policy.

## Audit Findings

Confirmation-split audits were generated for the medium hard-ST baseline, the
medium ES rescue, and the weak ES rescue:

- [seed950 audit](/home/catid/gnn2/results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1/artifacts/phase5_audit/audit_summary.json)
- [medium ES audit](/home/catid/gnn2/results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_p1/artifacts/phase5_audit/audit_summary.json)
- [weak ES audit](/home/catid/gnn2/results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from947_seed951_p1/artifacts/phase5_audit/audit_summary.json)

The audits reinforce the scalar results:

- Medium hard-ST baseline:
  final-query accuracy `0.2484`, route match `0.0039`, exit `41.9974`.
  Probe decodability of the `needs_final_query` signal is only partial
  (`~0.741` from packet, memory-read, and control state), and the policy still
  fails to act on it.
- Medium ES rescue:
  final-query accuracy `1.0`, route match `1.0`, exit `127.0`.
  Probe decodability rises to `~0.878`, and the policy now obeys the signal.
- Weak ES rescue:
  final-query route is still `1.0`, but task accuracy drops to `0.8296`.
  Probe decodability stays around `~0.878`, which supports the interpretation
  that routing has been fixed while content quality remains limited.

## Scientific Conclusion

The phase-5 answer is stronger than the phase-4 conclusion.

1. Hard-ST discovery is still not robust.
   The direct release-gate family and the simplified final-query-only control
   did not solve the problem.
2. Hybrid ES does more than polish a nearly solved checkpoint.
   It can fully rescue the controller from a medium basin and partially rescue
   it from a weaker one.
3. ES is not just winning through adapter flexibility.
   On the medium basin, router-only ES is essentially as good as router+adapter
   ES.
4. Adapters matter in weaker basins.
   When the controller route can be fixed but the content representation is
   still imperfect, adapter-enabled ES recovers noticeably more task accuracy
   than router-only ES.

So the current best map is:

- hard-ST discovery from scratch: still unstable and not solved
- hybrid ES from a medium checkpoint: strongly positive and now confirmed
- hybrid ES from a weak checkpoint: positive for routing, partial for task
  quality
- hybrid ES from a strong checkpoint: still consistent with phase 4 as a late
  polish stage

## Recommended Next Experiment

The single next experiment I recommend is:

`ES route rescue + short gradient content refinement`

Concretely:

1. start from a weak hard-ST checkpoint such as `seed947`,
2. use ES to recover the controller route to perfect timing,
3. freeze routing and run a short gradient-only content/readout refinement
   phase,
4. measure whether final-query accuracy climbs from the current `0.83` plateau
   toward the medium-basin `1.0` regime.

That is the narrowest next step implied by the current evidence.

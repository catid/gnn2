# Phase 7 Report

## Starting Point

Phase 7 started from the phase-6 map:

- the benchmark was learnable,
- from-scratch hard-ST discovery was still not robust across seeds,
- weak-basin ES rescue plus gradient refinement already worked,
- and once a genuine keepalive basin existed, reopening `memory_` alone was
  enough for near-perfect long-horizon recovery.

The three reproduced anchors at the start of phase 7 were:

| Anchor | Run | Base Verify Overall | Base Verify FQ Acc | Base Verify FQ Route | Base Verify FQ Exit |
| --- | --- | ---: | ---: | ---: | ---: |
| direct keepalive discovery | [seed989_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1) | 0.5540 | 0.3382 | 0.4358 | 92.86 |
| weak-basin staged recovery | [seed973_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1) | 0.9173 | 0.8643 | 0.8155 | 115.62 |
| strong-source memory-only reopen | [seed989 memory-only rerun](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_refine_memoryonly_seed989_rerun1) | 0.9564 | 0.9800 | 0.9719 | 125.76 |

## What Changed In Code

Phase 7 added or materially extended:

- teacher-distillation support in [run.py](/home/catid/gnn2/src/train/run.py),
- partial parameter initialization / transfer support in [run.py](/home/catid/gnn2/src/train/run.py),
- config-level resume override handling in [run.py](/home/catid/gnn2/src/train/run.py),
- stronger phase-7 verification coverage in [phase7_verify.py](/home/catid/gnn2/src/utils/phase7_verify.py),
- integration and unit coverage in
  [test_integration_smoke.py](/home/catid/gnn2/tests/test_integration_smoke.py),
  [test_phase7_verify.py](/home/catid/gnn2/tests/test_phase7_verify.py),
  [test_resume_overrides.py](/home/catid/gnn2/tests/test_resume_overrides.py), and
  [test_teacher_distillation.py](/home/catid/gnn2/tests/test_teacher_distillation.py).

## Coverage And Budget

The regenerated ledger at [phase7_run_matrix.csv](/home/catid/gnn2/docs/phase7_run_matrix.csv) now covers 148 substantive phase-7 entries:

- 72 fruitful
- 76 exploration
- 40 reruns
- 18 seed-panel rows
- 17 promoted rows

That satisfies the intended 50/50 split in the final accounting. The earlier
live imbalance came from transfer / confirmation / partial-init work being
logged under the recovery buckets instead of the exploration side.

## Headline Findings

### 1. From-scratch hard-ST discovery is still not robust

Phase 7 did not produce a new five-seed, locked-confirmed direct-discovery win.
The best direct-discovery checkpoint is still the reproduced phase-6 keepalive
anchor:

- [seed989_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1):
  `overall 0.5540`, `fq_acc 0.3382`, `fq_route 0.4358`, `fq_exit 92.86`

The new direct-discovery families stayed weak:

- keepalive shaping variants such as
  [releaseaux](/home/catid/gnn2/results/phase7_dev/hard_st_b_v2_controlsticky_keepalive_releaseaux_seed1005_p1)
  and
  [longerstrong rerun](/home/catid/gnn2/results/phase7_dev/hard_st_b_v2_controlsticky_keepalive_longerstrong_seed1006_rerun1)
  stayed in the early-exit regime,
- controller-family alternatives such as
  [monotone wait rerun](/home/catid/gnn2/results/phase7_dev/hard_st_b_v2_monotone_wait_direct_seed1310_rerun1)
  and the reproduced setclear family never achieved meaningful final-query
  route match,
- and the last stable-transfer scouts
  [coremem stable](/home/catid/gnn2/results/phase7_dev/20260320_171224_hard_st_benchmark_b_v2_controlsticky_keepalive_partialinit_coremem_stable)
  and
  [coremem sinkreadout stable](/home/catid/gnn2/results/phase7_dev/20260320_171224_hard_st_benchmark_b_v2_controlsticky_keepalive_partialinit_coremem_sinkreadout_stable)
  only showed brief transient route structure before drifting back to earlier
  exits.

| Run | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| `coremem_stable` | 0.4951 | 0.2614 | 0.0963 | 54.90 |
| `coremem_sinkreadout_stable` | 0.5462 | 0.2453 | 0.0100 | 39.35 |

Interpretation: the keepalive basin still exists, but phase-7 direct training
did not make entry into that basin reliable.

### 2. Medium-source staged recovery is now systematic

The strongest new recovery result is the medium-source
`forceoracle -> memoryreadout` family. After filling the verification gap on the
entire five-seed panel, the verified base-test mean is:

| Panel | Overall | Route | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `forceoracle_release_longerstrong_refine_memoryreadout` 5 seeds | 0.4400 +/- 0.0202 | 0.4821 +/- 0.0082 | 0.2988 +/- 0.0061 | 0.5702 +/- 0.0142 | 96.46 +/- 1.71 |

This is not a high-accuracy solution, but it is a genuine, reproducible
late-route source instead of a single-seed curiosity. That turns the
medium-source recovery story from anecdote into a stable operating regime.

The weak-basin recovery anchor remained strong:

- [seed973_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1):
  `overall 0.9173`, `fq_acc 0.8643`, `fq_route 0.8155`, `fq_exit 115.62`

So the phase-7 recovery map is now:

- weak sources can still be rescued strongly,
- medium sources can be generated reproducibly,
- and the remaining gap is content quality, not route existence.

### 3. Strong-source transfer is route-positive but still content-limited

The partial-init transfer family was the most informative exploration cluster.
The cross-polish branch
[sinkonlylonger->sinkreadout](/home/catid/gnn2/results/phase7_dev/20260320_163404_hard_st_benchmark_b_v2_controlsticky_keepalive_partialinit_coremem_sinkonlylonger_to_sinkreadout_seed1735)
ended up being the clearest result.

Verified five-seed base panel:

| Panel | Overall | Route | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sinkonlylonger -> sinkreadout` 5 seeds | 0.5143 +/- 0.0077 | 0.4542 +/- 0.0058 | 0.2521 +/- 0.0141 | 0.2460 +/- 0.0071 | 86.21 +/- 1.21 |

Verified locked confirmations on the same five seeds:

| Locked Split | Overall | Route | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_locked` | 0.6316 +/- 0.0010 | 0.9922 +/- 0.0000 | 0.2451 +/- 0.0020 | 0.9866 +/- 0.0000 | 126.20 +/- 0.00 |
| `finalquery_heavy` | 0.4133 +/- 0.0024 | 0.9902 +/- 0.0000 | 0.2609 +/- 0.0031 | 0.9885 +/- 0.0000 | 126.36 +/- 0.00 |

This is the clearest route-versus-content separation in the repo:

- route transfer is almost perfect under locked confirmations,
- exit timing is almost perfect,
- but final-query task accuracy stays near chance.

Phase 7 therefore localized the remaining failure more sharply than phase 6:
for these strong-source transfer branches, the missing ingredient is no longer
route retention. It is content recovery after route transfer.

### 4. ES now has a much clearer role map

#### Keepalive anchor: adapters are essential

The strongest confirmed ES result in phase 7 is still the keepalive-anchor
adapter run:

- [keepalive adapter ES](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1):
  base verify `0.9756 / 0.9500 / 1.0000 / 127.00`
  for `overall / fq_acc / fq_route / fq_exit`
- locked confirms stay near perfect:
  `0.9935 / 0.9900 / 0.9873 / 126.53`
  and
  `0.9906 / 0.9889 / 0.9869 / 126.36`

The source-matched router-only comparison collapses completely:

- [keepalive router-only ES](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_routeronly_from989_seed1202_p1):
  base verify `0.2561 / 0.2655 / 0.0000 / 0.00`

So on the strongest direct-discovery source, router-only ES is not enough.

#### Medium-source branch: router-only buys accuracy, adapters preserve routing

On the stable medium `forceoracle` branch, the tradeoff flips:

| Run | Base Overall | Base FQ Acc | Base FQ Route | Base FQ Exit | Locked FQ Route | Locked FQ Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [adapter ES from 1305](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_forceoracle_longerstrong_resume_from1305_seed1604_p1) | 0.4602 | 0.3380 | 0.9980 | 126.76 | ~0.638 | ~105 |
| [router-only ES from 1305](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_forceoracle_longerstrong_resume_routeronly_from1305_seed1604_p1) | 0.6467 | 0.2820 | 0.6265 | 92.36 | ~0.150 | ~54 |

The same source-matched tradeoff held on the teacher-shaped sources `1702`,
`1703`, `1705`, and `1706`:

- router-only ES usually buys more base overall accuracy,
- adapter ES preserves later exits and better final-query route fidelity,
- and the advantage of adapters grows as the source gets weaker.

This is the clearest new ES result of phase 7. ES is not just “good” or “bad.”
Its best role depends on what the source checkpoint already contains:

- keepalive anchor: adapters are necessary
- stable medium sources: router-only can inflate base accuracy, but adapters are
  better if route fidelity on held confirmations matters
- from scratch: still negative

### 5. Exploration clusters were fair and mostly negative

Cluster D, E, and the non-teacher parts of F all received real tuning and
reruns, and the negative story is stable:

- REINFORCE discovery did not solve the problem,
- monotone / setclear / waitstate alternatives did not solve the problem,
- wait-loss, exitmask, and related supervision tweaks did not solve the
  problem,
- and direct keepalive discovery remained seed-sensitive even after a broader
  tuning pass.

The useful exploration positives were narrower:

- force-oracle imitation-release is a real medium-source generator,
- teacher-shaped sources are real ES inputs,
- and strong-source transfer under locked confirmations revealed that route
  transfer is much easier than content transfer.

## Comparison To Phase 6

Relative to the phase-6 slice, phase 7 did **not** deliver a robust new
from-scratch discovery pipeline. But it did produce a much stronger map:

1. The discovery failure is now sharply localized.
   It is not “the model cannot route late” and not “ES is the only reason
   routing ever works.” It is “from-scratch hard-ST still does not reliably
   enter the keepalive basin.”
2. Recovery is more systematic.
   Medium-source late-route checkpoints can now be produced reproducibly.
3. ES has a source-dependent role map.
   Adapter ES is the route-faithful branch; router-only ES is the
   base-accuracy branch; and the choice depends on source quality.
4. Transfer generalization is clearer.
   Route transfer under strong-source partial init is easy; content repair is
   the slower, harder part.

## Direct Answers To The Phase-7 Questions

### Can from-scratch hard-routing discovery be made robust across seeds?

Not yet. Phase 7 did not produce a five-seed, confirmation-clean direct
discovery improvement over the phase-6 keepalive anchor.

### Can staged recovery be made more systematic and predictable?

Yes, partially. Weak-basin recovery remains strong, and medium-source
`forceoracle -> memoryreadout` recovery is now reproducible across a verified
five-seed panel. Strong-source transfer is also systematic, but it exposes a
content bottleneck after route transfer.

### Where does hybrid EGGROLL-inspired ES help best now?

The best current answer is:

- not from scratch,
- strongly as route-faithful rescue from the right source checkpoints,
- sometimes as base-accuracy rescue in router-only form on medium sources,
- and with a source-dependent adapter-vs-router-only tradeoff.

## Exit Condition

Phase 7 reaches a **strong mapping exit**, not a positive discovery exit.

Why:

- the 50/50 fruitful/exploration budget is honored in the final ledger,
- all required clusters were explored,
- the major positives and negatives have verify artifacts,
- the strongest remaining bottleneck is more localized than in phase 6,
- and the next experiment is narrower and higher-confidence.

## Recommended Next Experiment

Use the verified route-faithful adapter-ES checkpoint as a fixed teacher for a
direct keepalive model, but distill only the wait / release / control-state
channels into the student before releasing it back to task-only hard routing.

Reason:

- direct keepalive tuning alone did not widen the basin,
- plain teacher shaping produced good ES source checkpoints but not a direct
  win,
- the strongest phase-7 evidence says the missing step is direct basin entry
  into the already-known route-faithful regime.

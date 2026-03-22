# Phase 8 Report

## Starting Point

Phase 8 started from the verified phase-7 map:

| Anchor | Run | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | --- | ---: | ---: | ---: | ---: |
| direct discovery | [seed989_rerun1](/home/catid/gnn2/results/phase8_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1) | 0.5540 | 0.3382 | 0.4358 | 92.86 |
| weak-basin rescue | [seed973_rerun1](/home/catid/gnn2/results/phase8_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1) | 0.9173 | 0.8643 | 0.8155 | 115.62 |
| strong-source memory-only reopen | [seed989 memory-only rerun](/home/catid/gnn2/results/phase8_anchor/hard_st_b_v2_controlsticky_keepalive_refine_memoryonly_seed989_rerun1) | 0.9564 | 0.9800 | 0.9719 | 125.76 |
| keepalive-anchor ES | [seed1201](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1) | 0.9756 | 0.9500 | 1.0000 | 127.00 |

The phase-7 conclusion was already sharp:

- route-faithful late-routing basins exist,
- teacher-shaped and ES-shaped source checkpoints can preserve them,
- but from-scratch hard-ST still does not enter that basin robustly,
- and once route retention is solved, content quality becomes the next bottleneck.

Phase 8 targeted that exact gap with teacher-seeded direct basin entry.

## What Changed In Code

Phase 8 added or materially extended:

- teacher supervision schedules, release windows, and channel dropout in [run.py](/home/catid/gnn2/src/train/run.py),
- detached-prefix and late-window truncation controls in [packet_routing.py](/home/catid/gnn2/src/models/packet_routing.py),
- phase-8 independent recomputation and confirm-label handling in [phase8_verify.py](/home/catid/gnn2/src/utils/phase8_verify.py),
- phase-8 launcher scripts in
  [run_phase8_main.sh](/home/catid/gnn2/scripts/run_phase8_main.sh),
  [run_phase8_confirm.sh](/home/catid/gnn2/scripts/run_phase8_confirm.sh),
  [run_phase8_seed_panels.sh](/home/catid/gnn2/scripts/run_phase8_seed_panels.sh),
  [run_phase8_cluster_scouts.sh](/home/catid/gnn2/scripts/run_phase8_cluster_scouts.sh), and
  [run_phase8_teacher_sweeps.sh](/home/catid/gnn2/scripts/run_phase8_teacher_sweeps.sh),
- phase-8 coverage in
  [test_integration_smoke.py](/home/catid/gnn2/tests/test_integration_smoke.py),
  [test_phase8_verify.py](/home/catid/gnn2/tests/test_phase8_verify.py),
  [test_resume_overrides.py](/home/catid/gnn2/tests/test_resume_overrides.py), and
  [test_teacher_distillation.py](/home/catid/gnn2/tests/test_teacher_distillation.py).

## Coverage And Budget

The final ledger in [phase8_run_matrix.csv](/home/catid/gnn2/docs/phase8_run_matrix.csv) covers 116 phase-8 entries:

- 68 fruitful
- 44 exploration
- 4 anchor reproductions
- 10 explicit reruns
- 12 seed-panel rows, including 3 aggregate five-seed panel summaries
- 23 confirmation rows

Excluding the anchor reproductions, the campaign finished at `68 / 44`, which is
effectively the requested 60 / 40 split.

Phase 8 therefore exits as a **strong mapping result**:

- it did not produce a robust teacher-free positive exit across five seeds and locked confirms,
- but it did identify the exact teacher/channel/release recipe that reliably creates route-faithful teacher-free basin entry,
- and it localized the remaining failure more sharply than phase 7.

## Headline Findings

### 1. Teacher-seeded basin entry is real, teacher-free, and multi-seed on routing

The strongest new phase-8 direct-entry result is the strong-teacher
wait/release-only longrelease plus delayed-dropout recipe:

- [seed1874_p1](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_p1)
- exact rerun:
  [seed1874_rerun1](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_rerun1)

Best verified teacher-free checkpoint on the original seed:

- base `overall 0.7907`, `fq_acc 0.5702`, `fq_route 0.9499`, `fq_exit 122.17`
- full locked `0.6510 / 0.2841 / 0.8850 / 116.22`
- finalquery-heavy `0.4401 / 0.2939 / 0.8789 / 115.56`
- longdistance `0.5257 / 0.3298 / 0.8868 / 145.62`

The exact rerun matched the original trajectory checkpoint-for-checkpoint through
the breakthrough region, so the single-seed result is real.

The promoted five-seed panel is now summarized directly in the ledger at
[panel row](/home/catid/gnn2/docs/phase8_run_matrix.csv):

- [panel 1874/1878/1879/1880/1881](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_p1)

Five-seed verify means:

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.4410 | 0.3243 | 0.7326 | 122.74 |
| full_locked | 0.4065 | 0.2615 | 0.5225 | 120.73 |
| finalquery_heavy | 0.3189 | 0.2603 | 0.5252 | 120.96 |
| longdistance | 0.3400 | 0.2574 | 0.5185 | 150.98 |

This is the clearest phase-8 answer to the primary question:

- teacher-guided wait/release-only supervision can create a real teacher-free
  basin-entry effect,
- the effect survives exact rerun and multiple neighboring seeds on route
  fidelity and exit timing,
- but final-query accuracy is still not robust across the full panel.

Relative to the old direct-discovery anchor, the new recipe clearly improves
route match and exit time, but it does **not** robustly improve content quality
across five seeds. Phase 8 therefore did not reach a positive exit.

### 2. The teacher requirements are now sharp

The source/channel/release mapping is no longer vague.

What mattered:

- a **strong keepalive-like teacher**
- **wait/release-only** supervision
- a **long release**
- **delayed teacher dropout**

What did not help:

- adding control-state supervision
- using a medium teacher plus full-route upper-bound imitation
- using a weaker transfer teacher with the same wait/release channels

The strongest medium-teacher route-only result is the selectroute family,
summarized in the new five-seed ledger row:

- [panel 1826/1827/1828/1829/1830](/home/catid/gnn2/docs/phase8_run_matrix.csv)

Base five-seed mean:

- `overall 0.2508`, `fq_acc 0.2519`, `fq_route 0.8765`, `fq_exit 125.14`

So medium wait/release-only supervision is already enough to produce a genuine
teacher-free late-routing basin, but not enough to recover content.

The medium `selectacc` source
[seed1821_rerun1](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_medium_waitrelease_only_longrelease_selectacc_seed1821_rerun1)
is the key bridge result:

- base `0.6266 / 0.2807 / 0.9198 / 121.99`
- full_locked `0.6149 / 0.2560 / 0.8329 / 116.34`
- finalquery-heavy `0.4040 / 0.2607 / 0.8264 / 115.59`

Interpretation:

- strong teacher + wait/release only is the best basin-entry recipe,
- medium teacher + wait/release only is enough to create a useful route-faithful
  source,
- control-state is still harmful,
- and “more teacher” via full-route imitation was not the missing ingredient.

### 3. The best new staged-recovery pipeline is systematic, but still content-limited

The strongest new systematic recovery result is the `1821 -> memoryreadout`
longer low-LR family:

- [panel row](/home/catid/gnn2/docs/phase8_run_matrix.csv)
- representative seed:
  [seed1842_p1](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_teacher1821_refine_memoryreadout_longer_lowlr_seed1842_p1)
- exact rerun:
  [seed1842_rerun1](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_teacher1821_refine_memoryreadout_longer_lowlr_seed1842_rerun1)

Five-seed confirm means:

| Eval | Overall | FQ Acc | FQ Route | FQ Exit |
| --- | ---: | ---: | ---: | ---: |
| base | 0.6459 | 0.3085 | 0.9118 | 122.11 |
| full_locked | 0.5939 | 0.2578 | 0.8085 | 114.38 |
| finalquery_heavy | 0.3826 | 0.2416 | 0.8106 | 114.58 |
| longdistance | 0.4475 | 0.2391 | 0.7686 | 140.44 |

This is phase 8’s best new staged-recovery pipeline. It is clearly stronger than
the unrepaired `1821` source and it is stable across a real five-seed panel.

The absolute best staged-recovery result in the repo is still the reproduced
weak-basin anchor:

- [seed973_rerun1](/home/catid/gnn2/results/phase8_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1):
  `0.9173 / 0.8643 / 0.8155 / 115.62`

So phase 8 did not replace the weak-basin rescue champion, but it did produce
the best new **teacher-seeded** recovery pipeline.

### 4. The post-entry failure boundary is now exact: memory destabilizes fragile basins

The sharpest “why not?” result from phase 8 came from the fragile `1879`
teacher-free basin.

Memory-based reopenings destroyed the basin:

- [memory-only 1884](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_teacher1879_refine_memoryonly_longer_lowlr_seed1884_p1)
- [memory+readout 1885](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_teacher1879_refine_memoryreadout_longer_lowlr_seed1885_p1)

Both collapsed route fidelity.

Head-only reopenings preserved it:

- [readout-only 1886](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_teacher1879_refine_readoutonly_longer_lowlr_seed1886_p1):
  base `0.2497 / 0.2540 / 0.9004 / 123.00`
- [sink+readout 1887](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_teacher1879_refine_sinkreadout_longer_lowlr_seed1887_p1):
  base `0.2646 / 0.2654 / 0.8930 / 123.00`

Locked confirms on both stayed nearly route-perfect and late-exiting, but
final-query accuracy stayed near chance.

That is the single clearest boundary phase 8 established:

- teacher-seeded route-faithful basin entry is real,
- route retention is no longer the missing ingredient,
- and reopening `memory_` on fragile teacher-free basins is specifically what
  destabilizes them.

### 5. ES did not become a new basin-entry answer in phase 8

Phase-8 ES mapping was narrower than the phase-7 keepalive-anchor story.

Representative results:

- [adapter ES from raw 1821](/home/catid/gnn2/results/phase8_dev/hybrid_es_b_v2_teacher1821_resume_from1821_seed1851_p1):
  route-perfect but content-flat
- [router-only ES from raw 1821](/home/catid/gnn2/results/phase8_dev/hybrid_es_b_v2_teacher1821_resume_routeronly_from1821_seed1851_p1):
  similar base behavior, slightly weaker confirms
- [adapter ES from 1842](/home/catid/gnn2/results/phase8_dev/hybrid_es_b_v2_teacher1842_resume_from1842_seed1852_p1):
  negative
- [router-only ES from 1842](/home/catid/gnn2/results/phase8_dev/hybrid_es_b_v2_teacher1842_resume_routeronly_from1842_seed1852_p1):
  worse negative

So the phase-8 answer is:

- ES can preserve already-discovered teacher-shaped route structure,
- but it did not create a new robust content-recovery or basin-entry win here,
- and the best confirmed ES-assisted result is still the phase-7 keepalive-anchor
  adapter branch [seed1201](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1).

### 6. Exploration clusters were fair and mostly negative

The exploration half did real work and mostly closed branches cleanly:

- detached warmup / late-window discovery stayed route-dead, for example
  [seed1904](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_trunc32_detachprefix96_seed1904_p1)
  and
  [seed1905](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_detachprefix64_latewindow32_seed1905_p1)
- alternate controller families stayed route-dead, for example
  [seed1911](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_monotone_wait_direct_phase8_seed1911_p1),
  [seed1912](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsetclear_memoryheavy_exitselect_phase8_seed1912_p1), and
  [seed1913](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_recurrent_waitact_curriculum_phase8_seed1913_p1)
- REINFORCE stayed in the same immediate-exit basin across all serious variants,
  for example
  [seed1921](/home/catid/gnn2/results/phase8_dev/reinforce_b_v2_controlsticky_keepalive_base_phase8_seed1921_p1) and
  [seed1924](/home/catid/gnn2/results/phase8_dev/reinforce_b_v2_controlsticky_keepalive_exitmask_controlaux_phase8_seed1924_p1)

These negatives matter because they isolate what did work:

- not generic exploration noise,
- not different controller shape alone,
- not policy gradient,
- but specifically narrow teacher-guided wait/release shaping.

## Comparison To Phase 7

Relative to the phase-7 map, phase 8 delivered three real advances.

1. Teacher-guided basin entry is now a verified phenomenon, not just a next-step idea.
2. The teacher requirements are now exact: strong teacher, wait/release only, long release, delayed dropout.
3. The post-entry bottleneck is now localized much more sharply: fragile basins lose route when `memory_` is reopened, while head-only reopens preserve route but do not recover content.

What phase 8 did **not** deliver is equally clear:

- no robust teacher-free five-seed content win over the old direct-discovery anchor,
- no new ES-assisted result better than the old keepalive-anchor ES,
- and no exploration-family surprise that overturns the keepalive teacher story.

## Direct Answers To The Phase-8 Questions

### Can teacher-seeded control supervision create a robust, teacher-free basin-entry recipe?

Partially.

- Yes for route fidelity and exit timing.
- Not yet for robust content quality across five seeds and held confirms.

### Once the student enters a keepalive-like basin, what is the smallest reliable refinement?

For the stable medium-source teacher branch, `memory + readout` with a longer
low-LR reopen is the best systematic answer.

For fragile direct-entry basins, the smallest safe reopen is **head-only**.
Reopening `memory_` destabilizes the basin before it fixes content.

### How much teacher strength, channel selection, release schedule, and ES help are necessary?

The best current answer is:

- teacher strength matters a lot,
- wait/release-only supervision matters more than control-state,
- long release plus delayed dropout matters,
- and ES is still mostly downstream of basin entry, not the cause of it.

## Exit Type

Phase 8 exits as a **strong mapping result**.

It did not satisfy the positive-exit threshold because the best five-seed
teacher-free basin-entry panel still has weak final-query accuracy under locked
confirms. But it did answer the sharper scientific question:

- teacher-guided control supervision can create teacher-free route-faithful
  basin entry,
- and the exact remaining failure is content learning after entry, especially
  once `memory_` is reopened.

## Handoff

- Best confirmed teacher-free basin-entry result:
  [seed1874_p1](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_p1)
  with exact rerun
  [seed1874_rerun1](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_rerun1).
- Best confirmed staged-recovery result:
  absolute best remains
  [seed973_rerun1](/home/catid/gnn2/results/phase8_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1);
  best new teacher-seeded recovery panel is
  [teacher1821->memoryreadout panel](/home/catid/gnn2/docs/phase8_run_matrix.csv).
- Best confirmed ES-assisted result:
  [seed1201](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1)
  still leads.
- Teacher channels that mattered:
  wait/release only. Control-state supervision was harmful in the teacher-seeded
  basin-entry setting.
- What remains unstable:
  robust content learning after route-faithful teacher-free basin entry.
- Next recommended experiment:
  teacher-seeded wait/release-only basin entry followed by **memory-frozen
  head-only content shaping**.

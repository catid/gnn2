# Phase 8 Cluster Scorecards

Phase-8 ledger totals from [phase8_run_matrix.csv](/home/catid/gnn2/docs/phase8_run_matrix.csv):

- 116 total entries
- 68 fruitful, 44 exploration, 4 anchor reproductions
- 10 explicit reruns
- 12 seed-panel rows
- 23 confirmation rows

Excluding anchors, the final split is effectively the requested 60 / 40 budget.

## Scorecards

| Cluster | Half | Runs | Best Result | Verdict |
| --- | --- | ---: | --- | --- |
| A: teacher-seeded direct basin entry | fruitful | 16 | [medium-teacher selectroute panel](/home/catid/gnn2/docs/phase8_run_matrix.csv) base mean `0.2508 / 0.2519 / 0.8765 / 125.14` for overall / fq_acc / fq_route / fq_exit | Positive on route entry, negative on content. This cluster proved teacher-free basin entry is real. |
| B: teacher source / channel / release mapping | fruitful | 17 | [strong-teacher 1874](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_p1) base verify `0.7907 / 0.5702 / 0.9499 / 122.17`; five-seed panel mean `0.4410 / 0.3243 / 0.7326 / 122.74` | Strong positive mapping. Strong keepalive teacher plus wait/release-only supervision mattered; control-state and weaker teachers did not. |
| C: post-entry content recovery | fruitful | 29 | [teacher1821->memoryreadout panel](/home/catid/gnn2/docs/phase8_run_matrix.csv) base mean `0.6459 / 0.3085 / 0.9118 / 122.11`; full-locked `0.5939 / 0.2578 / 0.8085 / 114.38` | Positive but still content-limited. Best new systematic staged-recovery pipeline of phase 8. |
| D: ES on teacher-seeded and recovery sources | fruitful | 6 | [adapter ES from raw 1821](/home/catid/gnn2/results/phase8_dev/hybrid_es_b_v2_teacher1821_resume_from1821_seed1851_p1) preserved route perfectly but kept `fq_acc` near chance | Retired as a basin-entry/content fix. ES preserved route on teacher-shaped sources but did not create a new recovery win here. |
| E: detached-warmup / terminal-only discovery | exploration | 8 | Best scout, [seed1905](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_detachprefix64_latewindow32_seed1905_p1), still verified at `0.2552 / 0.2654 / 0.0000 / 38.87` | Retired as a fair negative. Later exits did not turn into final-query routing. |
| F: alternate control architectures | exploration | 8 | [monotone wait](/home/catid/gnn2/results/phase8_dev/hard_st_b_v2_monotone_wait_direct_phase8_seed1911_p1) delayed longest, but still verified `fq_route = 0.0` | Retired as stable negatives. These architectures changed timing more than discovery success. |
| G: alternative discovery algorithms | exploration | 5 | Best REINFORCE branch, [seed1921](/home/catid/gnn2/results/phase8_dev/reinforce_b_v2_controlsticky_keepalive_base_phase8_seed1921_p1), still verified `fq_route = 0.0`, `fq_exit = 0.0` | Retired as a direct-discovery fix. Policy-gradient discovery stayed in the immediate-exit basin. |
| H: stress / confirmation / generalization | exploration | 23 | The strong-teacher `1874` panel and the `1821 -> memoryreadout` recovery panel both held real locked-confirm route structure | Positive. This cluster made the route-vs-content split much harder to deny. |

## Cluster-Level Verdicts

- A succeeded at proving teacher-free route basin entry.
- B identified the exact teacher source/channel/release recipe that matters.
- C produced the best new systematic recovery pipeline, but content is still the limiter.
- D showed ES is still downstream of source quality, not the missing entry mechanism.
- E, F, and G were fair negatives.
- H confirmed that the strongest phase-8 results are real on held settings, but mostly on route fidelity rather than content.

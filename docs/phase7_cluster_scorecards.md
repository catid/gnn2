# Phase 7 Cluster Scorecards

Phase-7 ledger totals from [phase7_run_matrix.csv](/home/catid/gnn2/docs/phase7_run_matrix.csv):

- 148 substantive entries
- 72 fruitful and 76 exploration
- 40 exact reruns
- 18 seed-panel rows
- 17 promoted rows

## Scorecards

| Cluster | Half | Runs | Best Result | Verdict |
| --- | --- | ---: | --- | --- |
| A: direct keepalive discovery | fruitful | 12 | Phase-6 anchor reproduction still leads: [seed989_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1) base verify `0.5540 / 0.3382 / 0.4358 / 92.86` for overall / fq_acc / fq_route / fq_exit | Retired as unresolved. Phase-7 keepalive tuning did not widen the basin across seeds. |
| B: staged recovery pipelines | fruitful | 25 | Weak-basin anchor [seed973_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1) stays strong at `0.9173 / 0.8643 / 0.8155 / 115.62`; medium-source five-seed branch [forceoracle->memoryreadout](/home/catid/gnn2/results/phase7_dev/hard_st_b_v2_forceoracle_release_longerstrong_refine_memoryreadout_seed1305_p1) verified mean `0.4400 / 0.2988 / 0.5702 / 96.46` | Positive but content-limited. Recovery is systematic once real late-route structure exists. |
| C: ES-assisted rescue mapping | fruitful | 32 | Keepalive-anchor adapter ES [seed1201_p1](/home/catid/gnn2/results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1) verified `0.9756 / 0.9500 / 1.0000 / 127.00` with near-perfect locked confirms | Strong positive. ES is highly useful after basin entry, but source-dependent and not a from-scratch discovery method. |
| D: alternative discovery algorithms | exploration | 10 | Best non-hard-ST discovery family is force-oracle imitation-release; REINFORCE branch stayed near immediate exit, e.g. [entropyhigh rerun](/home/catid/gnn2/results/phase7_dev/reinforce_b_v2_controlsticky_keepalive_entropyhigh_seed1402_rerun1) | Retired as a direct discovery fix. Force-oracle can seed usable medium sources, policy-gradient discovery did not. |
| E: different controller / memory families | exploration | 12 | All promoted setclear / monotone / waitstate variants stayed weak. Example: [monotone rerun](/home/catid/gnn2/results/phase7_dev/hard_st_b_v2_monotone_wait_direct_seed1310_rerun1) `0.2446 / 0.2450 / 0.0000 / 51.58` | Retired as stable negatives. These families delay more sometimes but do not reliably discover final-query routing. |
| F: supervision / curriculum / pretraining | exploration | 8 | Teacher-shaped medium sources are useful, e.g. [teacher finalquery state 1705](/home/catid/gnn2/results/phase7_dev/hard_st_b_v2_forceoracle_release_longerstrong_refine_memoryreadout_teacherdistill_finalquery_state_seed1705_p1); controlwaitact / waitloss / exitmask lines stayed negative | Mixed. Extra supervision can create better ES source checkpoints, but not robust direct discovery. |
| G: transfer / generalization / confirmation stress | exploration | 46 | Cross-polished partial-init branch [sinkonlylonger->sinkreadout](/home/catid/gnn2/results/phase7_dev/20260320_163404_hard_st_benchmark_b_v2_controlsticky_keepalive_partialinit_coremem_sinkonlylonger_to_sinkreadout_seed1735) has five-seed base mean `0.5143 / 0.2521 / 0.2460 / 86.21` and locked means `fq_route ~= 0.987`, `fq_exit ~= 126.2` | Positive for route transfer, negative for content transfer. Routing generalizes under locked confirms long before content quality does. |

## Cluster-Level Verdicts

- A failed to produce a robust new from-scratch basin.
- B succeeded at making recovery more systematic, especially from medium sources.
- C produced the strongest verified positive of phase 7.
- D showed that non-hard-ST discovery ideas did not outperform the structured force-oracle branch.
- E was a fair but clear negative.
- F mattered mainly as teacher/source shaping, not as a direct discovery fix.
- G clarified that route transfer is easier than content transfer.

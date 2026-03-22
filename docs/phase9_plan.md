# Phase 9 Plan

## Starting Point

Phase 8 established the new boundary:

- Teacher-guided wait/release-only supervision can create a real teacher-free, route-faithful basin-entry effect.
- Route retention is no longer the missing ingredient on the best fragile teacher-free sources.
- Content recovery is still the bottleneck.
- Reopening `memory_` on fragile teacher-free basins destroys the basin.
- Head-only reopenings preserve route and late exits but leave final-query accuracy near chance.

Key inherited anchors:

- Best direct teacher-seeded basin-entry anchor:
  `results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_p1`
  Base verify: `overall 0.7907`, `fq_acc 0.5702`, `fq_route 0.9499`, `fq_exit 122.17`
- Fragile direct-entry route-faithful boundary:
  `results/phase8_dev/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1879_p1`
  Base verify: `overall 0.2679`, `fq_acc 0.2640`, `fq_route 0.9078`, `fq_exit 123.63`
- Fragile head-only reopen boundaries:
  `results/phase8_dev/hard_st_b_v2_teacher1879_refine_readoutonly_longer_lowlr_seed1886_p1`
  `results/phase8_dev/hard_st_b_v2_teacher1879_refine_sinkreadout_longer_lowlr_seed1887_p1`
- Best systematic teacher-seeded recovery family:
  `results/phase8_dev/hard_st_b_v2_teacher1821_refine_memoryreadout_longer_lowlr_seed1842_p1`
  5-seed means:
  base `overall 0.6459`, `fq_acc 0.3085`, `fq_route 0.9118`, `fq_exit 122.11`
  full_locked `overall 0.5939`, `fq_acc 0.2578`, `fq_route 0.8085`, `fq_exit 114.38`
- Best ES-assisted anchor remains:
  `results/phase8_anchor/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1`
  Base verify: `overall 0.9756`, `fq_acc 0.9500`, `fq_route 1.0000`, `fq_exit 127.00`

## Phase 9 Question

Primary:

- Can memory-frozen head-only content shaping convert teacher-free route-faithful basin entry into robust content recovery?

Secondary:

- Does the frozen route-faithful state already contain enough answer-relevant content to support head-only recovery?

Tertiary:

- If not, what is the smallest safe partial unfreeze or adapter that improves content while preserving late-route fidelity?

## Anchor Reproductions

These must be reproduced before broad phase-9 claims:

1. `1874` direct teacher-seeded basin-entry anchor
2. `1879` fragile route-faithful source
3. At least one fragile head-only reopen anchor: `1886` or `1887`
4. Representative `1821 -> memoryreadout longer lowlr`
5. Phase-7/8 ES anchor `1201`

## Budget Split

Phase 9 uses a 70 / 30 run-budget split.

- Fruitful, ~70%
  - Cluster A: frozen-state content audit and decodability map
  - Cluster B: strict head-only shaping on fragile direct-entry basins
  - Cluster C: strict head-only shaping on medium teacher-shaped sources
  - Cluster D: teacher content distillation and content transfer
  - Cluster E: minimal-safe partial unfreeze, only if gated by A-D
- Exploration, ~30%
  - Cluster F: alternative content readers / heads
  - Cluster G: head-only ES / black-box polish
  - Cluster H: stress / confirmation / generalization

Planned substantive run budget:

- Cluster A: 14
- Cluster B: 24
- Cluster C: 18
- Cluster D: 10
- Cluster E: 8, gated
- Cluster F: 8
- Cluster G: 6
- Cluster H: 10

Target total: 98 substantive entries, with a hard floor of 72.

## Cluster Definitions

### Cluster A

- Sources: `1879`, `1874`, `1821`, strong upper bound `1201` or strong recovered anchor
- Probe families:
  - linear probe on frozen final-query features
  - shallow MLP probe
  - query-conditioned linear probe
  - query-conditioned MLP probe
- Representation families:
  - `packet_state`
  - `memory_read_state`
  - `router_logits`
  - `router_probs`
  - `control_state`
  - `wait_state`
  - `sink_state`
  - final `sink_state`
- Required decision:
  - go / no-go for pure head-only shaping

### Cluster B

- Strictly frozen `memory_`, router, control, wait, release on fragile direct-entry family
- Trainable head families:
  - `readout`
  - `sink_proj + readout`
  - query-conditioned readout
  - query-conditioned `sink_proj + readout`
  - head-only + final-query-weighted CE
  - head-only + teacher logits distillation only

### Cluster C

- Medium teacher-shaped family rooted at `1821`
- Head-only comparisons mirror Cluster B, but with stable medium source instead of fragile direct-entry source

### Cluster D

- Content distillation only
- Teacher route/control weights remain zero
- Compare logits distillation against label-only head shaping on frozen sources

### Cluster E

- Gated cluster
- Enter only if Cluster A says content is not sufficiently decodable under frozen heads or if B/C saturate
- Allowed:
  - tiny low-rank adapter on memory read path
  - tiny adapter on sink input only
  - bias/LN-only updates adjacent to head path
  - route-preservation penalties and rollback

### Cluster F

- Alternative frozen-head readers:
  - query-conditioned bilinear style
  - gated reader
  - tiny iterative / recurrent head
  - mixture-of-heads if justified

### Cluster G

- Narrow ES question on head-only spaces:
  - readout only
  - `sink_proj + readout`
  - tiny head adapters

### Cluster H

- Locked confirmations on every headline result
- At least:
  - full_locked
  - finalquery_heavy
  - longdistance or distractor/query-distance shifted config

## Tuning, Promotion, Retirement

Each cluster follows the same ladder:

1. At least 4 concrete scout variants
2. At least 1 tuned top variant
3. At least 1 exact same-seed rerun
4. Promote only reproducible configs
5. If competitive:
   - 5-seed panel
   - locked confirms
   - independent verify recomputation

Retirement rule:

- No cluster is retired after one weak run.
- Every negative conclusion must have:
  - at least 4 concrete variants
  - at least 1 tuned top variant
  - at least 1 rerun
  - diagnostics
  - ledger retirement note

## Verification Protocol

Headlines are only confirmed after:

1. exact same-seed rerun
2. five-seed panel
3. locked confirmation evaluation
4. fair baseline comparison under similar budget
5. independent recomputation with `src/utils/phase9_verify.py`
6. ledger update to confirmed status

## Diagnostics

Mandatory phase-9 diagnostics:

- per-mode tables
- final-query accuracy, route match, exit time
- exit histograms by mode
- route-action histograms over time
- delay persistence curves
- content decodability from frozen states
- source-quality vs content-recovery tables
- source family vs recovery outcome tables
- seed variance plots
- confirmation vs validation vs test tables
- route-drift curves during refinement
- if Cluster E opens: route-preservation vs content-gain tradeoff
- if Cluster G opens: ES reward variance and update stats

## Stop Criteria

Positive exit:

- A head-only or minimal-safe pipeline reaches, across 5 seeds and locked confirms:
  - `fq_acc >= 0.45`
  - `fq_route >= 0.80`
  - `fq_exit >= 115`
  - `overall >= 0.60`
  - and clearly beats the relevant phase-8 baseline

Strong mapping exit:

- Clear answer to whether frozen route-faithful basins already contain decodable answer content
- Clear best head/objective family
- Clear boundary where minimal-safe partial unfreeze becomes necessary
- ES role at head level mapped
- Required clusters explored fairly and verified

## Exact Commands

Anchor reproductions:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase9_main.sh configs/phase8/dev/hard_st_benchmark_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874.yaml results/phase9_anchor/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_p1
CUDA_VISIBLE_DEVICES=1 ./scripts/run_phase9_main.sh configs/phase8/dev/hard_st_benchmark_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1879.yaml results/phase9_anchor/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1879_p1
CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase9_main.sh configs/phase8/dev/hard_st_benchmark_b_v2_teacher1879_refine_readoutonly_longer_lowlr.yaml results/phase9_anchor/hard_st_b_v2_teacher1879_refine_readoutonly_longer_lowlr_seed1886_p1
CUDA_VISIBLE_DEVICES=1 ./scripts/run_phase9_main.sh configs/phase8/dev/hard_st_benchmark_b_v2_teacher1821_refine_memoryreadout_longer_lowlr.yaml results/phase9_anchor/hard_st_b_v2_teacher1821_refine_memoryreadout_longer_lowlr_seed1842_p1
CUDA_VISIBLE_DEVICES=0 ./scripts/run_phase9_main.sh configs/phase7/dev/hybrid_es_benchmark_b_v2_controlsticky_keepalive_resume.yaml results/phase9_anchor/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1 results/phase8_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1/hard_st_best.pt 2
```

Cluster-A source audits:

```bash
./scripts/run_phase9_source_audits.sh results/phase9_anchor/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1874_p1 test
./scripts/run_phase9_source_audits.sh results/phase9_anchor/hard_st_b_v2_controlsticky_keepalive_teacher_keepalive_waitrelease_only_longrelease_delayed_dropout_selectacc_seed1879_p1 test
./scripts/run_phase9_source_audits.sh results/phase9_anchor/hard_st_b_v2_teacher1879_refine_readoutonly_longer_lowlr_seed1886_p1 test
./scripts/run_phase9_source_audits.sh results/phase9_anchor/hard_st_b_v2_teacher1821_refine_memoryreadout_longer_lowlr_seed1842_p1 test
./scripts/run_phase9_source_audits.sh results/phase9_anchor/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1 test
```

Confirmation verify:

```bash
./scripts/run_phase9_confirm.sh <run-dir>
```

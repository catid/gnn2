# Phase 10 Plan

## Starting Point

Phase 10 starts from the phase-9 mapping boundary:

- strong decodable frozen sources: `1874`, `1842`, `1201`
- medium but still decodable source family: `1821`
- fragile negative-control source: `1879`
- best strong-source baseline:
  - `9142 / 9146 / 9147 / 9148 / 9149`
  - base `0.8488 / 0.7004 / 0.9416 / 121.49`
  - full_locked `0.6534 / 0.3237 / 0.8771 / 115.49`
- best alternative-reader baseline:
  - `9132 / 9331 / 9332 / 9333 / 9334`
  - base `0.8231 / 0.6506 / 0.9454 / 121.86`
  - full_locked `0.6532 / 0.3232 / 0.8771 / 115.49`
- best medium-source baseline:
  - `9111 / 9351 / 9352 / 9353 / 9354`
  - base `0.6666 / 0.3498 / 0.9147 / 122.21`
  - full_locked `0.5936 / 0.2494 / 0.8394 / 115.99`
- fragile negative control:
  - base `0.2491 / 0.2500 / 0.8353 / 122.11`
  - full_locked `0.2498 / 0.2396 / 0.9887 / 126.97`

Primary question:

- can a better multi-view reader or tightly route-preserving read-path adapter
  improve held-confirm content on decodable frozen sources without harming route

## Anchors To Reproduce

Required reproductions before broad claims:

1. `9142` strong-source query-gated final-query-weighted family
2. `9132` query-FILM family
3. `9111` medium-source readout-only family
4. representative fragile `1879` head-only baseline
5. `1201` strong ES upper-bound source if used for content transfer

## Budget

- fruitful: about 75% of substantive runs
- exploration: about 25% of substantive runs

Target totals:

- at least 84 substantive runs
- hard floor 68
- at least 24 promoted runs
- at least 12 exact reruns
- at least 7 full five-seed panels
- at least 10 locked confirmations

## Clusters

### A. Multi-View Frozen-State Reader Architecture

Main target cluster.

- sources: `1874`, then `1842`, then `1821`, with `1879` only as negative control
- views:
  - `packet_state_query`
  - `final_sink_state`
  - `baseline_readout_input`
  - pairwise and triple combinations
- reader families:
  - concat + MLP fusion
  - query-gated fusion
  - FiLM fusion
  - cross-attention over frozen view bank

Minimum:

- 10 concrete variants
- 3 tuned top variants
- 3 promoted runs
- rerun for every promoted run
- 1 five-seed panel
- 2 locked confirmations

### B. Route-Preserving Read-Path Adapter

This is `gnn2-pyv`.

- start from best `1874` frozen-head baselines
- keep memory/router/control frozen
- only adapt fused read-path features or downstream read-path projections
- adapter families:
  - low-rank fused-feature adapter
  - residual MLP adapter
  - affine bias/scale adapter

Minimum:

- 8 concrete variants
- 3 tuned top variants
- 3 promoted runs
- rerun for every promoted run
- 1 five-seed panel
- 2 locked confirmations

### C. Reader Objective / Data-Mix / Content Distillation

- objective families:
  - plain CE
  - final-query-weighted CE
  - heavier final-query weighting
  - confirmation-mixed schedules
  - finalquery-heavy mixed schedules
  - content-only distillation from `1201` or another strong source

Minimum:

- 8 concrete variants
- 3 tuned top variants
- 2 promoted runs
- rerun for every promoted run
- 1 five-seed panel
- 2 locked confirmations

### D. Portability Across Decodable Families

- apply top 2-3 strong-source readers/adapters to `1842` and `1821`
- run at least one negative-control check on `1879`

### E. Iterative / Recurrent Readers

Exploration cluster.

- small iterative cross-attention or recurrent readers only
- frozen routing and memory

### F. Minimal-Safe Partial Unfreeze Beyond Read Path

Gated cluster only after A-C plateau cleanly.

- tiny adapter near read path only
- no broad `memory_`, router, or control reopening

### G. Head-Level ES / Black-Box Polish

Exploration cluster only after strong gradient readers exist.

### H. Stress / Confirmation / Generalization

Mandatory for every headline result:

- full_locked
- finalquery_heavy
- longdistance

## Promotion And Retirement Rules

Promotion:

1. scout result is competitive on base and does not obviously damage route
2. tuned rerun reproduces same-seed behavior
3. confirmation suite does not collapse route

Retirement:

1. at least 4 concrete variants
2. at least 1 tuned top variant
3. at least 1 rerun
4. failure mode is recorded in the ledger

## Verification Protocol

Headline result ladder:

1. exact same-seed rerun
2. five-seed panel
3. locked confirmations
4. baseline comparison
5. independent recomputation via `src/utils/phase10_verify.py`
6. ledger updated

## Commands

```bash
./scripts/run_phase10_cluster_scouts.sh results/phase10_dev anchor
./scripts/run_phase10_cluster_scouts.sh results/phase10_dev multiview
./scripts/run_phase10_cluster_scouts.sh results/phase10_dev adapter
./scripts/run_phase10_reader_sweeps.sh results/phase10_dev tuned
./scripts/run_phase10_main.sh <config> <results-dir> [resume] [nproc_per_node]
./scripts/run_phase10_confirm.sh <run-dir> [extra-eval-config ...]
./scripts/run_phase10_seed_panels.sh <config> <results-root> <resume> <seed1> [seed2 ...]
./scripts/run_phase10_source_audits.sh <run-dir> [split] [eval-config]
```

## Stop Criteria

Positive exit:

- a strong-source reader or read-path adapter clears:
  - full_locked `fq_acc >= 0.40`
  - full_locked `fq_route >= 0.88`
  - full_locked `fq_exit >= 115`
  - base `fq_acc >= 0.72`
  - with full verification

Otherwise strong mapping exit requires:

- clear view-importance map
- clear multi-view vs single-view answer
- clear adapter-vs-frozen-head answer
- fair negative on `1879`
- clear next experiment if the held-confirm ceiling remains

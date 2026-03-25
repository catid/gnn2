# Phase 12 Plan

## Starting Point

Phase 12 starts from local clean handoff `5844cbe` with `master...origin/master [ahead 77]`.

Current confirmed baselines:

- strong-source `1874` multiview query-gated family:
  - base `0.9891 / 0.9775 / 0.9445 / 121.99`
  - full_locked `0.6589 / 0.2996 / 0.8797 / 116.14`
- strong-source phase-9 frozen-head query-gated baseline:
  - base `0.8488 / 0.7004 / 0.9416 / 121.49`
  - full_locked `0.6534 / 0.3237 / 0.8771 / 115.49`
- medium-source `1821` multiview query-gated fq5 family:
  - base `0.8920 / 0.8027 / 0.9258 / 122.78`
  - full_locked `0.6135 / 0.2557 / 0.8289 / 115.23`
- frozen-`1201` readout-prefix transfer boundary `13411`:
  - base `0.6479 / 0.3826 / 0.7884 / 123.97`
  - full_locked `0.1306 / 0.2539 / 0.9876 / 126.43`
- negative-control `1879` remains route-faithful but content-poor.

Metric order everywhere is `overall / fq_acc / fq_route / fq_exit`.

## Main Question

Can a route-trace-conditioned temporal-bank reader break the strong-source held-confirm content ceiling on decodable frozen sources such as `1874`, without harming late-route fidelity?

## Working Hypotheses

- Main hypothesis:
  the missing held-confirm gains are in reader addressing over the frozen trajectory, not in routing or generic decodability.
- Alternative hypothesis:
  even better trajectory readers plateau because the sink compresses away answer-critical structure, in which case the first justified upstream change is a tiny keyed or multi-slot sink.

## Anchor Reproductions

Before broader claims:

1. Reproduce the best strong-source multiview query-gated baseline on `1874`.
2. Reproduce the strongest comparable alternative baseline on `1874` (`query-FILM`).
3. Reproduce the best medium-source portability baseline on `1821`.
4. Reproduce the strongest stable frozen-`1201` readout-prefix transfer boundary (`13411`).
5. Reproduce one `1879` negative-control head-only baseline.

## Budget

Planned target: `80` substantive new runs.

- fruitful half: `64` runs
- exploration half: `16` runs

This preserves the intended `80 / 20` split while comfortably exceeding the hard floor.

Planned allocation:

- Cluster A: `24`
- Cluster B: `14`
- Cluster C: `12`
- Cluster D: `14`
- Cluster E: `4`
- Cluster F: gated reserve, `0-4` only if A-D plateau
- Cluster G: `0-3`, only if a strong reader becomes competitive
- Cluster H: `8`

If Cluster F is entered, it consumes exploration reserve first and does not reduce A-C below the required coverage.

## Clusters

### Cluster A

Route-trace-conditioned temporal-bank readers on `1874`.

Reader families:

- final sink only control
- late sink window bank
- late readout/logit window bank
- mixed sink + readout bank
- query-anchored bank
- exit-relative bank
- route-delay-segment bank
- route-blind controls

Route-trace features:

- exit time
- delay counts / delay mask summaries
- compact route-action histogram
- route-relative positions

### Cluster B

Factorized content/query readers with narrow auxiliary latent supervision.

Core comparisons:

- content branch from endpoint vs temporal bank
- route-conditioned content branch vs route-blind
- bilinear / gated / FiLM combiners
- early vs late fusion
- no-aux vs content-aux control

### Cluster C

Contiguous-window transfer and portability on `1874`, `1201`, and one of `1821` / `1842`.

Core comparisons:

- contiguous slab vs sparse sample
- contiguous slab vs endpoint-only
- `24..72`-style slab transfer on `1201`
- analogous slabs on `1874` and `1821`
- half-window / midpoint / overlap negatives

### Cluster D

Objective/data-mix tuning only for the strongest reader families from A-C.

Allowed levers:

- plain CE
- final-query-weighted CE
- targeted held-confirm / finalquery-heavy mixes
- content-only distillation
- consistency regularization across confirmation settings

### Cluster E

Probe-guided downstream adapters only.

### Cluster F

Gated keyed or multi-slot sink changes only if A-D plateau.

### Cluster G

Optional narrow ES polish on a strong trajectory-aware reader only.

### Cluster H

Locked confirmation, finalquery-heavy, distractor/query-distance stress, panels, and verify.

## Promotion Rules

- A scout promotes only if it beats the strongest relevant baseline on either:
  - selected validation content at matched route, or
  - confirmation metrics after rerun.
- Every promoted variant must get:
  - exact rerun
  - locked confirmation
  - ledger update
- Every headline family must get:
  - five-seed panel
  - independent verify

## Retirement Rules

A negative is only retired after:

- at least `4` concrete variants inside the cluster
- at least `1` tuned top variant
- at least `1` exact rerun
- diagnostics showing where it fails
- retirement note recorded in the ledger / docs

## Verification Protocol

Headline result checklist:

1. exact same-seed rerun
2. five-seed panel
3. locked confirmation evaluation
4. comparison against strongest relevant baseline at similar budget
5. independent recomputation with `phase12_verify`
6. ledger entry updated with confirmation status

## Stop Criteria

Positive exit:

- on `1874` or another strong decodable source,
- five-seed and locked-confirm verified
- full_locked `fq_acc >= 0.40`
- full_locked `fq_route >= 0.88`
- full_locked `fq_exit >= 115`
- base `fq_acc >= 0.72`
- clear improvement over phase-10/11 baseline

Strong mapping exit:

- trajectory-aware vs endpoint-only answer is verified
- route-trace conditioning answer is verified
- contiguous vs sparse window answer is verified
- factorized content/query answer is verified
- sink-compression question is narrowed sharply enough to justify or reject Cluster F

## Commands

Anchors:

```bash
./scripts/run_phase12_cluster_scouts.sh results/phase12_anchor anchor
```

Trajectory-bank scouts:

```bash
./scripts/run_phase12_reader_banks.sh results/phase12_dev bank
```

Factorized-reader scouts:

```bash
./scripts/run_phase12_reader_banks.sh results/phase12_dev factorized
```

Contiguous-window / portability scouts:

```bash
./scripts/run_phase12_reader_banks.sh results/phase12_dev windows
```

Single run:

```bash
./scripts/run_phase12_main.sh <config> <results-dir> [resume] [nproc_per_node]
```

Confirmation:

```bash
./scripts/run_phase12_confirm.sh <run-dir> [extra-eval-config ...]
```

Panels:

```bash
./scripts/run_phase12_seed_panels.sh <config> <results-root> <resume> <seed1> [seed2 ...]
```

# gnn2

Synthetic experiment suite for testing whether an EGGROLL-inspired hybrid
low-rank evolutionary search method makes a hard-routing spatio-temporal
packet-routing graph network practical on a single 2-GPU machine.

The implementation is intentionally **EGGROLL-inspired**, not an exact
reproduction of the paper or the HyperscaleES JAX stack. The repo focuses on:

- real hard routing in the forward pass with `FORWARD`, `EXIT`, and `DELAY`
- forward-only low-rank ES over router and optional adapter parameters
- 2-GPU population sharding with `torchrun`
- direct optimization of task loss plus hop, delay, and TTL penalties
- reproducible synthetic benchmarks for adaptive routing and long-horizon memory

## Environment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

PyTorch is installed from the CUDA 13.0 nightly index through `uv`.

## Rebuild Parallelism

When building native extensions or rebuilding local packages, keep rebuilds at
16 threads:

```bash
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export MAKEFLAGS=-j16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

## Repo Layout

```text
src/
  data/      synthetic benchmarks
  models/    packet-routing graph model
  es/        low-rank ES implementation
  train/     training entrypoint
  utils/     config loading and report generation
configs/
  smoke/     minute-scale correctness checks
  dev/       moderate comparison runs
  main/      larger runs for the target machine
scripts/
  run_smoke.sh
  run_main.sh
  report.sh
tests/
docs/
results/
references/
```

## Reproduce

Smoke suite:

```bash
./scripts/run_smoke.sh results/smoke_suite
```

Dev comparison suite:

```bash
./scripts/run_main.sh dev results/dev_suite
```

Main suite:

```bash
./scripts/run_main.sh main results/main_suite
```

Regenerate the markdown report from an existing results root:

```bash
./scripts/report.sh results/dev_suite docs/experiment_report.md
```

## Direct Commands

Single-run commands:

```bash
uv run python -m src.train.run --config configs/smoke/soft.yaml
uv run python -m src.train.run --config configs/smoke/hard_st.yaml
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run --config configs/smoke/hybrid_es.yaml

uv run python -m src.train.run --config configs/dev/soft_benchmark_a.yaml
uv run python -m src.train.run --config configs/dev/hard_st_benchmark_a.yaml
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run --config configs/dev/hybrid_es_benchmark_a.yaml

uv run python -m src.train.run --config configs/dev/soft_benchmark_b.yaml
uv run python -m src.train.run --config configs/dev/hard_st_benchmark_b.yaml
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run --config configs/dev/hybrid_es_benchmark_b.yaml

uv run python -m src.utils.report --results-dir results/dev_suite --out docs/experiment_report.md
```

## Tests

```bash
uv run pytest -q
```

The test suite covers:

- hard-routing semantics
- mailbox / `DELAY` behavior
- low-rank ES perturbation shapes and antithetic behavior
- reproducibility for fixed seeds
- a tiny end-to-end CLI smoke run

## Current Report

The latest generated report is at [docs/experiment_report.md](docs/experiment_report.md).

Current headline outcome from the dev suite:

- Benchmark A: the soft model wins raw accuracy, but hybrid ES beats hard ST on
  both task quality and route optimality while using less compute than soft
- Benchmark B: all methods collapse to early exit and near-chance accuracy, and
  the current hybrid setup does not rescue long-horizon delay-memory behavior

## Phase 2 Commands

Phase 2 adds a benchmark audit, a revised adaptive Benchmark B v2, deeper route
diagnostics, and promoted long-horizon reruns. The main entrypoints are:

```bash
uv run python -m src.utils.benchmark_audit \
  --config configs/phase2/audit/benchmark_b_v1.yaml \
  --out results/phase2_audit/benchmark_b_v1/audit.json

uv run python -m src.utils.benchmark_audit \
  --config configs/phase2/audit/benchmark_b_v2.yaml \
  --out results/phase2_audit/benchmark_b_v2/audit.json

uv run python -m src.train.run \
  --config configs/phase2/dev/hard_st_benchmark_b_v2_gatedblend_maskcurr.yaml \
  --results-dir results/phase2_dev/hard_st_b_v2_gatedblend_maskcurr

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase2/dev/hybrid_es_benchmark_b_v2_gatedblend_writeaux_maskcurr_pop64.yaml \
  --results-dir results/phase2_dev/hybrid_es_b_v2_gatedblend_writeaux_maskcurr_pop64

uv run python -m src.train.run \
  --config configs/phase2/main/hard_st_benchmark_b_v2_maskcurr_h256.yaml \
  --results-dir results/phase2_main/hard_st_b_v2_maskcurr_h256

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase2/main/hybrid_es_benchmark_b_v2_maskcurr_h256_stable.yaml \
  --results-dir results/phase2_main/hybrid_es_b_v2_maskcurr_h256_stable

./scripts/run_phase2_seed_compare.sh results/phase2_final

uv run python -m src.utils.phase2_report \
  --results-dir results/phase2_audit \
  --results-dir results/phase2_dev \
  --results-dir results/phase2_main \
  --results-dir results/phase2_final \
  --out docs/phase2_report.md
```

The seed-sweep configs for the final hard-ST comparison live under
`configs/phase2/final/`.

## References

The original paper and reference code used to keep the implementation honest are
stored locally under:

- `references/eggroll_paper.pdf`
- `references/eggroll_paper.md`
- `references/HyperscaleES/`
- `docs/eggroll_reference_alignment.md`

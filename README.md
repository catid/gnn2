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

## References

The original paper and reference code used to keep the implementation honest are
stored locally under:

- `references/eggroll_paper.pdf`
- `references/eggroll_paper.md`
- `references/HyperscaleES/`
- `docs/eggroll_reference_alignment.md`

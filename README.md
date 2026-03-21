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

The latest targeted long-horizon reports are at
[docs/phase4_report.md](docs/phase4_report.md) and
[docs/phase5_report.md](docs/phase5_report.md).

Current headline outcome from the dev suite:

- Benchmark A: the soft model wins raw accuracy, but hybrid ES beats hard ST on
  both task quality and route optimality while using less compute than soft
- Benchmark B phase 2: all methods collapse to early exit and near-chance
  accuracy, and the initial hybrid setup does not rescue long-horizon
  delay-memory behavior
- Benchmark B phase 3: explicit packet memory plus oracle-route pretraining and
  a no-mask hard-routing release improves the 3-seed hard-routing result from
  `0.4446 +/- 0.0048` to `0.6110 +/- 0.0308`, with real delayed behavior
  (`delay_rate ~= 0.69`), but the gain is concentrated in the
  `delay_to_trigger_exit` mode while `delay_to_final_query` remains near chance
- Benchmark B phase 4: explicit control-state interventions can produce real
  single-seed final-query improvements, but the promoted hard-routing seed panel
  still collapses back to premature exit, so the primary story is a stronger
  negative result about robustness; the notable secondary result is that
  resume-based hybrid ES can polish a working controller checkpoint much more
  successfully than scratch hybrid warmstart can discover one
- Benchmark B phase 5: medium-basin resumed hybrid ES is now confirmed by an
  exact same-seed rerun and confirmation split, router-only ES is essentially as
  good as router+adapter ES on that medium basin, and weak-basin ES can repair
  routing completely while only partially recovering task accuracy

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

## Phase 3 Commands

Phase 3 adds explicit packet memory, payload-aware write/read auxiliaries, and
oracle-release follow-ups that isolate the Benchmark B v2 train/eval mismatch.
The main entrypoints are:

```bash
uv run python -m src.train.run \
  --config configs/phase3/dev/hard_st_benchmark_b_v2_keymem_payloadaux_oraclewarm.yaml \
  --results-dir results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase3/dev/hybrid_es_benchmark_b_v2_keymem_payloadaux_maskcurr_pop64.yaml \
  --results-dir results/phase3_dev/hybrid_es_b_v2_keymem_payloadaux_maskcurr_pop64

./scripts/run_phase3_release_followups.sh phase3_release
```

The phase-3 writeup is [docs/phase3_release_note.md](docs/phase3_release_note.md).

## Phase 4 Commands

Phase 4 targets the remaining `delay_to_final_query` failure with explicit
control-state diagnostics, sticky control-memory interventions, a promoted
hard-routing seed panel, and a hybrid-ES retest on the improved architecture.

```bash
uv run python -m src.utils.phase4_audit \
  --run-dir results/phase3_dev/hard_st_b_v2_keymem_payloadaux_release_nomask_from_oraclewarm \
  --num-batches 8 \
  --probe-train-batches 8 \
  --probe-test-batches 8

uv run python -m src.train.run \
  --config configs/phase4/dev/hard_st_benchmark_b_v2_control_sticky_aux_router2.yaml \
  --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
  --results-dir results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2

uv run python -m src.train.run \
  --config configs/phase4/main/hard_st_benchmark_b_v2_control_sticky_both_main.yaml \
  --resume results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm/hard_st_best.pt \
  --results-dir results/phase4_main/hard_st_b_v2_control_sticky_both_main_seed750

./scripts/run_phase4_seed_compare.sh

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase4/main/hybrid_es_benchmark_b_v2_control_sticky_aux_router2_resume_seed747.yaml \
  --resume results/phase4_dev/hard_st_b_v2_control_sticky_aux_router2/hard_st_best.pt \
  --results-dir results/phase4_main/hybrid_es_b_v2_control_sticky_aux_router2_resume_seed747
```

The phase-4 planning and final writeups are
[docs/phase4_plan.md](docs/phase4_plan.md),
[docs/phase4_report.md](docs/phase4_report.md), and
[docs/phase4_lessons.md](docs/phase4_lessons.md).

## Phase 5 Commands

Phase 5 focuses on controller-factorization follow-ups, stricter verification,
and ES role mapping across checkpoint quality.

```bash
./scripts/run_phase5_main.sh medium-adapter
./scripts/run_phase5_main.sh medium-routeronly
./scripts/run_phase5_main.sh weak-adapter
./scripts/run_phase5_main.sh weak-routeronly

./scripts/run_phase5_confirm.sh \
  results/phase5_dev/hybrid_es_b_v2_control_router2_setclear_oraclecontrol_resume_from950_seed951_p1

./scripts/run_phase5_seed_panels.sh

uv run python -m src.utils.phase5_audit \
  --run-dir results/phase5_dev/hard_st_b_v2_control_router2_setclear_oraclecontrol_opt_selectexit_seed950_p1 \
  --split confirm \
  --num-batches 6 \
  --probe-train-batches 6 \
  --probe-test-batches 6
```

The phase-5 planning and writeups are
[docs/phase5_plan.md](docs/phase5_plan.md),
[docs/phase5_report.md](docs/phase5_report.md), and
[docs/phase5_lessons.md](docs/phase5_lessons.md).

## Phase 7 Commands

Phase 7 broadens the campaign into a balanced discovery / recovery / ES map and
tracks the full run ledger in `docs/phase7_run_matrix.csv`.

Anchor reproduction:

```bash
./scripts/run_phase7_cluster_scouts.sh anchors
```

Representative cluster runs:

```bash
./scripts/run_phase7_main.sh \
  configs/phase7/dev/hard_st_benchmark_b_v2_forceoracle_release_longerstrong_refine_memoryreadout_seed1305.yaml \
  results/phase7_dev/hard_st_b_v2_forceoracle_release_longerstrong_refine_memoryreadout_seed1305_p1

./scripts/run_phase7_main.sh \
  configs/phase7/dev/hybrid_es_benchmark_b_v2_controlsticky_keepalive_resume.yaml \
  results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1 \
  results/phase7_anchor/hard_st_b_v2_controlsticky_keepalive_seed989_rerun1/hard_st_best.pt \
  2
```

Confirmation / verification:

```bash
./scripts/run_phase7_confirm.sh \
  results/phase7_dev/hybrid_es_b_v2_controlsticky_keepalive_resume_from989_seed1201_p1
```

Seed panels:

```bash
./scripts/run_phase7_seed_panels.sh \
  configs/phase7/dev/hard_st_benchmark_b_v2_forceoracle_release_longerstrong_refine_memoryreadout_seed1305.yaml \
  results/phase7_panel/hard_st_b_v2_forceoracle_release_longerstrong_refine_memoryreadout \
  '' \
  1302 1303 1304 1305 1306
```

The phase-7 planning and final writeups are
[docs/phase7_plan.md](docs/phase7_plan.md),
[docs/phase7_report.md](docs/phase7_report.md),
[docs/phase7_lessons.md](docs/phase7_lessons.md),
[docs/phase7_cluster_scorecards.md](docs/phase7_cluster_scorecards.md), and
[docs/phase7_run_matrix.csv](docs/phase7_run_matrix.csv).

## References

The original paper and reference code used to keep the implementation honest are
stored locally under:

- `references/eggroll_paper.pdf`
- `references/eggroll_paper.md`
- `references/HyperscaleES/`
- `docs/eggroll_reference_alignment.md`

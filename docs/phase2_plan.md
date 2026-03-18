# Phase 2 Plan

## Repo Start State

- Base report: `docs/experiment_report.md`
- Strongest hard-routing Benchmark B result so far:
  - `results/dev_suite/hard_st_benchmark_b_h128`: accuracy `0.271`
  - `results/dev_suite/hybrid_es_benchmark_b_h128`: accuracy `0.246`
- Shared failure mode across soft, hard-ST, and hybrid ES:
  - near-chance accuracy on Benchmark B
  - `early_exit_rate ~= 1.0`
  - `delays ~= 0.0`

## Revision After Audit

- Benchmark B v1 was preserved as `long_horizon_memory_v1`.
  - It is a real delayed-memory task, but not a good adaptive-routing benchmark because every sample shares the same oracle delay-until-final route.
- Benchmark B v2 was added as `long_horizon_memory_v2`.
  - It mixes `easy_exit`, `delay_to_trigger_exit`, and `delay_to_final_query` cases so adaptive routing is genuinely necessary.
- The phase-2 success condition was narrowed accordingly:
  - any headline accuracy gain must be checked against delayed-mode behavior rather than counted as a win automatically
  - the main question is whether delayed retrieval survives validation/test, not merely whether raw accuracy rises through better handling of `easy_exit`

## Main Question

Can the current EGGROLL-inspired hybrid low-rank ES method be made materially better on the
long-horizon hard-routing memory problem, or does the current setup reveal a clearer bottleneck in
the benchmark, architecture, objective, or ES search procedure?

## Working Hypotheses

1. Benchmark B v1 may be misaligned with the intended route-learning story.
   - The task is technically solvable by delayed routing, but the current objective may never expose a
     meaningful advantage for `DELAY` during early training.
2. The current memory architecture is too weak.
   - A one-step mailbox plus shallow packet state may be insufficient to preserve payload information
     over long horizons under repeated forced delay.
3. The current objective likely rewards immediate `EXIT` too strongly relative to eventual retrieval.
   - Compute penalties are low in absolute terms, but the optimization landscape may still favor the
     low-variance always-exit basin.
4. The current hybrid ES search is probably not the primary failure at the start.
   - Soft warm-start already collapses to immediate exit, so the first failure may be benchmark/objective
     or memory architecture rather than ES alone.

## Intervention Families

### Family A: Architecture / Memory

- Add a configurable multi-slot mailbox / delay memory instead of a single carry bucket.
- Add explicit packet age and trigger-distance features where available.
- Add a stronger persistent packet memory path so delayed packets do not repeatedly overwrite
  themselves with mostly-noise observations.

### Family B: Objective / Curriculum / Supervision

- Audit and rebalance compute penalties for Benchmark B.
- Add an auxiliary warm-start curriculum:
  - optional route imitation from an oracle policy
  - optional delayed enablement of `EXIT`
  - optional horizon curriculum
- Add anti-collapse terms such as route entropy bonus or explicit penalty on pathological always-exit.

### Family C: ES / Search / Optimization

- Improve ES diagnostics first: reward variance, reward normalization, sigma/rank/population traces.
- Sweep longer warm-starts and router-only vs router+adapter ES.
- Try alternating gradient updates and ES updates for router/control parameters.

### Benchmark Track: Benchmark B Audit / Revision

- Build audit tooling for Benchmark B v1.
- If v1 is misaligned, preserve it and add Benchmark B v2 beside it.
- Compare v1 vs v2 directly rather than silently replacing the old benchmark.

## Required Audit Outputs

1. Oracle or heuristic route policy for Benchmark B.
2. Degenerate-policy baselines:
   - always exit now
   - always delay until final step
   - simple rule-based trigger-query policy if feasible
3. Sensitivity slices:
   - trigger position
   - retrieval distance
   - horizon
   - noise level
4. Direct evidence of whether `DELAY` is ever rewarded under the current objective.
5. Recommendation on whether Benchmark B v2 is necessary.

## Run Budget

Minimum new substantive runs in this phase:

1. Audit runs and heuristic baselines: at least 3
2. Family A dev pilots/sweeps: at least 3
3. Family B dev pilots/sweeps: at least 3
4. Family C dev pilots/sweeps: at least 3
5. Promoted main runs: at least 2
6. Final 3-seed comparison on the best candidate vs strongest relevant baseline: at least 6 runs total

Target total: 15 to 20 new non-smoke runs under `results/phase2_*`.

## Promotion Criteria

Promote a config from dev to main only if at least one of the following holds at `T >= 128`:

- `>= 0.05` absolute accuracy gain over the current matching method family baseline
- `delays >= 0.20` with non-trivial compute-adjusted behavior
- clear route-distribution shift away from `early_exit_rate ~= 1.0`
- diagnostic evidence that the method is using delayed memory but undertrained

## Stop Criteria

Positive-result exit:

- At least one hard-routing method reaches `>= 0.10` absolute accuracy gain over the current strongest
  hard-routing Benchmark B result at `T >= 128`, or shows clearly non-trivial delay behavior together
  with improved compute-adjusted score, and the result reproduces across 3 seeds.

Strong-negative-result exit:

- After a benchmark audit, deep diagnostics, and at least 3 intervention families with promoted reruns,
  Benchmark B still fails, but the bottleneck is clearly demonstrated and documented.

## Executed / Promoted Commands

Benchmark audit and analysis:

```bash
uv run python -m src.utils.benchmark_audit \
  --config configs/phase2/audit/benchmark_b_v1.yaml \
  --out results/phase2_audit/benchmark_b_v1/audit.json

uv run python -m src.utils.benchmark_audit \
  --config configs/phase2/audit/benchmark_b_v2.yaml \
  --out results/phase2_audit/benchmark_b_v2/audit.json
```

Family A memory pilots:

```bash
uv run python -m src.train.run \
  --config configs/phase2/dev/hard_st_benchmark_b_v1_gru_oraclewarm.yaml \
  --results-dir results/phase2_dev/hard_st_b_v1_gru_oraclewarm

uv run python -m src.train.run \
  --config configs/phase2/dev/hard_st_benchmark_b_v1_gatedblend_oraclewarm.yaml \
  --results-dir results/phase2_dev/hard_st_b_v1_gatedblend_oraclewarm

uv run python -m src.train.run \
  --config configs/phase2/dev/hard_st_benchmark_b_v2_gatedblend.yaml \
  --results-dir results/phase2_dev/hard_st_b_v2_gatedblend
```

Family B objective/curriculum pilots:

```bash
uv run python -m src.train.run \
  --config configs/phase2/dev/hard_st_benchmark_b_v2_gatedblend_maskcurr.yaml \
  --results-dir results/phase2_dev/hard_st_b_v2_gatedblend_maskcurr

uv run python -m src.train.run \
  --config configs/phase2/dev/hard_st_benchmark_b_v2_gatedblend_writeaux_maskcurr.yaml \
  --results-dir results/phase2_dev/hard_st_b_v2_gatedblend_writeaux_maskcurr
```

Family C ES/search pilots:

```bash
uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase2/dev/hybrid_es_benchmark_b_v2_gatedblend_writeaux_maskcurr_pop64.yaml \
  --results-dir results/phase2_dev/hybrid_es_b_v2_gatedblend_writeaux_maskcurr_pop64

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase2/main/hybrid_es_benchmark_b_v2_maskcurr_h256_stable.yaml \
  --results-dir results/phase2_main/hybrid_es_b_v2_maskcurr_h256_stable
```

Promoted main runs:

```bash
uv run python -m src.train.run \
  --config configs/phase2/main/hard_st_benchmark_b_v2_maskcurr_h256.yaml \
  --results-dir results/phase2_main/hard_st_b_v2_maskcurr_h256

uv run torchrun --standalone --nproc_per_node=2 -m src.train.run \
  --config configs/phase2/main/hybrid_es_benchmark_b_v2_maskcurr_h256.yaml \
  --results-dir results/phase2_main/hybrid_es_b_v2_maskcurr_h256
```

Final seed comparison:

```bash
uv run python -m src.train.run \
  --config configs/phase2/final/hard_st_benchmark_b_v2_gatedblend_seed601.yaml \
  --results-dir results/phase2_final/hard_st_b_v2_gatedblend_seed601

uv run python -m src.train.run \
  --config configs/phase2/final/hard_st_benchmark_b_v2_maskcurr_seed611.yaml \
  --results-dir results/phase2_final/hard_st_b_v2_maskcurr_seed611

uv run python -m src.train.run \
  --config configs/phase2/final/hard_st_benchmark_b_v2_gatedblend_seed602.yaml \
  --results-dir results/phase2_final/hard_st_b_v2_gatedblend_seed602

uv run python -m src.train.run \
  --config configs/phase2/final/hard_st_benchmark_b_v2_maskcurr_seed612.yaml \
  --results-dir results/phase2_final/hard_st_b_v2_maskcurr_seed612

uv run python -m src.train.run \
  --config configs/phase2/final/hard_st_benchmark_b_v2_gatedblend_seed603.yaml \
  --results-dir results/phase2_final/hard_st_b_v2_gatedblend_seed603

uv run python -m src.train.run \
  --config configs/phase2/final/hard_st_benchmark_b_v2_maskcurr_seed613.yaml \
  --results-dir results/phase2_final/hard_st_b_v2_maskcurr_seed613
```

Report generation:

```bash
uv run python -m src.utils.phase2_report --results-dir results/phase2_audit --results-dir results/phase2_dev --results-dir results/phase2_main --results-dir results/phase2_final --out docs/phase2_report.md
```

## Expected Failure Modes To Watch

- always-exit collapse even when `DELAY` is necessary
- soft warm-start never discovering a useful delayed trajectory
- mailbox state carrying only noise due to observation overwrite
- compute penalties becoming irrelevant after collapse instead of shaping behavior
- ES variance dominating once the warm-start is already collapsed
- benchmark v1 being technically delay-based but too weakly coupled to actual routing decisions

# Phase 3 Benchmark B Release Note

This note captures the follow-up experiments run after the phase-2 conclusion
that Benchmark B v2 always collapsed to immediate `EXIT`.

## Question

Once explicit packet memory fixes the content-storage problem, can a hard-routing
policy keep useful `DELAY` behavior at evaluation time?

## Setup

Phase 3 added:

- explicit packet-carried memory with keyed read/write
- trigger-time write supervision and optional payload retrieval supervision
- an oracle-routed warm start that nearly solves Benchmark B v2 when routing is
  fixed
- a release-stage comparison that resumes from the oracle-routed checkpoint and
  then trains under free hard routing

The critical control is whether the release stage keeps the phase-2
metadata-driven action masks.

## Key Results

| Run | Test acc | Delay rate | Route match | Early-exit rate | Compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase-2 hard-ST baseline, 3 seeds | 0.4446 +/- 0.0048 | 0.0002 +/- 0.0001 | 0.2559 +/- 0.0000 | 0.9997 +/- 0.0003 | 1.0003 +/- 0.0003 |
| Phase-3 key-memory hybrid ES, maskcurr | 0.4401 | 0.0000 | 0.2559 | 1.0000 | 1.0000 |
| Phase-3 oracle release with masks | 0.4473 | 0.0039 | 0.2559 | 0.9922 | 1.0081 |
| Phase-3 oracle release without masks, 3 seeds | 0.6110 +/- 0.0308 | 0.6922 +/- 0.0145 | 0.4851 +/- 0.0399 | 0.2614 +/- 0.0078 | 23.4136 +/- 2.6542 |

Result directories:

- `results/phase3_dev/hybrid_es_b_v2_keymem_payloadaux_maskcurr_pop64`
- `results/phase3_dev/hard_st_b_v2_keymem_payloadaux_release_maskcurr_from_oraclewarm`
- `results/phase3_dev/hard_st_b_v2_keymem_payloadaux_release_nomask_from_oraclewarm`
- `results/phase3_dev/hard_st_b_v2_keymem_payloadaux_release_nomask_seed712`
- `results/phase3_dev/hard_st_b_v2_keymem_payloadaux_release_nomask_seed713`

## Interpretation

The mask curriculum was a major part of the old failure mode.

- With the train-time masks left on during release, the model falls straight back
  to the phase-2 collapse.
- With the masks removed, the same oracle-routed initialization keeps real hard
  `DELAY` behavior at evaluation time and improves hard-routing accuracy by
  about `+0.166` absolute over the previous 3-seed baseline.

This is not a full solution. The gain is mode-specific:

- `easy_exit` stays solved at `1.0` accuracy.
- `delay_to_trigger_exit` improves from `0.2464 +/- 0.0093` to
  `0.9190 +/- 0.1146`.
- `delay_to_final_query` remains near chance:
  `0.2574 +/- 0.0147` in phase 2 versus `0.2440 +/- 0.0028` here.

So phase 3 changed the scientific story:

- hard delayed routing is practical for the trigger-exit subproblem
- the remaining Benchmark B bottleneck is long-distance final-query retrieval,
  not willingness to delay
- mask-driven training can make train metrics look route-perfect without
  learning a deployable policy

## Reproduction

Generate the oracle-routed keyed-memory checkpoint:

```bash
uv run python -m src.train.run \
  --config configs/phase3/dev/hard_st_benchmark_b_v2_keymem_payloadaux_oraclewarm.yaml \
  --results-dir results/phase3_dev/hard_st_b_v2_keymem_payloadaux_oraclewarm
```

Run the release-stage control and seed panel:

```bash
./scripts/run_phase3_release_followups.sh phase3_release
```

Inspect the summaries:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("results/phase3_dev").glob("hard_st_b_v2_keymem_payloadaux_release*/summary.json")):
    d = json.loads(path.read_text())["summary"]["test"]
    print(path.parent.name, d["accuracy"], d["delay_rate"], d["route_match"])
PY
```

## Next Bottleneck

The next intervention should target `delay_to_final_query` directly. The model is
now willing to delay and can solve trigger-timed retrieval, but it still fails
to preserve or decode the payload across the longest query distances.


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

## Follow-Up Sweep

After the first 3-seed no-mask release result, I ran a small release-stage
route-supervision sweep with the same resume checkpoint and stronger
non-masking `oracle_route_weight` values:

| Run | Test acc | Delay rate | Route match | Final-query acc | Final-query exit time |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-mask release, `oracle_route_weight=0.2` | 0.6328 | 0.7027 | 0.5130 | 0.2460 | 33.19 |
| No-mask release, `oracle_route_weight=0.5` | 0.6445 | 0.7019 | 0.5127 | 0.2701 | 33.21 |
| No-mask release, `oracle_route_weight=1.0` | 0.6429 | 0.7026 | 0.5143 | 0.2680 | 34.18 |

This did not materially solve the final-query mode. Stronger route CE nudged
overall accuracy upward a bit, but `delay_to_final_query` still exited around
step `34` instead of step `127`, and its route match stayed effectively zero.
That narrows the remaining failure further: it is not just weak release-stage
route supervision.

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

The next intervention should target a durable `needs_final_query` signal. The
model is now willing to delay and can solve trigger-timed retrieval, but on the
final-query mode it still behaves like a generic trigger-timed policy and exits
around the average trigger distance instead of persisting to the end.

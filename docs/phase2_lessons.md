# Phase 2 Lessons

## What Changed Relative to Phase 1

- Added Benchmark B v1 audit tooling and preserved it as a fixed-route delayed-memory stress test.
- Added Benchmark B v2 as an adaptive-routing version with mixed `easy_exit`, `delay_to_trigger_exit`, and `delay_to_final_query` modes.
- Added route, mailbox, TTL, confidence, write-rate, and ES reward-variance diagnostics.
- Added architecture and curriculum interventions targeted at the long-horizon collapse.

## What Surprised Me

- Benchmark B v1 is harsher than the phase-1 report implied. The model can be pushed to near-perfect delayed routing while still staying at chance accuracy, so routing discovery is not the dominant bottleneck there.
- The adaptive delay-preservation gate introduces a second collapse mode: the model can preserve state almost perfectly while never writing the trigger payload.
- On Benchmark B v2, delayed behavior can appear transiently on training batches without surviving validation at all. The generalization gap is much larger than the phase-1 summary suggested.

## Dead Ends / Expensive Paths

- Repeatedly improving route supervision on Benchmark B v1 does not buy useful task accuracy once the content pathway is already failing.
- Delay-write auxiliary targets can make train-time behavior look more plausible while leaving validation completely collapsed.
- Hybrid ES can optimize the training-time routing pattern very effectively, but that does not automatically translate into a usable long-horizon policy.

## Current Bottleneck Hypothesis

- Benchmark B v1 mainly exposes a content-memory bottleneck, not a router-search bottleneck.
- Benchmark B v2 exposes a train-to-validation generalization failure on delayed retrieval.
- The current packet-memory path is still too weak or too local: it can preserve state or route correctly on train, but it does not robustly encode and retrieve the sparse trigger payload under the hard-routing policy.

## Results That Look Robust So Far

- Benchmark B v1 requires real delay under the current objective, yet routing alone is not enough.
- Benchmark B v2 improves the benchmark design by making adaptive routing necessary.
- Mask-based curriculum helps raw accuracy on Benchmark B v2, but mostly by preserving the `easy_exit` mode rather than unlocking successful delayed retrieval.
  - 3-seed baseline: `0.320 +/- 0.001`
  - 3-seed mask-curriculum: `0.445 +/- 0.005`
- Hybrid ES can optimize the training-time route pattern much more strongly than the validation/test outcome suggests. In the completed `pop64` write-aux run, train generations reached perfect route match with heavy delay, while the final test checkpoint still collapsed to `easy_exit=1.0` and `delay_rate=0.0`.
- The promoted `T=256` main runs agree across methods:
  - hard-ST h256: `0.4258`, `delay_rate ~= 0.0008`, `early_exit ~= 0.9984`
  - hybrid ES h256: `0.4225`, `delay_rate = 0.0`, `early_exit = 1.0`
- The promoted `T=256` hybrid main rerun shows the same failure at a harder horizon: delayed train behavior survives well past the first hundred warmstart steps, warmstart validation checkpoints at steps `39`, `79`, `119`, and `139` remain essentially pure immediate exit, and hybrid-ES validation checkpoints at generations `11`, `23`, and `35` also stay collapsed.

## What The Next Worker Should Try First If Phase 2 Stays Negative

- Strengthen the content memory itself rather than the router:
  - explicit trigger-state storage separate from the delay carry path
  - multi-slot or keyed mailbox rather than a single recurrent carry
  - retrieval-aware auxiliary objective tied to payload reconstruction, not just route behavior
- If hybrid ES is revisited, search over the part of the model that actually controls what gets written into memory, not only the route head.

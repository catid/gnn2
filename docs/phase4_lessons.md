# Phase 4 Lessons

- The phase-3 failure was narrower than it looked. The wait signal was already
  linearly decodable from packet and memory features, so the bottleneck was not
  just representational capacity.
- A direct control-to-router path matters more than a generic increase in latent
  width.
- Sticky control works. Set-clear control does not, at least on this benchmark
  and training schedule.
- Pure auxiliary control supervision is too weak.
- Pure anti-exit pressure is too weak.
- `sticky_aux_router2` is a real single-seed win, not an eval mirage. The
  larger-batch re-audit still gives about `0.50` final-query accuracy with real
  waiting behavior.
- The dim8 sticky+anti-exit family is still useful because it proves the policy
  can be pushed much later into the sequence when the control channel is strong
  enough.
- The promoted seed panel did not hold. Both promoted families regressed back to
  near-chance final-query accuracy with zero route match.
- That means the real phase-4 bottleneck is training robustness. The model can
  encode and sometimes use the control state, but the current objective and
  optimization path do not make that behavior stable across seeds.
- Probe-perfect decodability is not enough. The router can still ignore a clean
  control signal.
- `router2 + anti_exit` looked plausible and failed. It mostly destroyed the
  benefit of `router2` instead of combining the strengths.
- `router4` also failed. More control-to-router scale is not monotonic.
- The multi-GPU hybrid path had a real systems bug: rank 1 could time out at the
  warmstart barrier while rank 0 was still training. Phase 4 fixed that by
  adding an explicit distributed timeout before re-running hybrid ES.
- Scratch hybrid ES is still a bad discovery method on this problem. Its own
  warmstart relearned the old immediate-exit failure.
- Resume-based hybrid ES is the interesting result. Starting from the best
  working hard-ST checkpoint, ES improved the checkpoint from a decent but
  imperfect final-query controller to a perfect one on the same Benchmark B v2
  configuration.
- So the right ES question is no longer "can ES discover the controller from
  scratch?" The more promising question is "can ES reliably polish a controller
  once gradient training has already found the right basin?"
- The next worker should prioritize stabilizing the controller, not adding more
  memory width. The strongest next idea is a factorized wait controller or other
  route-head design that makes "must keep waiting" a first-class persistent
  decision rather than one logit configuration inside a 3-way router. After
  that, run a proper multi-seed ES-polish panel from working checkpoints.

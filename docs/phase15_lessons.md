# Phase 15 Lessons

## Status

Complete.

## Working Notes

- A richer content path is not automatically better. The multi-slot family improved summary-time flexibility but still collapsed onto the same confirm ceiling once reruns and locked confirms were complete.
- Route isolation held. The new sidecar and multi-slot paths did not reopen the old shortcut instability story, which means the phase-15 failures are scientifically cleaner than the old phase-12/13 failures.
- Sidecar memory is the more justified richer-path direction than multi-slot channels. The best sidecar families beat the best multi-slot family on confirm-time held-content metrics, even though the margin is still modest.
- Stronger training contracts matter only after the richer path exists. The plain richer-path baselines (`18026`, `18035`, `18036`, `18059`, `18060`) all closed as stable-ceiling results, while `18052` and `18057` produced the first small but real confirm lifts.
- Hard-slice selection still matters, but it is not enough by itself. `18052` is better than the plain teacher-on-sidecar branches, yet the gain is still incremental.
- Dual-anchor training matters more in phase 15 than it did in phase 14, but it is still not decisive. `18057` is the cleanest fully paneled phase-15 result so far, and it improves confirm content slightly, but not by enough to call the ceiling broken.
- Secondary-source summary wins can be real and still not portable. `18222` reproduced exactly on rerun, but bounded confirm still fell back to the same weak medium-source regime, so the portability problem is not just seed noise.
- Clean negative controls stayed clean even with the richer path. Both `18291` and `18292` remained properly negative on `1879`, which makes the phase-15 sidecar lifts easier to trust.
- The remaining bottleneck already looks narrower than “content path architecture” in the broad sense. Width helped a little, supervision helped a little, but neither changed the regime dramatically. The likely next bottleneck is how content gets written and retrieved inside the isolated path, not just how many readout channels exist.

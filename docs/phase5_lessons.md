# Phase 5 Lessons

- The strongest new result is not another hard-ST seed. It is the confirmed
  medium-basin ES rescue from `seed950`, including an exact same-seed rerun and
  confirmation split.
- Medium-basin ES rescue is mostly controller search. Router-only ES is already
  enough there.
- Weak-basin ES rescue is not a routing failure. It reaches perfect route match
  and correct exit timing, then tops out on task accuracy because the content
  representation is weaker.
- Adapters are not universally necessary. They matter when the basin is weak,
  but not when the starting checkpoint already contains a decent representation.
- The direct release-gate hard-ST family was worth trying, but it still did not
  discover the correct final-query behavior. The `finalquery-only` slice dying
  at chance was the clearest negative result of this continuation.
- The phase-4 statement “ES can polish” was too weak. The current better
  statement is:
  ES can fully rescue medium basins, partially rescue weak basins, and the
  remaining weak-basin error is mostly content quality after routing is fixed.
- The next worker should not start with another large hard-ST sweep. The most
  leveraged next step is to exploit the new separation between route rescue and
  content rescue.
- First thing to try next:
  ES route rescue from a weak checkpoint, then short gradient-only content
  refinement with routing frozen.
- Fragile result:
  hard-ST direct-gate discovery. It still looks brittle and did not survive the
  simplified slice control.
- Robust result:
  medium-basin ES rescue from `seed950`. It reproduced exactly and survived the
  confirmation split.

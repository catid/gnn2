# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

## Project Preferences

- Use `uv venv` for local environment management.
- Keep Python pinned to `3.12` via `.python-version` and project metadata.
- Prefer the latest PyTorch nightly builds from the CUDA 13.0 index.
- When rebuilding native code or extensions, use 16-way parallelism:
  `MAX_JOBS=16`, `CMAKE_BUILD_PARALLEL_LEVEL=16`, and `MAKEFLAGS=-j16`.
- Track substantive work in `bd` instead of ad hoc notes.

## EGGROLL Reference Discipline

When implementing, discussing, or evaluating any EGGROLL-style method in this repo, always consult the
local paper and the cloned reference code first:

- `references/eggroll_paper.pdf`
- `references/eggroll_paper.md`
- `references/HyperscaleES/README.org`
- `references/HyperscaleES/eggroll.ipynb`
- `references/HyperscaleES/src/hyperscalees/noiser/eggroll.py`
- `references/HyperscaleES/src/hyperscalees/noiser/eggroll_baseline_subtraction.py`
- `references/HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py`
- `references/HyperscaleES/tests/end_to_end_test.py`
- `docs/eggroll_reference_alignment.md`

Rules:

- Treat our PyTorch implementation as `EGGROLL-inspired` unless a behavior is directly verified against the
  paper and reference repo.
- Preserve the core transferable ideas: low-rank perturbations, forward-only population evaluation,
  antithetic sampling, deterministic seed-based perturbation reconstruction, multi-GPU population execution,
  and direct objective optimization with compute-aware penalties.
- Do not claim exact fidelity to the paper's systems tricks, kernels, shared-activation implementation, vLLM
  stack, RWKV setup, or full experiment suite unless we actually implement and validate those pieces.
- Explicitly document deviations from the original paper and the HyperscaleES code whenever we make them.

## Aline OneContext

When the user asks about existing code, features, past decisions, debugging context,
or anything that might have been discussed in previous conversations, proactively
use the `onecontext` skill to search Aline history before answering.

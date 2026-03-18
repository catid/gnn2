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

## Aline OneContext

When the user asks about existing code, features, past decisions, debugging context,
or anything that might have been discussed in previous conversations, proactively
use the `onecontext` skill to search Aline history before answering.

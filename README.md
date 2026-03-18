# gnn2

Bootstrap for a Python 3.12 + `uv` + PyTorch nightly CUDA 13.0 workspace.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

## Rebuild Parallelism

When building native extensions or rebuilding local packages, keep rebuilds at 16
threads:

```bash
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export MAKEFLAGS=-j16
```

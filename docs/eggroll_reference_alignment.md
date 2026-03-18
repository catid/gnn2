# EGGROLL Reference Alignment

## Scope

This note compares the downloaded EGGROLL paper in `references/eggroll_paper.pdf` / `references/eggroll_paper.md`
against the cloned reference implementation in `references/HyperscaleES`.

The goal is not to claim a full artifact audit. The goal is to establish what is clearly implemented in the
reference repo, what matches the paper's core method, and what our later PyTorch packet-routing work must
treat as an approximation.

## Bottom Line

The `HyperscaleES` repo matches the paper on the core EGGROLL mechanism:

- low-rank perturbations of the form `E = (1 / sqrt(r)) A B^T`
- forward-only population evaluation
- antithetic positive / negative perturbation signs
- seed-based perturbation reconstruction instead of storing full noise tensors
- batched low-rank matmul application during forward evaluation
- weighted population aggregation into a full-rank update
- a distributed multi-GPU evaluation/update path

The repo does **not** look like a polished one-to-one reproduction package for every figure, appendix system
detail, or theorem statement in the paper. It is explicitly a research preview.

## Strong Matches

### 1. Low-rank perturbations are the main mechanism

The paper's section 4 defines the search matrix as `E = (1 / sqrt(r)) A B^T` and describes EGGROLL as a
low-rank ES approximation that still aggregates into a higher-rank overall update.

The repo implements the same structure in `references/HyperscaleES/src/hyperscalees/noiser/eggroll.py`:

- `get_lora_update_params(...)` samples a concatenated `(a + b, rank)` random tensor and splits it into `A`
  and `B`.
- `do_mm(...)` and `do_Tmm(...)` apply the perturbation as low-rank products instead of materializing a dense
  `E`.
- `_simple_lora_update(...)` reconstructs the weighted update by `einsum` over population members.

This is the clearest code-level match to the paper's core idea.

### 2. Antithetic sampling and deterministic reconstruction are present

The paper highlights seed-based reconstruction and hardware-efficient population evaluation.

In `references/HyperscaleES/src/hyperscalees/noiser/eggroll.py`:

- perturbation sign flips are driven by thread parity via `thread_id % 2`
- perturbation identity is reconstructed from `(epoch, thread_id)` plus a base key
- `noise_reuse` controls how often the effective random seed changes

This is consistent with the paper's emphasis on deterministic noise regeneration and antithetic ES structure.

### 3. Shared-base forward computation is implemented at the matmul level

The paper's hardware section states that a forward pass can be written as a shared base matmul plus cheap
low-rank per-perturbation work.

The repo mirrors that in `references/HyperscaleES/src/hyperscalees/noiser/eggroll.py`:

- `do_mm(...)`: `x @ param.T + x @ B @ A.T`
- `do_Tmm(...)`: `x @ param + x @ A @ B.T`

This is the code analogue of the paper's `u_i M^T + (sigma / sqrt(r)) (u_i B_i) A_i^T` decomposition.

### 4. The weighted ES update is computed without storing dense perturbations

The paper notes that the key update term can be computed efficiently without explicitly materializing each
perturbation.

The repo does the same in `_simple_lora_update(...)` with:

- a population batch of `A`
- a population batch of `B`
- a weighted contraction via `jnp.einsum('nir,njr->ij', A, B) / num_envs`

That is directly aligned with the paper's description of aggregating low-rank perturbations into the update.

### 5. Multi-GPU population evaluation is present

The paper claims large-scale distributed evaluation. The repo contains a clear multi-GPU execution path in
`references/HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py`:

- `jax.distributed.initialize(...)`
- `shard_map(...)`
- device mesh construction
- per-device generation counts
- compiled generation and update kernels

That is sufficient evidence that the repo is designed for distributed population evaluation, even though our
later PyTorch work will implement a simpler two-GPU version.

### 6. A baseline-subtraction / trust-region variant exists

The paper discusses EGGROLL variants and approximation choices. The repo includes a concrete variant in
`references/HyperscaleES/src/hyperscalees/noiser/eggroll_baseline_subtraction.py` with:

- grouped fitness handling
- zero-noise baseline slots inside each group
- fitness normalization relative to the baseline
- `trust_region_norm` clipping through Optax

This does not mean every paper variant is fully surfaced, but it does show the repo includes more than one
EGGROLL-style update rule.

## Partial Matches And Limits

### 1. The repo is a research preview, not a full artifact bundle

`references/HyperscaleES/README.org` explicitly labels the codebase a research preview and points readers to
`eggroll.ipynb` and `tests/end_to_end_test.py` for worked examples. That lowers confidence that every paper
claim maps cleanly to a single script or config.

### 2. Core method alignment is strong; experiment-by-experiment alignment is weaker

I found strong evidence for:

- the low-rank noiser
- the ES update path
- the multi-GPU evolution runner
- LLM-oriented experiment scaffolding

I did **not** fully verify, line by line, every item below against a single runnable artifact:

- every figure in the paper
- every theorem / appendix derivation
- the exact shared-activation systems kernels implied by the writeup
- every RWKV, GRPO, vLLM, or pure-int8 systems detail claimed in the paper

Some of those appear spread across notebooks, scripts, or adjacent repos like `nano-egg`.

### 3. The repo is JAX-first, while our implementation will be PyTorch

This matters. A faithful *idea-level* port is realistic; an exact *implementation-level* reproduction is not.

Our later PyTorch code should therefore claim:

- "EGGROLL-inspired"
- "core-method approximation"
- "aligned to the original paper and HyperscaleES reference code"

It should **not** claim:

- exact reproduction of HyperscaleES kernels
- exact reproduction of the shared-activation systems stack
- exact reproduction of paper-scale RWKV / reasoning infrastructure

unless we actually build and verify those pieces.

## Requirements For This Repo

When implementing EGGROLL-style methods here, the minimum reference set should be:

- `references/eggroll_paper.pdf`
- `references/eggroll_paper.md`
- `references/HyperscaleES/README.org`
- `references/HyperscaleES/eggroll.ipynb`
- `references/HyperscaleES/src/hyperscalees/noiser/eggroll.py`
- `references/HyperscaleES/src/hyperscalees/noiser/eggroll_baseline_subtraction.py`
- `references/HyperscaleES/llm_experiments/general_do_evolution_multi_gpu.py`
- `references/HyperscaleES/tests/end_to_end_test.py`

For our packet-routing project specifically, we should preserve these transferable ideas:

- real hard routing in the forward pass
- forward-only ES evaluation for routing/control parameters
- low-rank perturbations for evolved matrices
- antithetic sampling
- deterministic seed-based perturbation reconstruction
- multi-GPU population splitting
- direct optimization of task score with compute-aware penalties

We should explicitly document every deliberate deviation from the paper and the JAX reference code.

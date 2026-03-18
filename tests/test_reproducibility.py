from __future__ import annotations

import torch

from src.data import build_benchmark
from src.models import PacketRoutingModel
from src.train.run import seed_everything


def benchmark_config() -> dict:
    return {
        "name": "mixed_oracle_routing",
        "num_nodes": 4,
        "obs_dim": 20,
        "num_classes": 2,
        "seq_len": 2,
        "noise_std": 0.05,
        "train_seed": 11,
        "val_seed": 101,
        "test_seed": 1001,
    }


def test_benchmark_sampling_is_reproducible() -> None:
    benchmark = build_benchmark(benchmark_config())
    batch_a = benchmark.sample_batch(batch_size=8, split="train", step=3, device="cpu")
    batch_b = benchmark.sample_batch(batch_size=8, split="train", step=3, device="cpu")

    assert torch.equal(batch_a.observations, batch_b.observations)
    assert torch.equal(batch_a.labels, batch_b.labels)
    assert torch.equal(batch_a.oracle_hops, batch_b.oracle_hops)
    assert torch.equal(batch_a.oracle_delays, batch_b.oracle_delays)


def test_model_initialization_is_reproducible_for_fixed_seed() -> None:
    config = {
        "num_nodes": 4,
        "obs_dim": 20,
        "hidden_dim": 16,
        "num_classes": 2,
        "max_internal_steps": 4,
        "max_total_steps": 32,
        "adapter_rank": 0,
        "packet_memory_slots": 2,
        "packet_memory_dim": 8,
    }
    seed_everything(123)
    model_a = PacketRoutingModel(config)
    seed_everything(123)
    model_b = PacketRoutingModel(config)

    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name in params_a:
        assert torch.equal(params_a[name], params_b[name]), name

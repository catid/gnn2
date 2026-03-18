from __future__ import annotations

import torch

from src.data import build_benchmark
from src.data.benchmarks import BenchmarkBatch, LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY
from src.models import PacketRoutingModel
from src.train.run import build_control_controls, seed_everything


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


def test_build_control_controls_marks_wait_window_for_final_query_mode() -> None:
    batch = BenchmarkBatch(
        observations=torch.zeros(2, 8, 2, 4),
        labels=torch.zeros(2, dtype=torch.long),
        modes=torch.tensor([LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY, 0], dtype=torch.long),
        oracle_hops=torch.zeros(2, dtype=torch.long),
        oracle_delays=torch.zeros(2, dtype=torch.long),
        oracle_exit_time=torch.zeros(2, dtype=torch.long),
        oracle_depth=torch.zeros(2, dtype=torch.long),
        metadata={
            "trigger_time": torch.tensor([2, 0], dtype=torch.long),
            "query_time": torch.tensor([7, 0], dtype=torch.long),
            "needs_final_query": torch.tensor([1, 0], dtype=torch.long),
        },
    )

    targets, target_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
        batch,
        {"control_state_weight": 0.4, "anti_exit_weight": 0.7},
        split="train",
    )

    assert control_weight == 0.4
    assert anti_exit_weight == 0.7
    assert targets is not None
    assert target_mask is not None
    assert anti_exit_mask is not None
    assert torch.equal(targets[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(target_mask[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(anti_exit_mask[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(targets[1], torch.zeros(8))

from __future__ import annotations

import torch

from src.data import build_benchmark
from src.data.benchmarks import (
    BenchmarkBatch,
    LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY,
    LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT,
)
from src.models import PacketRoutingModel
from src.train.run import (
    build_control_controls,
    build_release_controls,
    build_wait_controls,
    seed_everything,
)


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
        "confirm_seed": 2001,
    }


def test_benchmark_sampling_is_reproducible() -> None:
    benchmark = build_benchmark(benchmark_config())
    batch_a = benchmark.sample_batch(batch_size=8, split="train", step=3, device="cpu")
    batch_b = benchmark.sample_batch(batch_size=8, split="train", step=3, device="cpu")

    assert torch.equal(batch_a.observations, batch_b.observations)
    assert torch.equal(batch_a.labels, batch_b.labels)
    assert torch.equal(batch_a.oracle_hops, batch_b.oracle_hops)
    assert torch.equal(batch_a.oracle_delays, batch_b.oracle_delays)


def test_confirmation_split_uses_its_own_seed_reproducibly() -> None:
    benchmark = build_benchmark(benchmark_config())
    confirm_a = benchmark.sample_batch(batch_size=8, split="confirm", step=2, device="cpu")
    confirm_b = benchmark.sample_batch(batch_size=8, split="confirm", step=2, device="cpu")
    test_batch = benchmark.sample_batch(batch_size=8, split="test", step=2, device="cpu")

    assert torch.equal(confirm_a.observations, confirm_b.observations)
    assert torch.equal(confirm_a.labels, confirm_b.labels)
    assert not torch.equal(confirm_a.observations, test_batch.observations)


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


def test_build_control_controls_can_exclude_query_step_from_wait_latch() -> None:
    batch = BenchmarkBatch(
        observations=torch.zeros(1, 8, 2, 4),
        labels=torch.zeros(1, dtype=torch.long),
        modes=torch.tensor([LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY], dtype=torch.long),
        oracle_hops=torch.zeros(1, dtype=torch.long),
        oracle_delays=torch.zeros(1, dtype=torch.long),
        oracle_exit_time=torch.zeros(1, dtype=torch.long),
        oracle_depth=torch.zeros(1, dtype=torch.long),
        metadata={
            "trigger_time": torch.tensor([2], dtype=torch.long),
            "query_time": torch.tensor([7], dtype=torch.long),
            "needs_final_query": torch.tensor([1], dtype=torch.long),
        },
    )

    targets, target_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
        batch,
        {
            "control_state_weight": 0.4,
            "anti_exit_weight": 0.7,
            "control_target_scope": "final_query_wait_only",
        },
        split="train",
    )

    assert control_weight == 0.4
    assert anti_exit_weight == 0.7
    assert targets is not None
    assert target_mask is not None
    assert anti_exit_mask is not None
    assert torch.equal(targets[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(target_mask[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(anti_exit_mask[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))


def test_build_control_controls_oracle_delayed_only_tracks_delay_actions() -> None:
    batch = BenchmarkBatch(
        observations=torch.zeros(2, 4, 2, 4),
        labels=torch.zeros(2, dtype=torch.long),
        modes=torch.tensor([LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY, LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT], dtype=torch.long),
        oracle_hops=torch.zeros(2, dtype=torch.long),
        oracle_delays=torch.zeros(2, dtype=torch.long),
        oracle_exit_time=torch.zeros(2, dtype=torch.long),
        oracle_depth=torch.zeros(2, dtype=torch.long),
        oracle_actions=torch.tensor(
            [
                [2, 2, 2, 1],
                [2, 2, 1, 0],
            ],
            dtype=torch.long,
        ),
        oracle_action_mask=torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        ),
        metadata={
            "trigger_time": torch.tensor([0, 0], dtype=torch.long),
            "query_time": torch.tensor([3, 0], dtype=torch.long),
            "needs_final_query": torch.tensor([1, 0], dtype=torch.long),
        },
    )

    targets, target_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
        batch,
        {
            "control_state_weight": 0.25,
            "anti_exit_weight": 0.1,
            "control_target_scope": "oracle_delayed_only",
        },
        split="train",
    )

    assert control_weight == 0.25
    assert anti_exit_weight == 0.1
    assert targets is not None
    assert target_mask is not None
    assert anti_exit_mask is not None
    assert torch.equal(targets[0], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(targets[1], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(target_mask[1], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(anti_exit_mask[0], torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_build_wait_controls_targets_wait_until_query_then_act() -> None:
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

    targets, target_mask, wait_weight, wait_positive_weight, wait_negative_weight = build_wait_controls(
        batch,
        {"wait_loss_weight": 0.5},
        split="train",
    )

    assert wait_weight == 0.5
    assert wait_positive_weight == 1.0
    assert wait_negative_weight == 1.0
    assert targets is not None
    assert target_mask is not None
    assert torch.equal(targets[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(target_mask[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(targets[1], torch.zeros(8))
    assert torch.equal(target_mask[1], torch.zeros(8))


def test_build_wait_controls_oracle_all_uses_oracle_delay_actions() -> None:
    batch = BenchmarkBatch(
        observations=torch.zeros(2, 4, 2, 4),
        labels=torch.zeros(2, dtype=torch.long),
        modes=torch.tensor([LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY, LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT], dtype=torch.long),
        oracle_hops=torch.zeros(2, dtype=torch.long),
        oracle_delays=torch.zeros(2, dtype=torch.long),
        oracle_exit_time=torch.zeros(2, dtype=torch.long),
        oracle_depth=torch.zeros(2, dtype=torch.long),
        oracle_actions=torch.tensor(
            [
                [2, 2, 2, 1],
                [2, 2, 1, 0],
            ],
            dtype=torch.long,
        ),
        oracle_action_mask=torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        ),
        metadata={
            "trigger_time": torch.tensor([0, 0], dtype=torch.long),
            "query_time": torch.tensor([3, 0], dtype=torch.long),
            "needs_final_query": torch.tensor([1, 0], dtype=torch.long),
        },
    )

    targets, target_mask, wait_weight, wait_positive_weight, wait_negative_weight = build_wait_controls(
        batch,
        {"wait_loss_weight": 0.5, "wait_target_scope": "oracle_all"},
        split="train",
    )

    assert wait_weight == 0.5
    assert wait_positive_weight == 1.0
    assert wait_negative_weight == 1.0
    assert targets is not None
    assert target_mask is not None
    assert torch.equal(targets[0], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(targets[1], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(target_mask[1], torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_build_release_controls_can_focus_on_final_query_mode() -> None:
    batch = BenchmarkBatch(
        observations=torch.zeros(2, 4, 2, 4),
        labels=torch.zeros(2, dtype=torch.long),
        modes=torch.tensor([LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY, LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT], dtype=torch.long),
        oracle_hops=torch.zeros(2, dtype=torch.long),
        oracle_delays=torch.zeros(2, dtype=torch.long),
        oracle_exit_time=torch.zeros(2, dtype=torch.long),
        oracle_depth=torch.zeros(2, dtype=torch.long),
        oracle_actions=torch.tensor(
            [
                [2, 2, 2, 1],
                [2, 2, 1, 0],
            ],
            dtype=torch.long,
        ),
        oracle_action_mask=torch.ones(2, 4),
        metadata={
            "needs_final_query": torch.tensor([1, 0], dtype=torch.long),
        },
    )

    targets, target_mask, release_weight, positive_weight = build_release_controls(
        batch,
        {
            "release_loss_weight": 0.25,
            "release_target_scope": "final_query_only",
            "release_positive_weight": 4.0,
        },
        split="train",
    )

    assert release_weight == 0.25
    assert positive_weight == 4.0
    assert targets is not None
    assert target_mask is not None
    assert torch.equal(targets[0], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(targets[1], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert torch.equal(target_mask[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(target_mask[1], torch.zeros(4))

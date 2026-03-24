from __future__ import annotations

import torch
import yaml

from src.data import build_benchmark
from src.data.benchmarks import (
    BenchmarkBatch,
    LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY,
    LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT,
)
from src.models import PacketRoutingModel
from src.train.run import (
    build_task_sample_weights,
    build_control_controls,
    configure_trainable_parameters,
    current_routing_cfg,
    build_release_controls,
    build_wait_controls,
    load_checkpoint,
    load_auxiliary_eval_benchmarks,
    load_auxiliary_train_benchmarks,
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


def test_current_routing_cfg_applies_boolean_and_scalar_schedules() -> None:
    cfg = {
        "force_oracle_actions": True,
        "force_oracle_actions_until_step": 5,
        "exit_mask_final_query_only": True,
        "exit_mask_final_query_only_start_step": 3,
        "oracle_route_weight_start": 0.4,
        "oracle_route_weight_end": 0.1,
        "oracle_route_weight_schedule_steps": 10,
    }

    step0 = current_routing_cfg(cfg, 0)
    step4 = current_routing_cfg(cfg, 4)
    step7 = current_routing_cfg(cfg, 7)

    assert step0["force_oracle_actions"] is True
    assert step0["exit_mask_final_query_only"] is False
    assert abs(step0["oracle_route_weight"] - 0.4) < 1e-6

    assert step4["force_oracle_actions"] is True
    assert step4["exit_mask_final_query_only"] is True
    assert abs(step4["oracle_route_weight"] - 0.28) < 1e-6

    assert step7["force_oracle_actions"] is False
    assert step7["exit_mask_final_query_only"] is True
    assert abs(step7["oracle_route_weight"] - 0.19) < 1e-6


def test_configure_trainable_parameters_respects_allowlist_and_freeze_rules() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 2,
            "packet_memory_slots": 2,
            "packet_memory_dim": 3,
            "control_state_dim": 2,
        }
    )

    summary = configure_trainable_parameters(
        model,
        {
            "trainable_prefixes": ["memory_", "readout", "core.packet_adapter"],
            "freeze_prefixes": ["memory_read_gate"],
        },
    )

    enabled = set(summary["trainable_parameter_names"])
    assert "readout.0.weight" in enabled
    assert "core.packet_adapter.down.weight" in enabled
    assert "memory_write_gate.weight" in enabled
    assert "memory_read_gate.weight" not in enabled
    assert "core.router_mlp.1.weight" not in enabled
    assert summary["trainable_parameter_count"] < summary["total_parameter_count"]


def test_build_task_sample_weights_can_upweight_final_query_examples() -> None:
    batch = BenchmarkBatch(
        observations=torch.zeros(3, 4, 2, 4),
        labels=torch.zeros(3, dtype=torch.long),
        modes=torch.tensor([LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY, LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT, 0], dtype=torch.long),
        oracle_hops=torch.zeros(3, dtype=torch.long),
        oracle_delays=torch.zeros(3, dtype=torch.long),
        oracle_exit_time=torch.zeros(3, dtype=torch.long),
        oracle_depth=torch.zeros(3, dtype=torch.long),
        metadata={
            "needs_final_query": torch.tensor([1, 0, 0], dtype=torch.long),
        },
    )

    weights = build_task_sample_weights(
        batch,
        {"final_query_weight": 3.0, "non_final_query_weight": 0.5},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert weights is not None
    assert torch.equal(weights, torch.tensor([3.0, 0.5, 0.5]))


def test_load_auxiliary_train_benchmarks_resolves_relative_configs_and_weight_overrides(tmp_path) -> None:
    aux_path = tmp_path / "confirm_locked.yaml"
    aux_path.write_text(yaml.safe_dump({"benchmark": benchmark_config()}))
    main_path = tmp_path / "main.yaml"
    main_path.write_text("training: {}\n")

    sources = load_auxiliary_train_benchmarks(
        {
            "config_path": str(main_path),
            "training": {
                "batch_size": 8,
                "final_query_weight": 3.0,
                "non_final_query_weight": 0.5,
                "auxiliary_train_benchmarks": [
                    {
                        "name": "full_locked",
                        "config": "confirm_locked.yaml",
                        "split": "confirm",
                        "loss_weight": 0.75,
                        "final_query_weight": 5.0,
                    }
                ],
            },
        }
    )

    assert len(sources) == 1
    source = sources[0]
    assert source.name == "full_locked"
    assert source.split == "confirm"
    assert source.batch_size == 8
    assert source.loss_weight == 0.75
    assert source.task_weight_cfg == {"final_query_weight": 5.0, "non_final_query_weight": 0.5}

    confirm_a = source.benchmark.sample_batch(batch_size=4, split="confirm", step=1, device="cpu")
    confirm_b = source.benchmark.sample_batch(batch_size=4, split="confirm", step=1, device="cpu")
    assert torch.equal(confirm_a.observations, confirm_b.observations)


def test_load_auxiliary_eval_benchmarks_resolves_relative_configs_and_batch_overrides(tmp_path) -> None:
    aux_path = tmp_path / "confirm_locked.yaml"
    aux_path.write_text(yaml.safe_dump({"benchmark": benchmark_config()}))
    main_path = tmp_path / "main.yaml"
    main_path.write_text("training: {}\n")

    sources = load_auxiliary_eval_benchmarks(
        {
            "config_path": str(main_path),
            "training": {
                "batch_size": 8,
                "val_batch_size": 16,
                "val_batches": 6,
                "selection_eval_benchmarks": [
                    {
                        "name": "full_locked",
                        "config": "confirm_locked.yaml",
                        "split": "confirm",
                        "batch_size": 12,
                        "num_batches": 4,
                    }
                ],
            },
        }
    )

    assert len(sources) == 1
    source = sources[0]
    assert source.name == "full_locked"
    assert source.benchmark_name == "mixed_oracle_routing"
    assert source.split == "confirm"
    assert source.batch_size == 12
    assert source.num_batches == 4

    confirm_a = source.benchmark.sample_batch(batch_size=4, split="confirm", step=1, device="cpu")
    confirm_b = source.benchmark.sample_batch(batch_size=4, split="confirm", step=1, device="cpu")
    assert torch.equal(confirm_a.observations, confirm_b.observations)


def test_load_checkpoint_strict_false_skips_shape_mismatches(tmp_path) -> None:
    source = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "packet_memory_slots": 2,
            "packet_memory_dim": 3,
        }
    )
    target = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "packet_memory_slots": 4,
            "packet_memory_dim": 3,
        }
    )
    with torch.no_grad():
        source.sink_proj.weight.fill_(0.25)
        source.memory_read_slots.bias.fill_(1.0)

    ckpt = tmp_path / "shape_mismatch.pt"
    torch.save({"model": source.state_dict(), "step": 0, "extra": {}}, ckpt)

    load_checkpoint(ckpt, target, strict=False)

    assert torch.allclose(target.sink_proj.weight, torch.full_like(target.sink_proj.weight, 0.25))


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

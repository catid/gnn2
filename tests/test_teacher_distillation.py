from __future__ import annotations

from types import SimpleNamespace

import torch

from src.data import BenchmarkBatch
from src.data.benchmarks import (
    LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY,
    LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT,
    LONG_MEMORY_MODE_EASY_EXIT,
)
from src.train.run import (
    TeacherDistillation,
    compute_teacher_distillation_loss,
    teacher_step_controls,
    teacher_sample_weights,
)


def _benchmark_batch() -> BenchmarkBatch:
    return BenchmarkBatch(
        observations=torch.zeros(3, 4, 2, 5),
        labels=torch.tensor([0, 1, 2]),
        modes=torch.tensor(
            [
                LONG_MEMORY_MODE_EASY_EXIT,
                LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT,
                LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY,
            ]
        ),
        oracle_hops=torch.zeros(3, dtype=torch.long),
        oracle_delays=torch.zeros(3, dtype=torch.long),
        oracle_exit_time=torch.zeros(3, dtype=torch.long),
        oracle_depth=torch.zeros(3, dtype=torch.long),
        metadata={
            "needs_final_query": torch.tensor([0.0, 0.0, 1.0]),
            "trigger_time": torch.tensor([0, 1, 1]),
            "query_time": torch.tensor([0, 2, 3]),
        },
    )


def test_teacher_sample_weights_respect_scope() -> None:
    batch = _benchmark_batch()
    dtype = batch.observations.dtype
    device = batch.observations.device

    all_weights = teacher_sample_weights(batch, "all", device=device, dtype=dtype)
    final_query_weights = teacher_sample_weights(batch, "final_query_only", device=device, dtype=dtype)
    delayed_only_weights = teacher_sample_weights(batch, "delayed_only", device=device, dtype=dtype)

    assert torch.allclose(all_weights, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(final_query_weights, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(delayed_only_weights, torch.tensor([0.0, 1.0, 1.0]))


def test_compute_teacher_distillation_loss_uses_matching_traces() -> None:
    batch = _benchmark_batch()
    student_output = SimpleNamespace(
        logits=torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.0, -0.5], [0.1, 0.2, 0.3]]),
        trace={
            "router_probs": torch.full((3, 4, 3), 1 / 3),
            "control_prob": torch.zeros(3, 4),
            "release_prob": torch.zeros(3, 4),
            "memory_read_state": torch.zeros(3, 4, 2),
        },
    )
    teacher_output = SimpleNamespace(
        logits=torch.tensor([[3.0, -1.0, -2.0], [0.0, 2.0, -1.0], [-0.5, 0.2, 1.5]]),
        trace={
            "router_probs": torch.tensor(
                [
                    [[0.1, 0.2, 0.7]] * 4,
                    [[0.2, 0.1, 0.7]] * 4,
                    [[0.05, 0.05, 0.9]] * 4,
                ],
                dtype=torch.float32,
            ),
            "control_prob": torch.tensor(
                [[0.0] * 4, [0.25] * 4, [0.95] * 4],
                dtype=torch.float32,
            ),
            "release_prob": torch.tensor(
                [[1.0] * 4, [0.5] * 4, [0.05] * 4],
                dtype=torch.float32,
            ),
            "memory_read_state": torch.ones(3, 4, 2),
        },
    )
    teacher = TeacherDistillation(
        model=None,  # type: ignore[arg-type]
        route_mode="hard",
        temperature=1.0,
        estimator="straight_through",
        distill_temperature=1.0,
        target_scope="final_query_only",
        logits_weight=0.5,
        route_weight=1.0,
        route_action_weight=0.5,
        control_prob_weight=0.25,
        release_prob_weight=0.25,
        wait_prob_weight=0.0,
        control_state_weight=0.0,
        memory_read_weight=0.1,
        start_step=0,
        stop_step=-1,
        scale_start=1.0,
        scale_end=1.0,
        scale_schedule_steps=0,
        dropout_prob_start=0.0,
        dropout_prob_end=0.0,
        dropout_schedule_steps=0,
    )

    loss, metrics = compute_teacher_distillation_loss(
        batch=batch,
        student_output=student_output,
        teacher_output=teacher_output,
        teacher=teacher,
        device=batch.observations.device,
    )

    assert loss.item() > 0.0
    assert metrics["teacher_distill_loss"] > 0.0
    assert metrics["teacher_logits_loss"] > 0.0
    assert metrics["teacher_route_loss"] > 0.0
    assert metrics["teacher_route_action_loss"] > 0.0
    assert metrics["teacher_control_prob_loss"] > 0.0
    assert metrics["teacher_release_prob_loss"] > 0.0
    assert metrics["teacher_memory_read_loss"] > 0.0


def test_teacher_step_controls_support_schedule_and_stop() -> None:
    teacher = TeacherDistillation(
        model=None,  # type: ignore[arg-type]
        route_mode="hard",
        temperature=1.0,
        estimator="straight_through",
        distill_temperature=1.0,
        target_scope="all",
        logits_weight=0.1,
        route_weight=0.0,
        route_action_weight=0.0,
        control_prob_weight=0.0,
        release_prob_weight=0.0,
        wait_prob_weight=0.0,
        control_state_weight=0.0,
        memory_read_weight=0.0,
        start_step=10,
        stop_step=20,
        scale_start=1.0,
        scale_end=0.2,
        scale_schedule_steps=10,
        dropout_prob_start=0.0,
        dropout_prob_end=0.5,
        dropout_schedule_steps=10,
    )

    assert teacher_step_controls(teacher, 0) == (0.0, 0.0)
    scale_mid, drop_mid = teacher_step_controls(teacher, 15)
    assert 0.2 < scale_mid < 1.0
    assert 0.0 < drop_mid < 0.5
    assert teacher_step_controls(teacher, 20) == (0.0, 0.0)

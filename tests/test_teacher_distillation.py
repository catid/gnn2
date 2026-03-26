from __future__ import annotations

from types import SimpleNamespace

import math
import torch

from src.data import BenchmarkBatch
from src.train.run import TeacherDistillation, compute_teacher_distillation_loss


def _teacher(**overrides) -> TeacherDistillation:
    base = dict(
        model=None,
        route_mode="hard",
        temperature=1.0,
        estimator="straight_through",
        distill_temperature=1.0,
        target_scope="all",
        logits_weight=0.0,
        route_weight=0.0,
        route_action_weight=0.0,
        control_prob_weight=0.0,
        release_prob_weight=0.0,
        wait_prob_weight=0.0,
        control_state_weight=0.0,
        memory_read_weight=0.0,
        factorized_content_weight=0.0,
        factorized_query_weight=0.0,
        start_step=0,
        stop_step=-1,
        scale_start=1.0,
        scale_end=1.0,
        scale_schedule_steps=0,
        dropout_prob_start=0.0,
        dropout_prob_end=0.0,
        dropout_schedule_steps=0,
    )
    base.update(overrides)
    return TeacherDistillation(**base)


def _batch() -> BenchmarkBatch:
    return BenchmarkBatch(
        observations=torch.zeros(2, 1, 1, 1),
        labels=torch.tensor([0, 1]),
        modes=torch.tensor([2, 2]),
        oracle_hops=torch.zeros(2, dtype=torch.long),
        oracle_delays=torch.zeros(2, dtype=torch.long),
        oracle_exit_time=torch.zeros(2, dtype=torch.long),
        oracle_depth=torch.zeros(2, dtype=torch.long),
        metadata={"needs_final_query": torch.tensor([1, 0])},
    )


def test_compute_teacher_distillation_loss_supports_factorized_content_hidden() -> None:
    batch = _batch()
    student_output = SimpleNamespace(
        logits=torch.zeros(2, 3),
        trace={"factorized_content_hidden": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
    )
    teacher_output = SimpleNamespace(
        logits=torch.zeros(2, 3),
        trace={"factorized_content_hidden": torch.tensor([[3.0, 2.0], [3.0, 4.0]])},
    )
    teacher = _teacher(factorized_content_weight=1.5, target_scope="final_query_only")

    loss, metrics = compute_teacher_distillation_loss(
        batch=batch,
        student_output=student_output,
        teacher_output=teacher_output,
        teacher=teacher,
        device=torch.device("cpu"),
        scale=1.0,
        dropout_prob=0.0,
    )

    # Sample 0 only: sum((1-3)^2, (2-2)^2) = 4.0, then weight 1.5.
    assert math.isclose(metrics["teacher_factorized_content_loss"], 4.0)
    assert math.isclose(metrics["teacher_distill_loss_raw"], 6.0)
    assert math.isclose(metrics["teacher_distill_loss"], 6.0)
    assert math.isclose(float(loss.item()), 6.0)


def test_compute_teacher_distillation_loss_supports_factorized_query_hidden() -> None:
    batch = _batch()
    student_output = SimpleNamespace(
        logits=torch.zeros(2, 3),
        trace={"factorized_query_hidden": torch.tensor([[0.0, 2.0], [5.0, 5.0]])},
    )
    teacher_output = SimpleNamespace(
        logits=torch.zeros(2, 3),
        trace={"factorized_query_hidden": torch.tensor([[2.0, 2.0], [5.0, 5.0]])},
    )
    teacher = _teacher(factorized_query_weight=0.5, target_scope="final_query_only")

    loss, metrics = compute_teacher_distillation_loss(
        batch=batch,
        student_output=student_output,
        teacher_output=teacher_output,
        teacher=teacher,
        device=torch.device("cpu"),
        scale=2.0,
        dropout_prob=0.0,
    )

    # Sample 0 only: sum((0-2)^2, (2-2)^2) = 4.0; weight 0.5 then scale 2.0 => 4.0.
    assert math.isclose(metrics["teacher_factorized_query_loss"], 4.0)
    assert math.isclose(metrics["teacher_distill_loss_raw"], 2.0)
    assert math.isclose(metrics["teacher_distill_loss"], 4.0)
    assert math.isclose(float(loss.item()), 4.0)

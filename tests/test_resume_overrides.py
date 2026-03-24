from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import math
import torch

from src.train.run import (
    apply_cli_overrides,
    apply_partial_init,
    initial_selection_score,
    resolve_resume_checkpoint,
    validation_score,
)


def test_apply_cli_overrides_preserves_config_resume_when_flag_missing() -> None:
    cfg = {"resume": "from_config.pt"}
    args = SimpleNamespace(resume=None, seed=None)

    updated = apply_cli_overrides(cfg, args)

    assert updated["resume"] == "from_config.pt"


def test_apply_cli_overrides_replaces_config_resume_when_flag_present() -> None:
    cfg = {"resume": "from_config.pt"}
    args = SimpleNamespace(resume="from_cli.pt", seed=None)

    updated = apply_cli_overrides(cfg, args)

    assert updated["resume"] == "from_cli.pt"


def test_apply_cli_overrides_preserves_config_seed_when_flag_missing() -> None:
    cfg = {"experiment": {"seed": 123}}
    args = SimpleNamespace(resume=None, seed=None)

    updated = apply_cli_overrides(cfg, args)

    assert updated["experiment"]["seed"] == 123


def test_apply_cli_overrides_replaces_config_seed_when_flag_present() -> None:
    cfg = {"experiment": {"seed": 123}}
    args = SimpleNamespace(resume=None, seed=456)

    updated = apply_cli_overrides(cfg, args)

    assert updated["experiment"]["seed"] == 456


def test_apply_cli_overrides_creates_experiment_mapping_for_seed_override() -> None:
    cfg = {}
    args = SimpleNamespace(resume=None, seed=789)

    updated = apply_cli_overrides(cfg, args)

    assert updated["experiment"]["seed"] == 789


def test_resolve_resume_checkpoint_supports_mapping() -> None:
    cfg = {"resume": {"checkpoint": "from_mapping.pt", "strict": False}, "resume_strict": True}

    checkpoint, strict = resolve_resume_checkpoint(cfg)

    assert checkpoint == "from_mapping.pt"
    assert strict is False


def test_resolve_resume_checkpoint_uses_global_strict_for_string_resume() -> None:
    cfg = {"resume": "from_config.pt", "resume_strict": False}

    checkpoint, strict = resolve_resume_checkpoint(cfg)

    assert checkpoint == "from_config.pt"
    assert strict is False


def test_validation_score_supports_weighted_sum_composite_selector() -> None:
    cfg = {
        "training": {
            "selection_metric_mode": "weighted_sum",
            "selection_metric_terms": [
                {"path": "selection_eval.full_locked.accuracy", "weight": 0.5},
                {"path": "selection_eval.full_locked.per_mode.delay_to_final_query.route_match", "weight": 1.0},
            ],
        }
    }
    metrics = {
        "selection_eval": {
            "full_locked": {
                "accuracy": 0.6,
                "per_mode": {"delay_to_final_query": {"route_match": 0.8}},
            }
        }
    }

    score, metric_name = validation_score(metrics, cfg)

    assert math.isclose(score, 1.1)
    assert metric_name == (
        "weighted_sum("
        "0.5*selection_eval.full_locked.accuracy + "
        "1*selection_eval.full_locked.per_mode.delay_to_final_query.route_match)"
    )


def test_validation_score_supports_weighted_geomean_composite_selector() -> None:
    cfg = {
        "training": {
            "selection_metric_mode": "weighted_geomean",
            "selection_metric_terms": [
                {"path": "selection_eval.full_locked.accuracy", "weight": 1.0},
                {"path": "selection_eval.full_locked.per_mode.delay_to_final_query.route_match", "weight": 2.0},
            ],
        }
    }
    metrics = {
        "selection_eval": {
            "full_locked": {
                "accuracy": 0.5,
                "per_mode": {"delay_to_final_query": {"route_match": 0.8}},
            }
        }
    }

    score, metric_name = validation_score(metrics, cfg)

    assert math.isclose(score, math.exp((math.log(0.5) + 2.0 * math.log(0.8)) / 3.0))
    assert metric_name == (
        "weighted_geomean("
        "1*selection_eval.full_locked.accuracy + "
        "2*selection_eval.full_locked.per_mode.delay_to_final_query.route_match)"
    )


def test_validation_score_supports_lexicographic_selector() -> None:
    cfg = {
        "training": {
            "selection_metric_mode": "lexicographic",
            "selection_metric_terms": [
                {"path": "selection_eval.full_locked.per_mode.delay_to_final_query.route_match"},
                {"path": "selection_eval.full_locked.accuracy"},
            ],
        }
    }
    metrics = {
        "selection_eval": {
            "full_locked": {
                "accuracy": 0.6,
                "per_mode": {"delay_to_final_query": {"route_match": 0.8}},
            }
        }
    }

    score, metric_name = validation_score(metrics, cfg)

    assert score == (0.8, 0.6)
    assert metric_name == (
        "lexicographic("
        "selection_eval.full_locked.per_mode.delay_to_final_query.route_match > "
        "selection_eval.full_locked.accuracy)"
    )


def test_validation_score_supports_lexicographic_floor_selector() -> None:
    cfg = {
        "training": {
            "selection_metric_mode": "lexicographic",
            "selection_metric_terms": [
                {
                    "path": "selection_eval.full_locked.per_mode.delay_to_final_query.route_match",
                    "minimum": 0.5,
                },
                {"path": "selection_eval.full_locked.accuracy"},
            ],
        }
    }
    route_passing_metrics = {
        "selection_eval": {
            "full_locked": {
                "accuracy": 0.45,
                "per_mode": {"delay_to_final_query": {"route_match": 0.55}},
            }
        }
    }
    route_failing_metrics = {
        "selection_eval": {
            "full_locked": {
                "accuracy": 0.9,
                "per_mode": {"delay_to_final_query": {"route_match": 0.4}},
            }
        }
    }

    passing_score, metric_name = validation_score(route_passing_metrics, cfg)
    failing_score, _ = validation_score(route_failing_metrics, cfg)

    assert passing_score == (1.0, 0.55, 0.45)
    assert failing_score == (0.0, 0.4, 0.9)
    assert passing_score > failing_score
    assert initial_selection_score(cfg["training"]) == (float("-inf"), float("-inf"), float("-inf"))
    assert metric_name == (
        "lexicographic("
        "selection_eval.full_locked.per_mode.delay_to_final_query.route_match>=0.5 > "
        "selection_eval.full_locked.per_mode.delay_to_final_query.route_match > "
        "selection_eval.full_locked.accuracy)"
    )


def test_apply_partial_init_supports_weighted_source_interpolation(tmp_path: Path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    current_state = model.state_dict()
    original_second_weight = current_state["1.weight"].clone()
    original_second_bias = current_state["1.bias"].clone()

    source_a = {
        "0.weight": torch.full_like(current_state["0.weight"], 2.0),
        "0.bias": torch.full_like(current_state["0.bias"], 1.0),
        "1.weight": torch.full_like(current_state["1.weight"], 9.0),
        "1.bias": torch.full_like(current_state["1.bias"], 9.0),
    }
    source_b = {
        "0.weight": torch.full_like(current_state["0.weight"], 6.0),
        "0.bias": torch.full_like(current_state["0.bias"], 5.0),
        "1.weight": torch.full_like(current_state["1.weight"], 3.0),
        "1.bias": torch.full_like(current_state["1.bias"], 3.0),
    }

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    torch.save({"model": source_a}, run_a / "hard_st_best.pt")
    torch.save({"model": source_b}, run_b / "hard_st_best.pt")

    summary = apply_partial_init(
        model,
        {
            "sources": [
                {"run_dir": str(run_a), "weight": 0.25},
                {"run_dir": str(run_b), "weight": 0.75},
            ],
            "include_prefixes": ["0."],
        },
    )

    updated = model.state_dict()
    assert torch.allclose(updated["0.weight"], torch.full_like(updated["0.weight"], 5.0))
    assert torch.allclose(updated["0.bias"], torch.full_like(updated["0.bias"], 4.0))
    assert torch.allclose(updated["1.weight"], original_second_weight)
    assert torch.allclose(updated["1.bias"], original_second_bias)
    assert summary["partial_init_parameter_count"] == 2
    assert summary["partial_init_weights"] == [0.25, 0.75]


def test_apply_partial_init_supports_source_specific_filters(tmp_path: Path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    current_state = model.state_dict()

    source_a = {
        "0.weight": torch.full_like(current_state["0.weight"], 2.0),
        "0.bias": torch.full_like(current_state["0.bias"], 1.0),
        "1.weight": torch.full_like(current_state["1.weight"], 9.0),
        "1.bias": torch.full_like(current_state["1.bias"], 9.0),
    }
    source_b = {
        "0.weight": torch.full_like(current_state["0.weight"], 6.0),
        "0.bias": torch.full_like(current_state["0.bias"], 5.0),
        "1.weight": torch.full_like(current_state["1.weight"], 3.0),
        "1.bias": torch.full_like(current_state["1.bias"], 3.0),
    }

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    torch.save({"model": source_a}, run_a / "hard_st_best.pt")
    torch.save({"model": source_b}, run_b / "hard_st_best.pt")

    summary = apply_partial_init(
        model,
        {
            "sources": [
                {"run_dir": str(run_a), "weight": 0.25},
                {"run_dir": str(run_b), "weight": 0.75, "include_prefixes": ["0."]},
            ],
            "include_prefixes": ["0.", "1."],
        },
    )

    updated = model.state_dict()
    assert torch.allclose(updated["0.weight"], torch.full_like(updated["0.weight"], 5.0))
    assert torch.allclose(updated["0.bias"], torch.full_like(updated["0.bias"], 4.0))
    assert torch.allclose(updated["1.weight"], torch.full_like(updated["1.weight"], 9.0))
    assert torch.allclose(updated["1.bias"], torch.full_like(updated["1.bias"], 9.0))
    assert summary["partial_init_parameter_count"] == 4
    assert summary["partial_init_weights"] == [0.25, 0.75]

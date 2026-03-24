from __future__ import annotations

from types import SimpleNamespace

import math

from src.train.run import apply_cli_overrides, resolve_resume_checkpoint, validation_score


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

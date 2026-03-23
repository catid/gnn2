from __future__ import annotations

from types import SimpleNamespace

from src.train.run import apply_cli_overrides, resolve_resume_checkpoint


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

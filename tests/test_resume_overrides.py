from __future__ import annotations

from types import SimpleNamespace

from src.train.run import apply_cli_overrides


def test_apply_cli_overrides_preserves_config_resume_when_flag_missing() -> None:
    cfg = {"resume": "from_config.pt"}
    args = SimpleNamespace(resume=None)

    updated = apply_cli_overrides(cfg, args)

    assert updated["resume"] == "from_config.pt"


def test_apply_cli_overrides_replaces_config_resume_when_flag_present() -> None:
    cfg = {"resume": "from_config.pt"}
    args = SimpleNamespace(resume="from_cli.pt")

    updated = apply_cli_overrides(cfg, args)

    assert updated["resume"] == "from_cli.pt"

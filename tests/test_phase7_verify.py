from __future__ import annotations

from src.utils.phase7_verify import infer_method_settings, resolve_checkpoint


def test_infer_method_settings_supports_reinforce() -> None:
    cfg = {
        "method": {"name": "reinforce", "temperature": 1.25},
        "training": {"train_steps": 320},
    }

    method_name, route_mode, temperature, estimator, total_steps, section_cfg = infer_method_settings(cfg)

    assert method_name == "reinforce"
    assert route_mode == "hard"
    assert temperature == 1.25
    assert estimator == "straight_through"
    assert total_steps == 320
    assert section_cfg is cfg["training"]


def test_resolve_checkpoint_supports_reinforce_best(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate = run_dir / "reinforce_best.pt"
    candidate.write_bytes(b"")

    resolved = resolve_checkpoint(run_dir, "reinforce", None)
    assert resolved == candidate

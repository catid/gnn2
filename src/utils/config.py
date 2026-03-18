from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must parse to a mapping.")

    if "_base_" in raw:
        base_path = (config_path.parent / raw["_base_"]).resolve()
        base_cfg = load_config(base_path)
        merged = deep_update(base_cfg, {k: v for k, v in raw.items() if k != "_base_"})
        merged["config_path"] = str(config_path)
        return merged

    raw["config_path"] = str(config_path)
    return raw


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

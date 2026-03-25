from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from src.data import build_benchmark
from src.models import PacketRoutingModel
from src.train.run import (
    current_compute_penalties,
    evaluate_model,
    load_checkpoint,
    seed_everything,
)
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute and cross-check phase-12 metrics from a saved checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Existing run directory with summary/config/checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--eval-config",
        action="append",
        default=[],
        help="Optional evaluation config path. May be passed multiple times for locked confirmations.",
    )
    parser.add_argument("--out", default=None, help="Optional explicit output JSON path.")
    parser.add_argument("--val-batches", type=int, default=0, help="Override validation batches.")
    parser.add_argument("--test-batches", type=int, default=0, help="Override test batches.")
    parser.add_argument(
        "--confirm-batches",
        type=int,
        default=-1,
        help="Override confirmation batches. Use 0 to skip confirm, -1 to auto-detect from config.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Numeric tolerance for metric diffs.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_run_stem(run_name: str) -> str:
    stem = re.sub(r"_reboot_partial$", "", run_name)
    stem = re.sub(r"^\d{8}_\d{6}_", "", stem)
    stem = re.sub(r"_seed\d+(?:_(?:p\d+|rerun\d+))?$", "", stem)
    return stem


def _stem_to_config_basename(stem: str) -> str | None:
    direct_config_prefixes = (
        "hard_st_benchmark_",
        "hybrid_es_benchmark_",
        "reinforce_benchmark_",
        "soft_benchmark_",
    )
    if stem.startswith(direct_config_prefixes):
        return f"{stem}.yaml"
    prefix_map = {
        "hard_st_b_v2_": "hard_st_benchmark_b_v2_",
        "hybrid_es_b_v2_": "hybrid_es_benchmark_b_v2_",
        "reinforce_b_v2_": "reinforce_benchmark_b_v2_",
        "soft_b_v2_": "soft_benchmark_b_v2_",
    }
    for run_prefix, config_prefix in prefix_map.items():
        if stem.startswith(run_prefix):
            return f"{config_prefix}{stem[len(run_prefix):]}.yaml"
    return None


def run_name_to_config_basenames(run_name: str) -> list[str]:
    stem = re.sub(r"_reboot_partial$", "", run_name)
    stem = re.sub(r"^\d{8}_\d{6}_", "", stem)
    stage_stripped = re.sub(r"_(?:p\d+|rerun\d+)$", "", stem)
    candidates = [stage_stripped, canonical_run_stem(run_name)]
    basenames: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        basename = _stem_to_config_basename(candidate)
        if basename is None or basename in seen:
            continue
        seen.add(basename)
        basenames.append(basename)
    return basenames


def eval_config_label(eval_config_path: str, idx: int) -> str:
    stem = Path(eval_config_path).stem
    prefixes = (
        "hard_st_benchmark_b_v2_confirm_",
        "hybrid_es_benchmark_b_v2_confirm_",
        "reinforce_benchmark_b_v2_confirm_",
        "soft_benchmark_b_v2_confirm_",
    )
    for prefix in prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    alias_map = {
        "finalqueryheavy": "finalquery_heavy",
        "fulllocked": "full_locked",
    }
    return alias_map.get(stem, stem or f"eval_{idx}")


def find_repo_config_for_run(run_dir: Path) -> Path | None:
    basenames = run_name_to_config_basenames(run_dir.name)
    if not basenames:
        return None
    search_roots = [
        repo_root() / "configs" / "phase12" / "dev",
        repo_root() / "configs" / "phase12" / "main",
        repo_root() / "configs" / "phase11" / "dev",
        repo_root() / "configs" / "phase11" / "main",
        repo_root() / "configs" / "phase10" / "dev",
        repo_root() / "configs" / "phase10" / "main",
        repo_root() / "configs" / "phase9" / "dev",
        repo_root() / "configs" / "phase9" / "main",
        repo_root() / "configs",
    ]
    for basename in basenames:
        seen: set[Path] = set()
        matches: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            for candidate in root.rglob(basename):
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                matches.append(resolved)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            for preferred_fragment in ("/phase12/dev/", "/phase11/dev/", "/phase10/dev/", "/phase9/dev/"):
                preferred = [path for path in matches if preferred_fragment in str(path)]
                if len(preferred) == 1:
                    return preferred[0]
            return sorted(matches)[0]
    return None


def resolve_config_path(run_dir: Path, config_hint: Path) -> Path:
    if config_hint.exists():
        try:
            load_config(config_hint)
            return config_hint
        except FileNotFoundError:
            pass
    fallback = find_repo_config_for_run(run_dir)
    if fallback is not None:
        return fallback
    if config_hint.exists():
        return config_hint
    raise FileNotFoundError(f"Could not resolve config for run {run_dir} from hint {config_hint}.")


def resolve_run_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    payload = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    config_hint = Path(payload.get("config_path") or run_dir / "config.yaml")
    config_path = resolve_config_path(run_dir, config_hint)
    payload.setdefault("config_path", str(config_path))
    return load_config(str(config_path)), payload


def resolve_checkpoint(run_dir: Path, method_name: str, explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    candidates = []
    if method_name in {"hard_st", "soft", "reinforce"}:
        candidates.append(run_dir / f"{method_name}_best.pt")
    if method_name == "hybrid_es":
        candidates.append(run_dir / "hybrid_es_best.pt")
    candidates.extend(sorted(run_dir.glob("*_best.pt")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_dir}.")


def infer_method_settings(cfg: dict[str, Any]) -> tuple[str, str, float, str, int, dict[str, Any]]:
    method_name = str(cfg["method"]["name"])
    if method_name == "soft":
        route_mode = "soft"
        estimator = "straight_through"
        section_cfg = cfg["training"]
        total_steps = int(section_cfg["train_steps"])
    elif method_name == "hard_st":
        route_mode = "hard_st"
        estimator = str(cfg["method"].get("estimator", "straight_through"))
        section_cfg = cfg["training"]
        total_steps = int(section_cfg["train_steps"])
    elif method_name == "reinforce":
        route_mode = "hard"
        estimator = "straight_through"
        section_cfg = cfg["training"]
        total_steps = int(section_cfg["train_steps"])
    elif method_name == "hybrid_es":
        route_mode = "hard"
        estimator = "straight_through"
        section_cfg = cfg["es"]
        total_steps = int(section_cfg["generations"])
    else:
        raise ValueError(f"Unknown method: {method_name}")
    temperature = float(cfg["method"].get("temperature", 1.0))
    return method_name, route_mode, temperature, estimator, total_steps, section_cfg


def resolve_eval_requests(
    *,
    payload: dict[str, Any],
    section_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    args: argparse.Namespace,
    allow_summary_compare: bool,
) -> list[tuple[str, int]]:
    requests = [
        ("val", int(args.val_batches or section_cfg.get("val_batches", train_cfg.get("val_batches", 8)))),
        ("test", int(args.test_batches or section_cfg.get("test_batches", train_cfg.get("test_batches", 16)))),
    ]
    if args.confirm_batches == 0:
        return requests
    auto_confirm = allow_summary_compare and "confirm" in payload.get("summary", {})
    confirm_batches = args.confirm_batches
    if confirm_batches < 0:
        confirm_batches = int(section_cfg.get("confirm_batches", train_cfg.get("confirm_batches", 0)))
    if confirm_batches > 0 or auto_confirm:
        requests.append(
            (
                "confirm",
                int(confirm_batches if confirm_batches > 0 else section_cfg.get("test_batches", train_cfg.get("test_batches", 16))),
            )
        )
    return requests


def resolve_metric_path(metrics: dict[str, Any], path: str) -> float:
    current: Any = metrics
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return float("nan")
        current = current[part]
    if isinstance(current, (int, float)):
        return float(current)
    return float("nan")


def compare_metrics(expected: dict[str, Any], actual: dict[str, Any], tolerance: float) -> dict[str, float]:
    diffs: dict[str, float] = {}
    for key, expected_value in expected.items():
        if isinstance(expected_value, dict):
            actual_value = actual.get(key, {})
            if isinstance(actual_value, dict):
                nested = compare_metrics(expected_value, actual_value, tolerance)
                for nested_key, diff in nested.items():
                    diffs[f"{key}.{nested_key}"] = diff
            continue
        if not isinstance(expected_value, (int, float)):
            continue
        actual_value = actual.get(key)
        if not isinstance(actual_value, (int, float)):
            diffs[key] = float("inf")
            continue
        diff = abs(float(expected_value) - float(actual_value))
        if diff > tolerance:
            diffs[key] = diff
    return diffs


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg, payload = resolve_run_config(run_dir)
    seed_everything(int(cfg["experiment"]["seed"]))
    method_name, route_mode, temperature, estimator, total_steps, section_cfg = infer_method_settings(cfg)
    checkpoint = resolve_checkpoint(run_dir, method_name, args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PacketRoutingModel(cfg["model"]).to(device)
    checkpoint_payload = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_payload["model"])
    model.eval()

    benchmark_cfg = cfg["benchmark"]
    benchmark_name = benchmark_cfg["name"]
    benchmark = build_benchmark(benchmark_name, benchmark_cfg)

    objective_cfg = cfg.get("objective", {})
    schedule_cfg = cfg.get("training", {}).get("schedules", {})
    compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, total_steps)
    requests = resolve_eval_requests(
        payload=payload,
        section_cfg=section_cfg,
        train_cfg=cfg.get("training", {}),
        args=args,
        allow_summary_compare=not args.eval_config,
    )

    results: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "config_path": payload.get("config_path"),
        "evaluations": {},
    }
    for split, num_batches in requests:
        metrics = evaluate_model(
            model,
            benchmark,
            split=split,
            route_mode=route_mode,
            estimator=estimator,
            temperature=temperature,
            compute_penalties=compute_penalties,
            batch_size=int(section_cfg.get("batch_size", cfg.get("training", {}).get("batch_size", 32))),
            num_batches=num_batches,
            device=device,
            objective_cfg=objective_cfg,
        )
        results["evaluations"][split] = metrics

    for idx, eval_config_path in enumerate(args.eval_config):
        eval_cfg = load_config(eval_config_path)
        eval_benchmark = build_benchmark(eval_cfg["benchmark"]["name"], eval_cfg["benchmark"])
        eval_training = eval_cfg.get("training", {})
        eval_objective = eval_cfg.get("objective", objective_cfg)
        eval_schedule = eval_training.get("schedules", {})
        eval_penalties = current_compute_penalties(eval_objective, eval_schedule, total_steps)
        label = eval_config_label(eval_config_path, idx)
        metrics = evaluate_model(
            model,
            eval_benchmark,
            split="test",
            route_mode=route_mode,
            estimator=estimator,
            temperature=temperature,
            compute_penalties=eval_penalties,
            batch_size=int(eval_training.get("batch_size", section_cfg.get("batch_size", 32))),
            num_batches=int(args.confirm_batches if args.confirm_batches > 0 else eval_training.get("test_batches", 16)),
            device=device,
            objective_cfg=eval_objective,
        )
        results["evaluations"][label] = metrics

    summary_metrics = payload.get("summary", {})
    results["summary_diffs"] = compare_metrics(summary_metrics, results["evaluations"], args.tolerance)
    results["matches_summary_within_tolerance"] = not bool(results["summary_diffs"])

    out_path = Path(args.out) if args.out is not None else run_dir / "artifacts" / "phase12_verify" / "verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

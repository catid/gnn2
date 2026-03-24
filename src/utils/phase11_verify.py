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
    parser = argparse.ArgumentParser(description="Recompute and cross-check phase-10 metrics from a saved checkpoint.")
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
        repo_root() / "configs" / "phase11" / "dev",
        repo_root() / "configs" / "phase11" / "main",
        repo_root() / "configs" / "phase9" / "dev",
        repo_root() / "configs" / "phase9" / "main",
        repo_root() / "configs" / "phase8" / "dev",
        repo_root() / "configs" / "phase8" / "main",
        repo_root() / "configs" / "phase7" / "dev",
        repo_root() / "configs" / "phase7" / "main",
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
            preferred = [path for path in matches if "/phase11/dev/" in str(path)]
            if len(preferred) == 1:
                return preferred[0]
            preferred = [path for path in matches if "/phase9/dev/" in str(path)]
            if len(preferred) == 1:
                return preferred[0]
            preferred = [path for path in matches if "/phase8/dev/" in str(path)]
            if len(preferred) == 1:
                return preferred[0]
            preferred = [path for path in matches if "/phase7/dev/" in str(path)]
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


def metric_paths(metrics: dict[str, Any]) -> list[str]:
    paths = [
        "accuracy",
        "compute",
        "route_match",
        "premature_exit_rate",
        "exit_time",
        "per_mode.delay_to_final_query.accuracy",
        "per_mode.delay_to_final_query.exit_time",
        "per_mode.delay_to_final_query.route_match",
        "per_mode.delay_to_final_query.premature_exit_rate",
        "per_mode.delay_to_trigger_exit.accuracy",
        "per_mode.easy_exit.accuracy",
    ]
    return [path for path in paths if resolve_metric_path(metrics, path) == resolve_metric_path(metrics, path)]


def compare_metrics(
    expected: dict[str, Any],
    actual: dict[str, Any],
    tolerance: float,
) -> dict[str, dict[str, float | bool]]:
    comparisons: dict[str, dict[str, float | bool]] = {}
    for path in metric_paths(expected):
        expected_value = resolve_metric_path(expected, path)
        actual_value = resolve_metric_path(actual, path)
        if expected_value != expected_value or actual_value != actual_value:
            continue
        diff = abs(expected_value - actual_value)
        comparisons[path] = {
            "expected": expected_value,
            "actual": actual_value,
            "abs_diff": diff,
            "matches": diff <= tolerance,
        }
    return comparisons


def evaluate_checkpoint(
    *,
    cfg: dict[str, Any],
    payload: dict[str, Any],
    run_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    summary_key_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method_name, route_mode, temperature, estimator, total_steps, section_cfg = infer_method_settings(cfg)
    train_cfg = cfg["training"]
    system_cfg = cfg.get("system", {})
    amp_enabled = bool(system_cfg.get("amp", False))
    amp_dtype = str(system_cfg.get("amp_dtype", "bf16"))

    seed_everything(int(cfg["experiment"]["seed"]))
    benchmark = build_benchmark(cfg["benchmark"])
    model_cfg = {
        **cfg["model"],
        "num_nodes": benchmark.num_nodes,
        "obs_dim": benchmark.obs_dim,
        "num_classes": benchmark.num_classes,
        "max_total_steps": benchmark.config.get(
            "max_total_steps",
            benchmark.config.get("seq_len", 2) * max(benchmark.num_nodes, 1) * 2,
        ),
    }
    if hasattr(benchmark, "query_offset"):
        model_cfg["query_offset"] = int(benchmark.query_offset)
    if hasattr(benchmark, "query_cardinality"):
        model_cfg["query_cardinality"] = int(benchmark.query_cardinality)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PacketRoutingModel(model_cfg).to(device)
    load_checkpoint(checkpoint_path, model, strict=bool(cfg.get("resume_strict", True)))

    compute_penalties = current_compute_penalties(cfg["objective"], cfg.get("objective_schedule", {}), total_steps)
    eval_batch_size = int(
        section_cfg.get(
            "eval_batch_size",
            section_cfg.get("val_batch_size", train_cfg.get("val_batch_size", train_cfg["batch_size"])),
        )
    )
    split_metrics: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    allow_summary_compare = summary_key_prefix == "base"
    for split_name, num_batches in resolve_eval_requests(
        payload=payload,
        section_cfg=section_cfg,
        train_cfg=train_cfg,
        args=args,
        allow_summary_compare=allow_summary_compare,
    ):
        metrics = evaluate_model(
            model=model,
            benchmark=benchmark,
            device=device,
            benchmark_name=cfg["benchmark"]["name"],
            split=split_name,
            num_batches=num_batches,
            batch_size=eval_batch_size,
            route_mode=route_mode,
            compute_penalties=compute_penalties,
            temperature=temperature,
            estimator=estimator,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            routing_cfg=None,
        )
        split_metrics[split_name] = metrics
        if allow_summary_compare:
            expected = payload.get("summary", {}).get(split_name)
            if isinstance(expected, dict):
                comparisons[split_name] = compare_metrics(expected, metrics, args.tolerance)
    return split_metrics, comparisons


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    base_cfg, payload = resolve_run_config(run_dir)
    method_name, _, _, _, _, _ = infer_method_settings(base_cfg)
    checkpoint_path = resolve_checkpoint(run_dir, method_name, args.checkpoint)

    verification_payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "base_config_path": payload.get("config_path"),
        "checkpoint": str(checkpoint_path),
        "method": method_name,
        "tolerance": args.tolerance,
        "evaluations": {},
    }

    base_metrics, base_comparisons = evaluate_checkpoint(
        cfg=base_cfg,
        payload=payload,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        args=args,
        summary_key_prefix="base",
    )
    verification_payload["evaluations"]["base"] = {
        "config_path": payload.get("config_path"),
        "metrics": base_metrics,
        "comparisons": base_comparisons,
    }

    for idx, eval_config_path in enumerate(args.eval_config):
        eval_cfg = load_config(eval_config_path)
        metrics, comparisons = evaluate_checkpoint(
            cfg=eval_cfg,
            payload=payload,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            args=args,
            summary_key_prefix=f"extra_{idx}",
        )
        verification_payload["evaluations"][eval_config_label(eval_config_path, idx)] = {
            "config_path": eval_config_path,
            "metrics": metrics,
            "comparisons": comparisons,
        }

    out_path = Path(args.out) if args.out else run_dir / "artifacts" / "phase11_verify" / "verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verification_payload, indent=2, sort_keys=True))
    print(json.dumps({"out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()

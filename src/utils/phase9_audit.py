from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.data import build_benchmark
from src.models import PacketRoutingModel
from src.train.run import (
    autocast_context,
    build_control_controls,
    build_memory_controls,
    build_release_controls,
    build_routing_controls,
    build_wait_controls,
    load_checkpoint,
    seed_everything,
)
from src.utils.config import load_config
from src.utils.phase9_verify import resolve_config_path


REPRESENTATION_SPECS = {
    "packet_state_query": ("packet_state", "query_time"),
    "memory_read_state_query": ("memory_read_state", "query_time"),
    "router_logits_query": ("router_logits", "query_time"),
    "router_probs_query": ("router_probs", "query_time"),
    "control_state_query": ("control_state", "query_time"),
    "wait_state_query": ("wait_state", "query_time"),
    "sink_state_query": ("sink_state", "query_time"),
    "final_sink_state": ("final_sink_state", None),
    "final_readout_input": ("final_readout_input", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-9 frozen-state content audit for answer decodability."
    )
    parser.add_argument("--run-dir", required=True, help="Existing run directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--out-dir", default=None, help="Optional explicit audit output directory.")
    parser.add_argument("--eval-config", default=None, help="Optional benchmark/eval config override.")
    parser.add_argument("--split", default="test", choices=["val", "test", "confirm"], help="Main audit split.")
    parser.add_argument("--num-batches", type=int, default=8, help="Batches for main split trace collection.")
    parser.add_argument("--batch-size", type=int, default=0, help="Optional eval batch-size override.")
    parser.add_argument("--probe-train-batches", type=int, default=8, help="Train batches for fitting probes.")
    parser.add_argument("--probe-test-batches", type=int, default=8, help="Held-out batches for evaluating probes.")
    parser.add_argument("--epochs", type=int, default=200, help="Probe optimization epochs.")
    return parser.parse_args()


def resolve_run_config(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        config_hint = Path(summary.get("config_path") or run_dir / "config.yaml")
    else:
        config_hint = run_dir / "config.yaml"
    config_path = resolve_config_path(run_dir, config_hint)
    return load_config(str(config_path))


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


def infer_route_settings(cfg: dict[str, Any]) -> tuple[str, float, str]:
    method_name = str(cfg["method"]["name"])
    if method_name == "soft":
        return "soft", float(cfg["method"].get("temperature", 1.0)), "straight_through"
    if method_name == "hard_st":
        return "hard_st", float(cfg["method"].get("temperature", 1.0)), str(
            cfg["method"].get("estimator", "straight_through")
        )
    if method_name in {"hybrid_es", "reinforce"}:
        return "hard", float(cfg["method"].get("temperature", 1.0)), "straight_through"
    raise ValueError(f"Unknown method: {method_name}")


def fit_answer_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    num_classes: int,
    model_kind: str,
    epochs: int,
) -> dict[str, float]:
    if train_x.numel() == 0 or test_x.numel() == 0:
        return {"accuracy": float("nan"), "ce": float("nan")}

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    if model_kind == "linear":
        model = torch.nn.Linear(train_x.shape[1], num_classes)
    elif model_kind == "mlp":
        hidden = max(16, min(128, train_x.shape[1] * 2))
        model = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[1], hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, num_classes),
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(epochs):
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = model(test_x)
        accuracy = float((test_logits.argmax(dim=-1) == test_y).float().mean().item())
        ce = float(F.cross_entropy(test_logits, test_y).item())
    return {"accuracy": accuracy, "ce": ce}


def collect_split(
    *,
    cfg: dict[str, Any],
    model: PacketRoutingModel,
    benchmark,
    device: torch.device,
    split: str,
    num_batches: int,
    batch_size: int,
) -> dict[str, Any]:
    route_mode, temperature, estimator = infer_route_settings(cfg)
    amp_enabled = bool(cfg.get("system", {}).get("amp", False))
    amp_dtype = str(cfg.get("system", {}).get("amp_dtype", "bf16"))
    compute_penalties = {"hops": 0.0, "delays": 0.0, "ttl_fail": 0.0}

    traces: list[dict[str, torch.Tensor]] = []
    metadata: dict[str, list[torch.Tensor]] = {}

    model.eval()
    with torch.no_grad():
        for step in range(num_batches):
            batch = benchmark.sample_batch(batch_size=batch_size, split=split, step=step, device=device)
            forced_actions, action_masks, oracle_actions, oracle_action_mask, oracle_route_weight, delay_write_weight = build_routing_controls(
                batch,
                None,
                split=split,
            )
            memory_payload_targets, memory_payload_mask, memory_payload_weight = build_memory_controls(
                batch,
                None,
                split=split,
            )
            control_targets, control_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
                batch,
                cfg.get("routing"),
                split="train",
            )
            wait_targets, wait_mask, wait_weight, wait_positive_weight, wait_negative_weight = build_wait_controls(
                batch,
                cfg.get("routing"),
                split="train",
            )
            release_targets, release_mask, release_weight, release_positive_weight = build_release_controls(
                batch,
                cfg.get("routing"),
                split="train",
            )
            with autocast_context(device, amp_enabled, amp_dtype):
                output = model(
                    observations=batch.observations,
                    labels=batch.labels,
                    route_mode=route_mode,
                    compute_penalties=compute_penalties,
                    temperature=temperature,
                    estimator=estimator,
                    truncate_bptt_steps=0,
                    forced_actions=forced_actions,
                    action_masks=action_masks,
                    oracle_actions=oracle_actions,
                    oracle_action_mask=oracle_action_mask,
                    oracle_route_weight=oracle_route_weight,
                    delay_write_targets=batch.delay_write_targets,
                    delay_write_mask=batch.delay_write_mask,
                    delay_write_weight=delay_write_weight,
                    memory_payload_targets=memory_payload_targets,
                    memory_payload_mask=memory_payload_mask,
                    memory_payload_weight=memory_payload_weight,
                    control_targets=control_targets,
                    control_mask=control_mask,
                    control_weight=control_weight,
                    anti_exit_mask=anti_exit_mask,
                    anti_exit_weight=anti_exit_weight,
                    wait_targets=wait_targets,
                    wait_mask=wait_mask,
                    wait_weight=wait_weight,
                    wait_positive_weight=wait_positive_weight,
                    wait_negative_weight=wait_negative_weight,
                    release_targets=release_targets,
                    release_mask=release_mask,
                    release_weight=release_weight,
                    release_positive_weight=release_positive_weight,
                    return_trace=True,
                )
            trace = output.trace or {}
            traces.append({key: value.detach().cpu() for key, value in trace.items()})
            metadata.setdefault("labels", []).append(batch.labels.detach().cpu())
            metadata.setdefault("modes", []).append(batch.modes.detach().cpu())
            for key, value in batch.metadata.items():
                metadata.setdefault(key, []).append(value.detach().cpu())

    merged_meta = {key: torch.cat(values, dim=0) for key, values in metadata.items()}
    merged_trace: dict[str, torch.Tensor] = {}
    if traces:
        for key in traces[0]:
            merged_trace[key] = torch.cat([trace[key] for trace in traces], dim=0)
    return {"metadata": merged_meta, "trace": merged_trace}


def final_query_dataset(
    audit: dict[str, Any],
    *,
    benchmark,
    representation_name: str,
    conditioned: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    trace = audit["trace"]
    metadata = audit["metadata"]
    if representation_name not in REPRESENTATION_SPECS:
        raise KeyError(representation_name)
    trace_key, time_key = REPRESENTATION_SPECS[representation_name]
    if trace_key not in trace:
        return torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)

    final_mask = metadata.get("needs_final_query")
    if final_mask is None:
        return torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)
    final_mask = final_mask > 0
    if not final_mask.any():
        return torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)

    labels = metadata["labels"][final_mask].long()
    queries = metadata["query"][final_mask].long()
    features = trace[trace_key]
    if time_key is None:
        rep = features[final_mask]
    else:
        query_time = metadata[time_key][final_mask].long()
        rep = features[final_mask, query_time]
    if rep.ndim == 1:
        rep = rep.unsqueeze(-1)
    rep = rep.float()
    if conditioned:
        query_one_hot = F.one_hot(queries, num_classes=int(benchmark.query_cardinality)).float()
        rep = torch.cat([rep, query_one_hot], dim=-1)
    return rep, labels


def build_probe_tables(
    *,
    train_audit: dict[str, Any],
    test_audit: dict[str, Any],
    benchmark,
    epochs: int,
) -> dict[str, Any]:
    probe_tables: dict[str, Any] = {}
    for name in REPRESENTATION_SPECS:
        train_x, train_y = final_query_dataset(
            train_audit,
            benchmark=benchmark,
            representation_name=name,
            conditioned=False,
        )
        test_x, test_y = final_query_dataset(
            test_audit,
            benchmark=benchmark,
            representation_name=name,
            conditioned=False,
        )
        if train_x.numel() == 0 or test_x.numel() == 0:
            continue
        train_x_cond, train_y_cond = final_query_dataset(
            train_audit,
            benchmark=benchmark,
            representation_name=name,
            conditioned=True,
        )
        test_x_cond, test_y_cond = final_query_dataset(
            test_audit,
            benchmark=benchmark,
            representation_name=name,
            conditioned=True,
        )
        probe_tables[name] = {
            "linear": fit_answer_probe(
                train_x,
                train_y,
                test_x,
                test_y,
                num_classes=benchmark.num_classes,
                model_kind="linear",
                epochs=epochs,
            ),
            "mlp": fit_answer_probe(
                train_x,
                train_y,
                test_x,
                test_y,
                num_classes=benchmark.num_classes,
                model_kind="mlp",
                epochs=epochs,
            ),
            "linear_query_conditioned": fit_answer_probe(
                train_x_cond,
                train_y_cond,
                test_x_cond,
                test_y_cond,
                num_classes=benchmark.num_classes,
                model_kind="linear",
                epochs=epochs,
            ),
            "mlp_query_conditioned": fit_answer_probe(
                train_x_cond,
                train_y_cond,
                test_x_cond,
                test_y_cond,
                num_classes=benchmark.num_classes,
                model_kind="mlp",
                epochs=epochs,
            ),
        }
    return probe_tables


def best_probe_summary(probe_tables: dict[str, Any]) -> dict[str, Any]:
    best_name = None
    best_variant = None
    best_acc = float("-inf")
    best_linear = float("-inf")
    best_conditioned = float("-inf")
    for name, variants in probe_tables.items():
        linear_acc = float(variants.get("linear", {}).get("accuracy", float("nan")))
        if linear_acc == linear_acc and linear_acc > best_linear:
            best_linear = linear_acc
        conditioned_acc = float(variants.get("mlp_query_conditioned", {}).get("accuracy", float("nan")))
        if conditioned_acc == conditioned_acc and conditioned_acc > best_conditioned:
            best_conditioned = conditioned_acc
        for variant_name, metrics in variants.items():
            acc = float(metrics.get("accuracy", float("nan")))
            if acc == acc and acc > best_acc:
                best_acc = acc
                best_name = name
                best_variant = variant_name
    return {
        "best_representation": best_name,
        "best_probe_variant": best_variant,
        "best_accuracy": best_acc,
        "best_linear_accuracy": best_linear,
        "best_conditioned_accuracy": best_conditioned,
        "nonlinear_gain": best_acc - best_linear if best_linear == best_linear and best_acc == best_acc else float("nan"),
        "conditioning_gain": best_conditioned - best_linear if best_conditioned == best_conditioned and best_linear == best_linear else float("nan"),
        "head_only_go_signal": bool(best_acc == best_acc and best_acc >= 0.45),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = resolve_run_config(run_dir)
    if args.eval_config:
        eval_cfg = load_config(args.eval_config)
        cfg["benchmark"] = eval_cfg["benchmark"]

    method_name = str(cfg["method"]["name"])
    checkpoint_path = resolve_checkpoint(run_dir, method_name, args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "artifacts" / "phase9_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = PacketRoutingModel(model_cfg).to(device)
    load_checkpoint(checkpoint_path, model, strict=False)

    default_batch_size = int(
        cfg.get("es", {}).get(
            "eval_batch_size",
            cfg.get("training", {}).get("val_batch_size", cfg.get("training", {}).get("batch_size", 64)),
        )
    )
    batch_size = int(args.batch_size or default_batch_size)

    main_audit = collect_split(
        cfg=cfg,
        model=model,
        benchmark=benchmark,
        device=device,
        split=args.split,
        num_batches=int(args.num_batches),
        batch_size=batch_size,
    )
    probe_train = collect_split(
        cfg=cfg,
        model=model,
        benchmark=benchmark,
        device=device,
        split="train",
        num_batches=int(args.probe_train_batches),
        batch_size=batch_size,
    )
    probe_test = collect_split(
        cfg=cfg,
        model=model,
        benchmark=benchmark,
        device=device,
        split=args.split,
        num_batches=int(args.probe_test_batches),
        batch_size=batch_size,
    )

    probe_tables = build_probe_tables(
        train_audit=probe_train,
        test_audit=probe_test,
        benchmark=benchmark,
        epochs=int(args.epochs),
    )
    final_mask = main_audit["metadata"].get("needs_final_query", torch.zeros(0, dtype=torch.long)) > 0
    answer_labels = main_audit["metadata"].get("labels", torch.zeros(0, dtype=torch.long))
    answer_summary = {
        "count": int(final_mask.sum().item()),
        "num_classes": int(benchmark.num_classes),
        "chance_accuracy": float(1.0 / max(1, benchmark.num_classes)),
        "label_histogram": (
            torch.bincount(answer_labels[final_mask], minlength=benchmark.num_classes).tolist()
            if final_mask.numel() > 0
            else []
        ),
    }
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "eval_config": args.eval_config,
        "answer_summary": answer_summary,
        "probe_tables": probe_tables,
        "best_probe": best_probe_summary(probe_tables),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

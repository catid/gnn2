from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.data import build_benchmark
from src.models import ACTION_DELAY, ACTION_EXIT, ACTION_FORWARD, PacketRoutingModel
from src.train.run import (
    autocast_context,
    build_control_controls,
    build_memory_controls,
    build_release_controls,
    build_routing_controls,
    build_wait_controls,
    load_checkpoint,
    seed_everything,
    summarize_batch,
)
from src.utils.config import load_config


ACTION_INDEX = {
    "forward": ACTION_FORWARD,
    "exit": ACTION_EXIT,
    "delay": ACTION_DELAY,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-5 audit for final-query wait/control retention.")
    parser.add_argument("--run-dir", required=True, help="Existing run directory containing summary/config/checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--out-dir", default=None, help="Optional audit output directory.")
    parser.add_argument("--num-batches", type=int, default=8, help="Eval batches for trace collection.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override eval batch size.")
    parser.add_argument("--probe-train-batches", type=int, default=8, help="Train batches for linear probes.")
    parser.add_argument("--probe-test-batches", type=int, default=8, help="Held-out batches for linear probes.")
    parser.add_argument("--split", default="test", choices=["val", "test", "confirm"], help="Primary audit split.")
    return parser.parse_args()


def resolve_run_config(run_dir: Path) -> tuple[dict[str, Any], Path]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        config_path = Path(summary.get("config_path") or run_dir / "config.yaml")
        return load_config(str(config_path)), summary_path
    return load_config(str(run_dir / "config.yaml")), run_dir / "summary.json"


def resolve_checkpoint(run_dir: Path, method_name: str, explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    candidates = []
    if method_name in {"hard_st", "soft"}:
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
        route_mode = "soft"
        estimator = "straight_through"
    elif method_name == "hard_st":
        route_mode = "hard_st"
        estimator = str(cfg["method"].get("estimator", "straight_through"))
    elif method_name == "hybrid_es":
        route_mode = "hard"
        estimator = "straight_through"
    else:
        raise ValueError(f"Unknown method: {method_name}")
    temperature = float(cfg["method"].get("temperature", 1.0))
    return route_mode, temperature, estimator


def compute_probe_targets(
    batch,
    routing_cfg: dict[str, Any] | None,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    device = batch.labels.device
    dtype = batch.observations.dtype
    batch_size = batch.labels.shape[0]
    wait_targets, wait_mask, _, _, _ = build_wait_controls(batch, routing_cfg, split="train")
    if wait_targets is None or wait_mask is None:
        wait_targets = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        wait_mask = torch.zeros_like(wait_targets)
    release_targets, release_mask, _, _ = build_release_controls(batch, routing_cfg, split="train")
    if release_targets is None or release_mask is None:
        release_targets = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        release_mask = torch.zeros_like(release_targets)
    if batch.metadata and "needs_final_query" in batch.metadata:
        needs_final_query = batch.metadata["needs_final_query"].to(device=device, dtype=dtype).unsqueeze(1).expand(-1, seq_len)
    else:
        needs_final_query = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
    needs_mask = torch.ones_like(needs_final_query)
    return {
        "wait_targets": wait_targets,
        "wait_mask": wait_mask,
        "release_targets": release_targets,
        "release_mask": release_mask,
        "needs_final_query_targets": needs_final_query,
        "needs_final_query_mask": needs_mask,
    }


def mean_over_mask(values: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.float().sum().item()
    if weight <= 0:
        return float("nan")
    return float((values.float() * mask.float()).sum().item() / weight)


def fit_linear_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
) -> dict[str, float]:
    if train_features.numel() == 0 or test_features.numel() == 0:
        return {"accuracy": float("nan"), "bce": float("nan")}
    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_std = train_features.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x = (train_features - feature_mean) / feature_std
    test_x = (test_features - feature_mean) / feature_std
    model = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-4)
    for _ in range(200):
        logits = model(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_logits = model(test_x).squeeze(-1)
        test_probs = torch.sigmoid(test_logits)
        test_pred = (test_probs >= 0.5).float()
        accuracy = float((test_pred == test_targets).float().mean().item())
        bce = float(F.binary_cross_entropy(test_probs.clamp(1e-5, 1.0 - 1e-5), test_targets).item())
    return {"accuracy": accuracy, "bce": bce}


def flatten_probe_tensors(
    trace: dict[str, torch.Tensor],
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = trace.get(key)
    if features is None:
        return torch.zeros(0, 1), torch.zeros(0)
    if features.ndim == 2:
        features = features.unsqueeze(-1)
    valid = (target_mask > 0.0) & (trace["active_mass"] > 0.0)
    if not valid.any():
        return torch.zeros(0, features.shape[-1]), torch.zeros(0)
    flat_x = features[valid].detach().float().cpu()
    flat_y = targets[valid].detach().float().cpu()
    return flat_x, flat_y


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
    benchmark_name = cfg["benchmark"]["name"]
    compute_penalties = {"hops": 0.0, "delays": 0.0, "ttl_fail": 0.0}

    sample_metrics: list[dict[str, torch.Tensor]] = []
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
            sample_metrics.append(summarize_batch(batch, output, benchmark_name))
            trace = output.trace or {}
            probe_targets = compute_probe_targets(batch, cfg.get("routing"), batch.observations.shape[1])
            for key, value in probe_targets.items():
                trace[key] = value.detach().cpu()
            traces.append({key: value.detach().cpu() for key, value in trace.items()})
            metadata.setdefault("modes", []).append(batch.modes.detach().cpu())
            metadata.setdefault("labels", []).append(batch.labels.detach().cpu())
            for key, value in batch.metadata.items():
                metadata.setdefault(key, []).append(value.detach().cpu())

    merged_metrics: dict[str, torch.Tensor] = {}
    for item in sample_metrics:
        for key, value in item.items():
            merged_metrics.setdefault(key, []).append(value.detach().cpu())
    merged_metrics = {
        key: torch.cat([chunk.reshape(-1) for chunk in values], dim=0)
        for key, values in merged_metrics.items()
    }
    merged_meta = {
        key: torch.cat(values, dim=0)
        for key, values in metadata.items()
    }
    merged_trace: dict[str, torch.Tensor] = {}
    if traces:
        for key in traces[0]:
            merged_trace[key] = torch.cat([trace[key] for trace in traces], dim=0)
    return {"metrics": merged_metrics, "metadata": merged_meta, "trace": merged_trace}


def write_exit_histogram(audit: dict[str, Any], out_path: Path) -> None:
    metrics = audit["metrics"]
    meta = audit["metadata"]
    final_mask = meta.get("needs_final_query", torch.zeros_like(metrics["accuracy"])).float() > 0
    if not final_mask.any():
        return
    values = metrics["exit_time"][final_mask].numpy()
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=24, color="#d95f02", alpha=0.85)
    plt.axvline(float(meta["query_time"][final_mask][0].item()), color="#1b9e77", linestyle="--", label="final query")
    plt.title("Final-Query Exit-Time Histogram")
    plt.xlabel("Exit Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_action_traces(audit: dict[str, Any], out_path: Path) -> None:
    trace = audit["trace"]
    meta = audit["metadata"]
    final_mask = meta.get("needs_final_query", torch.zeros(trace["active_mass"].shape[0])).float() > 0
    if not final_mask.any():
        return
    action_mass = trace["action_mass"][final_mask]
    denom = action_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    action_rate = action_mass / denom
    plt.figure(figsize=(8, 4.5))
    for label, index, color in [
        ("FORWARD", ACTION_FORWARD, "#1b9e77"),
        ("EXIT", ACTION_EXIT, "#d95f02"),
        ("DELAY", ACTION_DELAY, "#7570b3"),
    ]:
        plt.plot(action_rate[:, :, index].mean(dim=0).numpy(), label=label, color=color)
    plt.title("Final-Query Action Rates Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Action Rate")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_memory_traces(audit: dict[str, Any], out_path: Path) -> None:
    trace = audit["trace"]
    meta = audit["metadata"]
    final_mask = meta.get("needs_final_query", torch.zeros(trace["active_mass"].shape[0])).float() > 0
    if not final_mask.any():
        return
    plt.figure(figsize=(9, 5))
    plt.plot(trace["memory_read_gate"][final_mask].mean(dim=0).numpy(), label="read_gate", color="#1b9e77")
    plt.plot(trace["memory_write_gate"][final_mask].mean(dim=0).numpy(), label="write_gate", color="#d95f02")
    plt.plot(trace["memory_read_entropy"][final_mask].mean(dim=0).numpy(), label="read_entropy", color="#7570b3")
    plt.plot(trace["memory_write_entropy"][final_mask].mean(dim=0).numpy(), label="write_entropy", color="#e7298a")
    if "control_prob" in trace:
        plt.plot(trace["control_prob"][final_mask].mean(dim=0).numpy(), label="control_prob", color="#66a61e")
    if "wait_prob" in trace:
        plt.plot(trace["wait_prob"][final_mask].mean(dim=0).numpy(), label="wait_prob", color="#a6761d")
    if "release_prob" in trace:
        plt.plot(trace["release_prob"][final_mask].mean(dim=0).numpy(), label="release_prob", color="#e6ab02")
    plt.title("Final-Query Memory / Control Traces")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_router_logit_traces(audit: dict[str, Any], out_path: Path) -> None:
    trace = audit["trace"]
    meta = audit["metadata"]
    final_mask = meta.get("needs_final_query", torch.zeros(trace["active_mass"].shape[0])).float() > 0
    if not final_mask.any():
        return
    logits = trace["router_logits"][final_mask]
    plt.figure(figsize=(8, 4.5))
    for label, index, color in [
        ("FORWARD", ACTION_FORWARD, "#1b9e77"),
        ("EXIT", ACTION_EXIT, "#d95f02"),
        ("DELAY", ACTION_DELAY, "#7570b3"),
    ]:
        plt.plot(logits[:, :, index].mean(dim=0).numpy(), label=label, color=color)
    plt.title("Final-Query Router Logits Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Logit")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_probe_plot(probes: dict[str, dict[str, float]], out_path: Path, title: str) -> None:
    keys = list(probes.keys())
    values = [probes[key]["accuracy"] for key in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(keys, values, color=["#1b9e77", "#d95f02", "#7570b3", "#66a61e"][: len(keys)])
    plt.ylabel("Probe Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def per_mode_table(audit: dict[str, Any], benchmark) -> dict[str, dict[str, float]]:
    metrics = audit["metrics"]
    modes = audit["metadata"]["modes"]
    table: dict[str, dict[str, float]] = {}
    for mode_id, mode_name in benchmark.mode_names.items():
        mask = modes == mode_id
        if not mask.any():
            continue
        table[mode_name] = {
            "count": int(mask.sum().item()),
            "accuracy": float(metrics["accuracy"][mask].float().mean().item()),
            "route_match": float(metrics.get("route_match", torch.zeros_like(metrics["accuracy"]))[mask].float().mean().item()),
            "delay_rate": float(metrics["delay_rate"][mask].float().mean().item()),
            "early_exit_rate": float(metrics["early_exit_rate"][mask].float().mean().item()),
            "exit_time": float(metrics["exit_time"][mask].float().mean().item()),
            "compute": float(metrics["compute"][mask].float().mean().item()),
            "premature_exit_rate": float(metrics.get("premature_exit_rate", torch.zeros_like(metrics["accuracy"]))[mask].float().mean().item()),
            "control_prob_mean": float(metrics.get("control_prob_mean", torch.zeros_like(metrics["accuracy"]))[mask].float().mean().item()),
        }
    return table


def build_probe_summary(
    train_audit: dict[str, Any],
    test_audit: dict[str, Any],
    *,
    target_key: str,
    mask_key: str,
) -> dict[str, dict[str, float]]:
    probe_keys = [
        "packet_state",
        "memory_read_state",
        "router_probs",
        "router_logits",
        "control_state",
        "wait_state",
        "release_prob",
        "control_prob",
        "wait_prob",
    ]
    summary: dict[str, dict[str, float]] = {}
    for key in probe_keys:
        train_x, train_y = flatten_probe_tensors(
            train_audit["trace"],
            train_audit["trace"][target_key],
            train_audit["trace"][mask_key],
            key,
        )
        test_x, test_y = flatten_probe_tensors(
            test_audit["trace"],
            test_audit["trace"][target_key],
            test_audit["trace"][mask_key],
            key,
        )
        if train_x.numel() == 0 or test_x.numel() == 0:
            continue
        summary[key] = fit_linear_probe(train_x, train_y, test_x, test_y)
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg, _ = resolve_run_config(run_dir)
    method_name = str(cfg["method"]["name"])
    checkpoint_path = resolve_checkpoint(run_dir, method_name, args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "artifacts" / "phase5_audit"
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
    load_checkpoint(checkpoint_path, model)

    default_batch_size = int(
        cfg.get("es", {}).get(
            "eval_batch_size",
            cfg.get("training", {}).get("val_batch_size", cfg.get("training", {}).get("batch_size", 64)),
        )
    )
    batch_size = int(args.batch_size or default_batch_size)

    audit = collect_split(
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
    needs_probes = build_probe_summary(
        probe_train,
        probe_test,
        target_key="needs_final_query_targets",
        mask_key="needs_final_query_mask",
    )
    wait_probes = build_probe_summary(
        probe_train,
        probe_test,
        target_key="wait_targets",
        mask_key="wait_mask",
    )
    release_probes = build_probe_summary(
        probe_train,
        probe_test,
        target_key="release_targets",
        mask_key="release_mask",
    )

    per_mode = per_mode_table(audit, benchmark)
    final_mask = audit["metadata"].get("needs_final_query", torch.zeros_like(audit["metrics"]["accuracy"])).float() > 0
    final_query_summary = {
        "count": int(final_mask.sum().item()),
        "accuracy": float(audit["metrics"]["accuracy"][final_mask].float().mean().item()) if final_mask.any() else float("nan"),
        "route_match": float(audit["metrics"].get("route_match", torch.zeros_like(audit["metrics"]["accuracy"]))[final_mask].float().mean().item()) if final_mask.any() else float("nan"),
        "exit_time": float(audit["metrics"]["exit_time"][final_mask].float().mean().item()) if final_mask.any() else float("nan"),
        "premature_exit_rate": float(audit["metrics"].get("premature_exit_rate", torch.zeros_like(audit["metrics"]["accuracy"]))[final_mask].float().mean().item()) if final_mask.any() else float("nan"),
        "needs_probe_packet_state": needs_probes.get("packet_state", {}).get("accuracy"),
        "needs_probe_memory_read_state": needs_probes.get("memory_read_state", {}).get("accuracy"),
        "needs_probe_control_state": needs_probes.get("control_state", {}).get("accuracy"),
        "wait_probe_wait_prob": wait_probes.get("wait_prob", {}).get("accuracy"),
        "release_probe_release_prob": release_probes.get("release_prob", {}).get("accuracy"),
    }
    if final_mask.any():
        query_time = audit["metadata"]["query_time"][final_mask].float()
        exit_time = audit["metrics"]["exit_time"][final_mask].float()
        final_query_summary["mean_exit_minus_query"] = float((exit_time - query_time).mean().item())

    write_exit_histogram(audit, out_dir / "final_query_exit_hist.png")
    write_action_traces(audit, out_dir / "final_query_action_traces.png")
    write_memory_traces(audit, out_dir / "final_query_memory_traces.png")
    write_router_logit_traces(audit, out_dir / "final_query_router_logits.png")
    if needs_probes:
        write_probe_plot(needs_probes, out_dir / "needs_probe_accuracy.png", "Needs-Final-Query Probe Accuracy")
    if wait_probes:
        write_probe_plot(wait_probes, out_dir / "wait_probe_accuracy.png", "Wait-Target Probe Accuracy")
    if release_probes:
        write_probe_plot(release_probes, out_dir / "release_probe_accuracy.png", "Release-Target Probe Accuracy")

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "per_mode": per_mode,
        "final_query_summary": final_query_summary,
        "needs_probe_summary": needs_probes,
        "wait_probe_summary": wait_probes,
        "release_probe_summary": release_probes,
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_cli_soft_smoke_end_to_end(tmp_path: Path) -> None:
    config = {
        "experiment": {"seed": 7, "results_root": str(tmp_path / "root")},
        "system": {"cpu_threads": 2, "amp": False, "amp_dtype": "bf16"},
        "benchmark": {
            "name": "mixed_oracle_routing",
            "num_nodes": 4,
            "obs_dim": 20,
            "num_classes": 2,
            "seq_len": 2,
            "noise_std": 0.05,
            "train_seed": 11,
            "val_seed": 101,
            "test_seed": 1001,
            "confirm_seed": 2001,
        },
        "model": {"hidden_dim": 32, "max_internal_steps": 4, "adapter_rank": 0},
        "objective": {"task_score": "neg_ce", "lambda_hops": 0.01, "lambda_delay": 0.01, "lambda_ttl": 0.1},
        "training": {
            "batch_size": 32,
            "val_batch_size": 32,
            "train_steps": 2,
            "val_every": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "truncate_bptt_steps": 0,
            "val_batches": 1,
            "test_batches": 1,
            "confirm_batches": 1,
        },
        "method": {"name": "soft", "temperature": 1.0, "estimator": "straight_through"},
        "warmstart": {"enabled": False},
        "es": {
            "generations": 2,
            "population": 4,
            "batch_size": 32,
            "eval_batch_size": 32,
            "sigma": 0.05,
            "rank": 1,
            "lr": 0.01,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "noise_reuse": 0,
            "val_every": 1,
            "val_batches": 1,
            "test_batches": 1,
            "confirm_batches": 1,
            "evolve_adapters": False,
        },
    }
    config_path = tmp_path / "smoke.yaml"
    results_dir = tmp_path / "results"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train.run",
            "--config",
            str(config_path),
            "--results-dir",
            str(results_dir),
        ],
        check=True,
        cwd="/home/catid/gnn2",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.utils.phase5_verify",
            "--run-dir",
            str(results_dir),
        ],
        check=True,
        cwd="/home/catid/gnn2",
    )

    assert (results_dir / "summary.json").exists()
    assert (results_dir / "metrics.jsonl").exists()
    summary = json.loads((results_dir / "summary.json").read_text())
    assert "confirm" in summary["summary"]
    assert (results_dir / "artifacts" / "phase5_verify" / "verification.json").exists()

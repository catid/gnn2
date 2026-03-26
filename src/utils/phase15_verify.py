from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.phase12_verify import (
    eval_config_label,
    evaluate_checkpoint,
    infer_method_settings,
    load_config,
    resolve_checkpoint,
    resolve_run_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute and cross-check phase-15 metrics from a saved checkpoint.")
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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    base_cfg, payload = resolve_run_config(run_dir)
    method_name, _, _, _, _, _ = infer_method_settings(base_cfg)
    checkpoint_path = resolve_checkpoint(run_dir, method_name, args.checkpoint)

    verification_payload: dict[str, object] = {
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

    out_path = (
        Path(args.out)
        if args.out is not None
        else run_dir / "artifacts" / "phase15_verify" / "verification.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verification_payload, indent=2, sort_keys=True))
    print(json.dumps({"out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()

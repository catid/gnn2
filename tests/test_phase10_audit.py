from __future__ import annotations

from types import SimpleNamespace

import torch

from src.utils.phase10_audit import build_model_cfg, final_query_dataset


def test_build_model_cfg_hydrates_benchmark_dimensions() -> None:
    benchmark = SimpleNamespace(
        num_nodes=7,
        obs_dim=11,
        num_classes=4,
        config={"seq_len": 5},
    )

    cfg = build_model_cfg({"hidden_dim": 8, "adapter_rank": 0}, benchmark)

    assert cfg["hidden_dim"] == 8
    assert cfg["adapter_rank"] == 0
    assert cfg["num_nodes"] == 7
    assert cfg["obs_dim"] == 11
    assert cfg["num_classes"] == 4
    assert cfg["max_total_steps"] == 70


def test_final_query_dataset_uses_query_indices_for_conditioning() -> None:
    benchmark = SimpleNamespace(query_cardinality=4)
    audit = {
        "metadata": {
            "labels": torch.tensor([1, 2]),
            "needs_final_query": torch.tensor([1, 1]),
            "query": torch.tensor([0, 3]),
            "query_time": torch.tensor([1, 0]),
        },
        "trace": {
            "packet_state": torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ]
            )
        },
    }

    features, labels = final_query_dataset(
        audit,
        benchmark=benchmark,
        representation_name="packet_state_query",
        conditioned=True,
    )

    assert labels.tolist() == [1, 2]
    assert features.shape == (2, 7)
    assert torch.allclose(features[0, :3], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.allclose(features[1, :3], torch.tensor([7.0, 8.0, 9.0]))
    assert torch.allclose(features[0, 3:], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(features[1, 3:], torch.tensor([0.0, 0.0, 0.0, 1.0]))

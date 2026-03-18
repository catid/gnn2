from __future__ import annotations

import torch
import torch.nn as nn

from src.es.low_rank_es import LowRankEvolutionStrategy


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(5, 4)
        self.lin2 = nn.Linear(4, 3)


def test_low_rank_delta_shapes_and_antithetic_pairs() -> None:
    model = TinyModel()
    es = LowRankEvolutionStrategy(
        model=model,
        parameter_names=["lin1.weight", "lin1.bias"],
        sigma=0.1,
        rank=2,
        lr=0.01,
    )

    weight_delta_pos = es.sample_delta(model.lin1.weight, generation=3, member_index=0, target_index=0)
    weight_delta_neg = es.sample_delta(model.lin1.weight, generation=3, member_index=1, target_index=0)
    bias_delta_pos = es.sample_delta(model.lin1.bias, generation=3, member_index=0, target_index=1)
    bias_delta_neg = es.sample_delta(model.lin1.bias, generation=3, member_index=1, target_index=1)

    assert weight_delta_pos.shape == model.lin1.weight.shape
    assert bias_delta_pos.shape == model.lin1.bias.shape
    assert torch.allclose(weight_delta_pos, -weight_delta_neg)
    assert torch.allclose(bias_delta_pos, -bias_delta_neg)


def test_compute_updates_returns_parameter_shaped_tensors() -> None:
    model = TinyModel()
    es = LowRankEvolutionStrategy(
        model=model,
        parameter_names=["lin1.weight", "lin1.bias", "lin2.weight"],
        sigma=0.05,
        rank=2,
        lr=0.01,
    )

    fitness = torch.tensor([1.0, -1.0, 0.5, -0.5])
    updates = es.compute_updates(generation=4, fitness=fitness)

    assert set(updates) == {"lin1.weight", "lin1.bias", "lin2.weight"}
    assert updates["lin1.weight"].shape == model.lin1.weight.shape
    assert updates["lin1.bias"].shape == model.lin1.bias.shape
    assert updates["lin2.weight"].shape == model.lin2.weight.shape


def test_population_split_requires_even_sharding() -> None:
    model = TinyModel()
    es = LowRankEvolutionStrategy(
        model=model,
        parameter_names=["lin1.weight"],
        sigma=0.05,
        rank=1,
        lr=0.01,
    )

    assert es.local_member_range(population=8, world_size=2, rank=1) == (4, 8)

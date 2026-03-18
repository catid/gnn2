from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class PerturbationTarget:
    name: str
    parameter: torch.nn.Parameter


def standardize_fitness(rewards: torch.Tensor) -> torch.Tensor:
    rewards = rewards.float()
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)


class LowRankEvolutionStrategy:
    def __init__(
        self,
        model: torch.nn.Module,
        parameter_names: Iterable[str],
        sigma: float,
        rank: int,
        lr: float,
        weight_decay: float = 0.0,
        noise_reuse: int = 0,
        optimizer_name: str = "adam",
    ):
        self.model = model
        self.sigma = float(sigma)
        self.rank = int(rank)
        self.noise_reuse = int(noise_reuse)

        name_to_param = dict(model.named_parameters())
        self.targets: list[PerturbationTarget] = [
            PerturbationTarget(name=name, parameter=name_to_param[name])
            for name in parameter_names
        ]
        params = [target.parameter for target in self.targets]
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported ES optimizer: {optimizer_name}")

    def local_member_range(self, population: int, world_size: int, rank: int) -> tuple[int, int]:
        if population % world_size != 0:
            raise ValueError(
                f"Population {population} must be divisible by world size {world_size}."
            )
        per_rank = population // world_size
        start = rank * per_rank
        return start, start + per_rank

    def perturb_member(self, generation: int, member_index: int) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        applied: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        for target_index, target in enumerate(self.targets):
            delta = self.sample_delta(
                parameter=target.parameter,
                generation=generation,
                member_index=member_index,
                target_index=target_index,
            )
            with torch.no_grad():
                target.parameter.add_(delta)
            applied.append((target.parameter, delta))
        return applied

    def revert_member(self, applied: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
        with torch.no_grad():
            for parameter, delta in applied:
                parameter.sub_(delta)

    def compute_updates(
        self,
        generation: int,
        fitness: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if fitness.ndim != 1:
            raise ValueError("fitness must be a flat population tensor.")
        updates: dict[str, torch.Tensor] = {}
        for target_index, target in enumerate(self.targets):
            update = torch.zeros_like(target.parameter, device=target.parameter.device)
            for member_index, score in enumerate(fitness):
                score_value = float(score.item())
                if score_value == 0.0:
                    continue
                delta = self.sample_delta(
                    parameter=target.parameter,
                    generation=generation,
                    member_index=member_index,
                    target_index=target_index,
                )
                update.add_(delta, alpha=score_value)
            updates[target.name] = update / max(1, fitness.numel())
        return updates

    def apply_updates(self, updates: dict[str, torch.Tensor]) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for target in self.targets:
                target.parameter.grad = -updates[target.name]
        self.optimizer.step()

    def broadcast_parameters(self, src: int = 0) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        for target in self.targets:
            torch.distributed.broadcast(target.parameter.data, src=src)

    def sample_delta(
        self,
        parameter: torch.nn.Parameter,
        generation: int,
        member_index: int,
        target_index: int,
    ) -> torch.Tensor:
        true_generation = generation
        if self.noise_reuse > 0:
            true_generation = generation // self.noise_reuse
        pair_index = member_index // 2
        sign = 1.0 if member_index % 2 == 0 else -1.0
        seed = self._seed(true_generation, pair_index, target_index)
        device = parameter.device
        generator = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
        generator.manual_seed(seed)

        if parameter.ndim == 2:
            out_dim, in_dim = parameter.shape
            noise = torch.randn(
                out_dim + in_dim,
                self.rank,
                generator=generator,
                device=device,
                dtype=parameter.dtype,
            )
            b = noise[:in_dim]
            a = noise[in_dim:]
            return (self.sigma * sign / (self.rank ** 0.5)) * (a @ b.T)

        return self.sigma * sign * torch.randn(
            parameter.shape,
            generator=generator,
            device=device,
            dtype=parameter.dtype,
        )

    @staticmethod
    def _seed(generation: int, pair_index: int, target_index: int) -> int:
        return (
            17_171
            + generation * 1_000_003
            + pair_index * 91_771
            + target_index * 13_337
        ) % (2**63 - 1)

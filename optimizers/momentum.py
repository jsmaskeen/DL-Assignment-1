from __future__ import annotations

import torch


class Momentum:
    name = "Momentum"

    def __init__(self, beta: float = 0.9) -> None:
        self.beta = beta
        self.velocity: torch.Tensor | None = None

    def step(self, point: torch.Tensor, gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
        if self.velocity is None:
            self.velocity = torch.zeros_like(point)
        self.velocity = self.beta * self.velocity + gradient
        return point - learning_rate * self.velocity

    def state_dict(self) -> dict[str, object]:
        return {"velocity": None if self.velocity is None else self.velocity.clone()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        velocity = state.get("velocity")
        self.velocity = None if velocity is None else velocity.clone()
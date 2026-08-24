from __future__ import annotations

import torch


class GradientDescent:
    name = "Gradient Descent"

    def step(self, point: torch.Tensor, gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return point - learning_rate * gradient

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        return None
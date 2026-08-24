from __future__ import annotations

import math

import torch


class SGD:
    name = "SGD"

    def __init__(self, noise_scale: float = 0.15) -> None:
        self.noise_scale = noise_scale

    def step(self, point: torch.Tensor, gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
        noisy_gradient = gradient + self.noise_scale * torch.randn_like(gradient)
        return point - learning_rate * noisy_gradient

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        return None


class MinibatchSGD:
    name = "Minibatch-SGD"

    def __init__(self, noise_scale: float = 0.15, batch_size: int = 16) -> None:
        self.noise_scale = noise_scale
        self.batch_size = max(1, batch_size)

    def step(self, point: torch.Tensor, gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
        effective_noise = self.noise_scale / math.sqrt(float(self.batch_size))
        noisy_gradient = gradient + effective_noise * torch.randn_like(gradient)
        return point - learning_rate * noisy_gradient

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        return None
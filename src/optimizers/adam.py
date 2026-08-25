from __future__ import annotations

import torch


class Adam:
    name = "Adam"

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.t = 0

    def step(self, point: torch.Tensor, gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
        if self.m is None or self.v is None:
            self.m = torch.zeros_like(point)
            self.v = torch.zeros_like(point)

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient * gradient)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return point - learning_rate * m_hat / (torch.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict[str, object]:
        return {
            "m": None if self.m is None else self.m.clone(),
            "v": None if self.v is None else self.v.clone(),
            "t": self.t,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        m = state.get("m")
        v = state.get("v")
        self.m = None if m is None else m.clone()
        self.v = None if v is None else v.clone()
        self.t = int(state.get("t", 0))
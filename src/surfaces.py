from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch

TensorFn = Callable[[torch.Tensor], torch.Tensor]
GridFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    torch_fn: TensorFn
    grid_fn: GridFn
    start: Tuple[float, float]
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    default_learning_rate: float
    minimum_point: Optional[Tuple[float, float]] = None
    minimum_tolerance: float = 1e-3
    escape_threshold: Optional[float] = None
    escape_measure: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def is_target_reached(self, point: torch.Tensor) -> bool:
        if self.minimum_point is not None:
            minimum = torch.tensor(self.minimum_point, dtype=point.dtype, device=point.device)
            return bool(torch.linalg.norm(point - minimum) <= self.minimum_tolerance)

        if self.escape_threshold is not None and self.escape_measure is not None:
            return bool(self.escape_measure(point) >= self.escape_threshold)

        return False


def bowl_torch(point: torch.Tensor) -> torch.Tensor:
    x, y = point[0], point[1]
    return x * x + y * y


def bowl_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * x + y * y


def ravine_torch(point: torch.Tensor) -> torch.Tensor:
    x, y = point[0], point[1]
    return x * x + 200.0 * y * y


def ravine_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * x + 200.0 * y * y


def saddle_torch(point: torch.Tensor) -> torch.Tensor:
    x, y = point[0], point[1]
    return x * x - y * y


def saddle_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * x - y * y


def rosenbrock_torch(point: torch.Tensor) -> torch.Tensor:
    x, y = point[0], point[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


def rosenbrock_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


Q1_SURFACES = [
    SurfaceSpec(
        name="Bowl",
        torch_fn=bowl_torch,
        grid_fn=bowl_grid,
        start=(-4.0, 4.0),
        xlim=(-5.0, 5.0),
        ylim=(-5.0, 5.0),
        default_learning_rate=0.08,
        minimum_point=(0.0, 0.0),
        minimum_tolerance=1e-3,
    ),
    SurfaceSpec(
        name="Ravine",
        torch_fn=ravine_torch,
        grid_fn=ravine_grid,
        start=(-4.0, 3.0),
        xlim=(-5.0, 5.0),
        ylim=(-4.0, 4.0),
        default_learning_rate=0.005,
        minimum_point=(0.0, 0.0),
        minimum_tolerance=1e-3,
    ),
    SurfaceSpec(
        name="Saddle",
        torch_fn=saddle_torch,
        grid_fn=saddle_grid,
        start=(-1.5, 0.001),
        xlim=(-3.0, 3.0),
        ylim=(-3.0, 3.0),
        default_learning_rate=0.05,
        escape_threshold=0.25,
        escape_measure=lambda point: torch.abs(point[1]),
    ),
]

Q2_SURFACES = [
    SurfaceSpec(
        name="Rosenbrock",
        torch_fn=rosenbrock_torch,
        grid_fn=rosenbrock_grid,
        start=(-1.5, 1.5),
        xlim=(-2.0, 2.0),
        ylim=(-1.0, 3.0),
        default_learning_rate=0.002,
        minimum_point=(1.0, 1.0),
        minimum_tolerance=1e-2,
    ),
]
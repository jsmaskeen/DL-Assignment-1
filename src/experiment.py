from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import hashlib

import numpy as np
import torch

from surfaces import SurfaceSpec
from viz import save_contour_plot, save_results_csv, save_trajectory_gif


@dataclass
class RunResult:
    surface: str
    optimizer_name: str
    trajectory: np.ndarray
    losses: np.ndarray
    reached_target: bool
    steps: int
    final_loss: float
    final_point: tuple[float, float]


OptimizerFactory = Callable[[], object]


def _is_offscreen(point: torch.Tensor, surface: SurfaceSpec) -> bool:
    x_value = float(point[0].item())
    y_value = float(point[1].item())
    return x_value < surface.xlim[0] or x_value > surface.xlim[1] or y_value < surface.ylim[0] or y_value > surface.ylim[1]


def _optimizer_state_snapshot(optimizer: object) -> dict[str, object] | None:
    state_dict = getattr(optimizer, "state_dict", None)
    if callable(state_dict):
        return state_dict()
    return None


def _optimizer_state_restore(optimizer: object, state: dict[str, object] | None) -> None:
    if state is None:
        return
    load_state_dict = getattr(optimizer, "load_state_dict", None)
    if callable(load_state_dict):
        load_state_dict(state)


def _stable_seed(surface_name: str, optimizer_name: str) -> int:
    digest = hashlib.sha256(f"{surface_name}:{optimizer_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def run_optimizer_on_surface(
    surface: SurfaceSpec,
    optimizer: object,
    learning_rate: float,
    max_steps: int = 1000,
    seed: int | None = None,
) -> RunResult:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    point = torch.tensor(surface.start, dtype=torch.float32)
    trajectory: list[np.ndarray] = [point.numpy().copy()]
    losses: list[float] = []
    current_lr = learning_rate
    reached_target = False
    min_learning_rate = 1e-12

    for _ in range(max_steps):
        point = point.detach().requires_grad_(True)
        loss = surface.torch_fn(point)
        loss.backward()
        gradient = point.grad.detach()

        snapshot = _optimizer_state_snapshot(optimizer)
        next_point = point.detach()

        while True:
            tentative_point = optimizer.step(point.detach(), gradient, current_lr)
            if torch.isfinite(tentative_point).all() and not _is_offscreen(tentative_point, surface):
                next_point = tentative_point
                break

            current_lr *= 0.5
            _optimizer_state_restore(optimizer, snapshot)

            if current_lr < min_learning_rate:
                next_point = point.detach()
                break

        trajectory.append(next_point.detach().cpu().numpy().copy())
        losses.append(float(loss.item()))
        point = next_point.detach()

        if surface.is_target_reached(point):
            reached_target = True
            break

    final_point = (float(point[0].item()), float(point[1].item()))
    final_loss = float(surface.torch_fn(point).item())
    return RunResult(
        surface=surface.name,
        optimizer_name=getattr(optimizer, "name", optimizer.__class__.__name__),
        trajectory=np.asarray(trajectory, dtype=np.float32),
        losses=np.asarray(losses, dtype=np.float32),
        reached_target=reached_target,
        steps=len(trajectory) - 1,
        final_loss=final_loss,
        final_point=final_point,
    )


def run_task_suite(
    surfaces: Sequence[SurfaceSpec],
    optimizer_factories: Sequence[tuple[str, OptimizerFactory]],
    output_root: Path,
    max_steps: int = 1000,
) -> list[RunResult]:
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []

    for surface in surfaces:
        surface_runs: list[RunResult] = []
        for optimizer_name, factory in optimizer_factories:
            optimizer = factory()
            seed = _stable_seed(surface.name, optimizer_name)
            run_result = run_optimizer_on_surface(
                surface=surface,
                optimizer=optimizer,
                learning_rate=surface.default_learning_rate,
                max_steps=max_steps,
                seed=seed,
            )
            run_result.optimizer_name = optimizer_name
            results.append(run_result)
            surface_runs.append(run_result)

        surface_folder = output_root / surface.name.lower().replace(" ", "_")
        save_contour_plot(surface, surface_runs, surface_folder / "plots" / f"{surface.name.lower().replace(' ', '_')}.png")
        save_trajectory_gif(surface, surface_runs, surface_folder / "gifs" / f"{surface.name.lower().replace(' ', '_')}.gif")

    save_results_csv(
        [
            {
                "surface": result.surface,
                "optimizer": result.optimizer_name,
                "reached_target": result.reached_target,
                "steps": result.steps,
                "final_loss": result.final_loss,
                "final_x": result.final_point[0],
                "final_y": result.final_point[1],
            }
            for result in results
        ],
        output_root / "tables" / "results.csv",
    )

    return results
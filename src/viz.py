from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from surfaces import SurfaceSpec


def _evaluate_grid(surface: SurfaceSpec, resolution: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.linspace(surface.xlim[0], surface.xlim[1], resolution)
    y_values = np.linspace(surface.ylim[0], surface.ylim[1], resolution)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_z = surface.grid_fn(grid_x, grid_y)
    return grid_x, grid_y, grid_z


def save_contour_plot(surface: SurfaceSpec, runs: Sequence["RunResult"], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_x, grid_y, grid_z = _evaluate_grid(surface)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.contour(grid_x, grid_y, grid_z, levels=35, cmap="viridis")

    for run in runs:
        path = run.trajectory
        ax.plot(path[:, 0], path[:, 1], linewidth=1.8, label=run.optimizer_name)
        ax.scatter(path[-1, 0], path[-1, 1], s=25)

    ax.set_xlim(*surface.xlim)
    ax.set_ylim(*surface.ylim)
    ax.set_title(f"{surface.name} trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_trajectory_gif(surface: SurfaceSpec, runs: Sequence["RunResult"], output_path: Path, fps: int = 12) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_x, grid_y, grid_z = _evaluate_grid(surface)
    max_length = max(run.trajectory.shape[0] for run in runs)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.contour(grid_x, grid_y, grid_z, levels=35, cmap="viridis")

    line_artists = []
    point_artists = []
    for run in runs:
        (line,) = ax.plot([], [], linewidth=2.0, label=run.optimizer_name)
        point = ax.scatter([], [], s=40)
        line_artists.append(line)
        point_artists.append(point)

    step_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, bbox=dict(facecolor="white", alpha=0.8))
    ax.set_xlim(*surface.xlim)
    ax.set_ylim(*surface.ylim)
    ax.set_title(f"{surface.name} optimizer comparison")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    def update(frame_index: int):
        for line, point, run in zip(line_artists, point_artists, runs):
            upper = min(frame_index + 1, run.trajectory.shape[0])
            path = run.trajectory[:upper]
            line.set_data(path[:, 0], path[:, 1])
            point.set_offsets(path[-1])
        step_text.set_text(f"step {frame_index}")
        return [*line_artists, *point_artists, step_text]

    animation = FuncAnimation(fig, update, frames=max_length, interval=1000 / fps, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


import csv

def save_results_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    
    # Capitalize headers for the CSV
    headers = ["Surface", "Optimizer", "Reached target", "Steps", "Final loss", "Final x", "Final y"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["surface"],
                row["optimizer"],
                "Yes" if bool(row["reached_target"]) else "No",
                row["steps"],
                f'{float(row["final_loss"]):.6f}',
                f'{float(row["final_x"]):.6f}',
                f'{float(row["final_y"]):.6f}',
            ])


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment import RunResult
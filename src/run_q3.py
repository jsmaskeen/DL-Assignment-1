from __future__ import annotations
import math
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from optimizers.adam import Adam
from experiment import run_optimizer_on_surface, RunResult
from surfaces import Q2_SURFACES
from viz import _evaluate_grid
import csv

class ScheduledAdam(Adam):
    def __init__(self, schedule_fn, name):
        super().__init__()
        self.schedule_fn = schedule_fn
        self.name = name
        self.lr_history = []

    def step(self, point, gradient, learning_rate):
        # Override the learning_rate with our scheduled rate
        lr = self.schedule_fn(self.t)
        self.lr_history.append(lr)
        return super().step(point, gradient, lr)

def get_schedules(eta_0: float, total_steps: int):
    eta_min = eta_0 * 0.01
    warmup = int(total_steps * 0.1)

    def constant(t): return eta_0
    def step_decay(t): return eta_0 * (0.1 ** (t // 600))
    def exp_decay(t): return eta_0 * (0.01 ** (t / total_steps))
    def cosine(t): return eta_min + 0.5 * (eta_0 - eta_min) * (1 + math.cos(math.pi * t / total_steps))
    def warmup_cosine(t):
        if t <= warmup:
            return (t / max(1, warmup)) * eta_0
        return eta_min + 0.5 * (eta_0 - eta_min) * (1 + math.cos(math.pi * (t - warmup) / (total_steps - warmup)))

    return [
        ("Constant", constant),
        ("Step Decay", step_decay),
        ("Exponential Decay", exp_decay),
        ("Cosine Annealing", cosine),
        ("Linear Warmup + Cosine", warmup_cosine),
    ]

def save_schedule_dual_gif(surface, runs: Sequence[RunResult], output_path: Path, fps: int = 30):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_x, grid_y, grid_z = _evaluate_grid(surface)
    max_length = max(run.trajectory.shape[0] for run in runs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)
    ax1.contour(grid_x, grid_y, grid_z, levels=35, cmap="viridis")

    line_artists = []
    point_artists = []
    lr_line_artists = []
    lr_point_artists = []
    for run in runs:
        (line,) = ax1.plot([], [], linewidth=1.5, label=run.optimizer_name)
        point = ax1.scatter([], [], s=25)
        (lr_line,) = ax2.plot([], [], linewidth=1.8, label=run.optimizer_name)
        lr_point = ax2.scatter([], [], s=25)

        line_artists.append(line)
        point_artists.append(point)
        lr_line_artists.append(lr_line)
        lr_point_artists.append(lr_point)

    step_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, fontsize=11, bbox=dict(facecolor="white", alpha=0.8))
    ax1.set_xlim(*surface.xlim)
    ax1.set_ylim(*surface.ylim)
    ax1.set_title(f"Rosenbrock Trajectories")
    ax1.legend(loc="upper right", fontsize=7)

    ax2.set_xlim(0, max_length)
    # Estimate max LR to set y limit
    max_lr = max(max(run.lr_history) for run in runs if hasattr(run, "lr_history") and run.lr_history) if runs else 0.01
    ax2.set_ylim(0, max_lr * 1.1)
    ax2.set_title("Learning Rate Schedules η_t")
    ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()

    def update(frame_index: int):
        # We sample 1000 frames evenly from max_length
        step = int(frame_index * (max_length / 1000)) if max_length > 1000 else frame_index
        if step >= max_length: step = max_length - 1

        for i, run in enumerate(runs):
            upper = min(step + 1, run.trajectory.shape[0])
            path = run.trajectory[:upper]
            line_artists[i].set_data(path[:, 0], path[:, 1])
            point_artists[i].set_offsets(path[-1])

            if hasattr(run, "lr_history") and run.lr_history:
                lrs = run.lr_history[:upper]
                lr_line_artists[i].set_data(range(len(lrs)), lrs)
                lr_point_artists[i].set_offsets((len(lrs)-1, lrs[-1]))

        step_text.set_text(f"step {step}")
        return [*line_artists, *point_artists, *lr_line_artists, *lr_point_artists, step_text]

    frames = min(max_length, 1000)
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def save_schedule_loss_plot(runs: Sequence[RunResult], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    for run in runs:
        ax.plot(run.losses, label=run.optimizer_name, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_title("Loss vs. Step across LR Schedules on Rosenbrock")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

def run_schedule_experiment(surface, schedules, output_root: Path, max_steps: int = 5000):
    output_root.mkdir(parents=True, exist_ok=True)
    results = []

    for name, fn in schedules:
        opt = ScheduledAdam(fn, name)
        # pass dummy lr=1.0 since ScheduledAdam ignores it
        res = run_optimizer_on_surface(surface, opt, learning_rate=1.0, max_steps=max_steps, seed=42)
        res.lr_history = opt.lr_history
        results.append(res)

    save_schedule_dual_gif(surface, results, output_root / "gifs" / "schedule_trajectories.gif")
    save_schedule_loss_plot(results, output_root / "plots" / "loss_vs_step.png")

    with open(output_root / "tables" / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Schedule", "Final loss", "Final x", "Final y", "Steps"])
        for res in results:
            writer.writerow([
                res.optimizer_name,
                f"{res.final_loss:.6f}",
                f"{res.final_point[0]:.6f}",
                f"{res.final_point[1]:.6f}",
                str(res.steps)
            ])

    return results

def main():
    schedules = get_schedules(0.5, 2000)
    rosenbrock = next(s for s in Q2_SURFACES if s.name == "Rosenbrock")
    run_schedule_experiment(rosenbrock, schedules, Path("../artifacts/Q3_lr_schedules"), 2000)

if __name__ == "__main__":
    main()

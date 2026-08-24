from __future__ import annotations

from pathlib import Path

from experiment import run_task_suite
from optimizers import Adam, GradientDescent, Momentum, MinibatchSGD, SGD
from surfaces import Q1_SURFACES


def build_optimizer_factories() -> list[tuple[str, object]]:
    return [
        ("Gradient Descent", GradientDescent),
        ("Momentum", Momentum),
        ("SGD", SGD),
        ("Minibatch-SGD", MinibatchSGD),
        ("Adam", Adam),
    ]


def main() -> None:
    output_root = Path("task_01_artifacts")
    run_task_suite(Q1_SURFACES, build_optimizer_factories(), output_root)


if __name__ == "__main__":
    main()
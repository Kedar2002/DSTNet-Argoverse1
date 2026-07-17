"""
evaluation.benchmark

Benchmark utilities for DSTNet.

Runs evaluation over a dataset and reports
standard motion forecasting metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from engine.evaluator import Evaluator

class Benchmark:
    """
    Benchmark runner.
    """

    def __init__(
        self,
        evaluator: Evaluator,
    ) -> None:

        self.evaluator = evaluator

    def run(
        self,
    ) -> dict[str, float]:
        """
        Run benchmark.
        """

        start = perf_counter()

        metrics = self.evaluator.evaluate()

        elapsed = perf_counter() - start

        metrics["evaluation_time"] = elapsed

        return metrics

    @staticmethod
    def summary(
        metrics: dict[str, float],
    ) -> None:

        print()

        print("=" * 60)

        print("DSTNet Benchmark")

        print("=" * 60)

        for key, value in metrics.items():

            print(

                f"{key:20s}: {value:.6f}"

            )

        print("=" * 60)

        print()

    @staticmethod
    def save(
        metrics: dict[str, float],
        path: str | Path,
    ) -> None:
        """
        Save benchmark results.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(

                metrics,

                f,

                indent=4,

            )

def benchmark(
    evaluator: Evaluator,
    *,
    save_path: str | None = None,
) -> dict[str, float]:
    """
    Run benchmark.
    """

    runner = Benchmark(
        evaluator,
    )

    metrics = runner.run()

    runner.summary(
        metrics,
    )

    if save_path is not None:

        runner.save(
            metrics,
            save_path,
        )

    return metrics



#!/usr/bin/env python3
"""Run the complete NVDA news-impact ARC idea 2 workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import DEFAULT_BARS, DEFAULT_NEWS, DEFAULT_OUTPUT, DEFAULT_QUOTES, HERE


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=HERE.parent)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "market", "dataset", "transformers", "models", "report"],
        default="all",
    )
    parser.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    parser.add_argument("--bars-dir", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--quotes-dir", type=Path, default=DEFAULT_QUOTES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--include-quotes", action="store_true")
    parser.add_argument("--include-indirect", action="store_true")
    parser.add_argument("--include-transformers", action="store_true")
    parser.add_argument("--finbert-model", default="ProsusAI/finbert")
    parser.add_argument("--finbert-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--force-market-cache", action="store_true")
    parser.add_argument(
        "--separate-period-models",
        action="store_true",
        help="Also train independent regular, premarket, after-hours, and overnight models",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.workers < 1 or args.finbert_batch_size < 1:
        parser.error("Worker and batch counts must be positive")
    output = args.output_dir.resolve()
    python = sys.executable

    if args.stage in {"all", "market"}:
        command = [
            python,
            str(HERE / "market_data.py"),
            "--bars-dir",
            str(args.bars_dir),
            "--quotes-dir",
            str(args.quotes_dir),
            "--output-dir",
            str(output),
            "--workers",
            str(args.workers),
        ]
        if args.include_quotes:
            command.append("--include-quotes")
        if args.force_market_cache:
            command.append("--force")
        run(command)

    if args.stage in {"all", "dataset"}:
        command = [
            python,
            str(HERE / "build_dataset.py"),
            "--news",
            str(args.news),
            "--market",
            str(output / "market_minutes.parquet"),
            "--output-dir",
            str(output),
        ]
        if args.include_indirect:
            command.append("--include-indirect")
        run(command)

    run_transformers = args.stage == "transformers" or (
        args.stage == "all" and args.include_transformers
    )
    if run_transformers:
        run(
            [
                python,
                str(HERE / "extract_transformer_features.py"),
                "--events",
                str(output / "news_events.parquet"),
                "--output",
                str(output / "transformer_features.parquet"),
                "--model",
                args.finbert_model,
                "--batch-size",
                str(args.finbert_batch_size),
                "--device",
                args.device,
            ]
        )

    if args.stage in {"all", "models"}:
        command = [
            python,
            str(HERE / "train_models.py"),
            "--events",
            str(output / "news_events.parquet"),
            "--controls",
            str(output / "no_news_controls.parquet"),
            "--output-dir",
            str(output / "models_output"),
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
        ]
        transformer_path = output / "transformer_features.parquet"
        if transformer_path.is_file():
            command.extend(["--transformer-features", str(transformer_path)])
        run(command)
        if args.separate_period_models:
            for period in ("regular", "premarket", "afterhours", "overnight"):
                period_command = command.copy()
                output_index = period_command.index("--output-dir") + 1
                period_command[output_index] = str(output / f"models_output_{period}")
                period_command.extend(["--market-periods", period])
                run(period_command)

    if args.stage in {"all", "report"}:
        run(
            [
                python,
                str(HERE / "generate_report.py"),
                "--output-dir",
                str(output),
            ]
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Chronologically fine-tune FinBERT for one binary NVDA event task.

This is deliberately separate from frozen feature extraction. It is an optional
HPC experiment and should be interpreted only against the already-computed
market-only and frozen-text baselines.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from common import DEFAULT_OUTPUT, atomic_csv, atomic_json, file_sha256


BINARY_TASKS = {
    "impact_5m": "impact_label_5m",
    "impact_15m": "impact_label_15m",
    "impact_60m": "impact_label_60m",
    "direction_15m": "direction_label_15m",
    "material_direction_15m": "material_direction_label_15m",
    "continuation_1m_to_15m": "continuation_direction_1m_to_15m",
    "continuation_5m_to_60m": "continuation_direction_5m_to_60m",
}


def chronological_three_way_split(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(frame.outcome_session_date.astype(str).unique()))
    if len(dates) < 10:
        raise ValueError("At least ten eligible outcome dates are required")
    train_end = max(1, min(len(dates) - 2, int(np.floor(len(dates) * train_fraction))))
    validation_end = max(
        train_end + 1,
        min(
            len(dates) - 1,
            int(np.floor(len(dates) * (train_fraction + validation_fraction))),
        ),
    )
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])
    values = frame.outcome_session_date.astype(str)
    return (
        frame.loc[values.isin(train_dates)].copy(),
        frame.loc[values.isin(validation_dates)].copy(),
        frame.loc[values.isin(test_dates)].copy(),
    )


def binary_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "average_precision": float(average_precision_score(actual, probabilities)),
        "brier": float(brier_score_loss(actual, probabilities)),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
    }


def day_block_auc_interval(
    actual: np.ndarray,
    probabilities: np.ndarray,
    dates: np.ndarray,
    iterations: int,
) -> dict[str, float | int]:
    unique_dates = np.unique(dates)
    rng = np.random.default_rng(0)
    values: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        indices = np.concatenate([np.flatnonzero(dates == date) for date in sampled])
        if len(np.unique(actual[indices])) < 2:
            continue
        values.append(float(roc_auc_score(actual[indices], probabilities[indices])))
    return {
        "low": float(np.quantile(values, 0.025)) if values else np.nan,
        "high": float(np.quantile(values, 0.975)) if values else np.nan,
        "valid_resamples": len(values),
    }


def fine_tune(
    events_path: Path,
    output_dir: Path,
    task: str,
    model_name: str,
    market_periods: set[str] | None,
    max_length: int,
    batch_size: int,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-transformers.txt before fine-tuning"
        ) from exc

    target = BINARY_TASKS[task]
    frame = pd.read_parquet(events_path)
    eligible = frame.loc[
        frame.market_data_available.fillna(False)
        & pd.to_numeric(frame[target], errors="coerce").notna()
    ].copy()
    if market_periods is not None:
        eligible = eligible.loc[
            eligible.market_period.astype(str).isin(market_periods)
        ].copy()
    eligible[target] = pd.to_numeric(eligible[target], errors="raise").astype(int)
    eligible = eligible.sort_values(["analysis_time_utc", "event_id"]).reset_index(drop=True)
    train, validation, test = chronological_three_way_split(eligible, 0.70, 0.15)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if split[target].nunique() < 2:
            raise ValueError(f"{name} split has fewer than two target classes")

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

    class EventTextDataset(torch.utils.data.Dataset):
        def __init__(self, source: pd.DataFrame):
            self.encodings = tokenizer(
                source.combined_text.fillna("").astype(str).tolist(),
                truncation=True,
                max_length=max_length,
            )
            self.labels = source[target].astype(int).tolist()

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, object]:
            item = {name: values[index] for name, values in self.encodings.items()}
            item["labels"] = self.labels[index]
            return item

    train_dataset = EventTextDataset(train)
    validation_dataset = EventTextDataset(validation)
    test_dataset = EventTextDataset(test)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        trust_remote_code=False,
    )
    model.config.id2label = {0: "negative_class", 1: "positive_class"}
    model.config.label2id = {"negative_class": 0, "positive_class": 1}

    def compute_metrics(evaluation: object) -> dict[str, float]:
        logits, labels = evaluation
        logits = np.asarray(logits)
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return binary_metrics(np.asarray(labels, dtype=int), probabilities[:, 1])

    checkpoint_dir = output_dir / "checkpoints"
    argument_values: dict[str, object] = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "gradient_accumulation_steps": 1,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "roc_auc",
        "greater_is_better": True,
        "save_total_limit": 1,
        "seed": seed,
        "data_seed": seed,
        "report_to": [],
        "fp16": bool(torch.cuda.is_available()),
    }
    argument_names = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in argument_names:
        argument_values["eval_strategy"] = "epoch"
    else:
        argument_values["evaluation_strategy"] = "epoch"
    training_arguments = TrainingArguments(**argument_values)

    trainer_values: dict[str, object] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=patience)],
    }
    trainer_names = inspect.signature(Trainer).parameters
    if "processing_class" in trainer_names:
        trainer_values["processing_class"] = tokenizer
    else:
        trainer_values["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_values)
    training_result = trainer.train()
    prediction = trainer.predict(test_dataset)
    logits = np.asarray(prediction.predictions)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    positive_probability = probabilities[:, 1]
    actual = test[target].to_numpy(dtype=int)
    metrics = binary_metrics(actual, positive_probability)
    interval = day_block_auc_interval(
        actual,
        positive_probability,
        test.outcome_session_date.astype(str).to_numpy(),
        bootstrap_iterations,
    )

    model_output = output_dir / "best_model"
    trainer.save_model(model_output)
    tokenizer.save_pretrained(model_output)
    predictions = test[
        [
            "event_id",
            "outcome_session_date",
            "analysis_time_utc",
            "market_period",
            "event_category",
            "combined_text",
        ]
    ].copy()
    predictions["actual"] = actual
    predictions["positive_probability"] = positive_probability
    predictions["prediction"] = (positive_probability >= 0.5).astype(int)
    atomic_csv(predictions, output_dir / "test_predictions.csv")

    payload: dict[str, object] = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "target": target,
        "market_period_filter": sorted(market_periods) if market_periods else None,
        "input": {
            "path": str(events_path.resolve()),
            "sha256": file_sha256(events_path),
        },
        "model": {
            "pretrained_model": model_name,
            "resolved_commit": getattr(model.config, "_commit_hash", None),
            "max_length": max_length,
            "epochs_requested": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": patience,
            "seed": seed,
            "output": str(model_output.resolve()),
        },
        "sample": {
            name: {
                "rows": len(split),
                "dates": int(split.outcome_session_date.nunique()),
                "date_range": [
                    str(split.outcome_session_date.min()),
                    str(split.outcome_session_date.max()),
                ],
                "positive_rate": float(split[target].mean()),
            }
            for name, split in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "held_out_test_metrics": metrics,
        "day_block_roc_auc_95_ci": interval,
        "training": {
            "global_steps": int(training_result.global_step),
            "training_loss": float(training_result.training_loss),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_metric": trainer.state.best_metric,
        },
        "look_ahead_policy": (
            "The pretrained checkpoint must predate the October 2025 evaluation period; "
            "only supplied event text is tokenized. Complete outcome dates are split "
            "chronologically before fine-tuning."
        ),
    }
    atomic_json(payload, output_dir / "metrics.json")
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", type=Path, default=DEFAULT_OUTPUT / "news_events.parquet"
    )
    parser.add_argument("--task", choices=sorted(BINARY_TASKS), default="material_direction_15m")
    parser.add_argument("--model", default="ProsusAI/finbert")
    parser.add_argument("--market-periods")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if not args.events.is_file():
        parser.error(f"Event dataset does not exist: {args.events}")
    if args.batch_size < 1 or args.max_length < 8 or args.epochs <= 0:
        parser.error("Invalid training dimensions")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.patience < 1:
        parser.error("Invalid optimizer or early-stopping settings")
    market_periods = (
        {item.strip() for item in args.market_periods.split(",") if item.strip()}
        if args.market_periods
        else None
    )
    output_dir = args.output_dir or DEFAULT_OUTPUT / "finetuned" / args.task
    payload = fine_tune(
        events_path=args.events,
        output_dir=output_dir,
        task=args.task,
        model_name=args.model,
        market_periods=market_periods,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    print(json.dumps({"status": "complete", "metrics": payload["held_out_test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train and evaluate the ARC event-level TF-IDF classifier.

The input is the consolidated ``news_price_events.parquet`` (or optional CSV)
produced by ``build_dataset.py``. Per-day minute quote cache files are price
inputs to the dataset builder; they do not contain text or outcome labels and
are therefore not direct inputs to this trainer.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = HERE / "output/news_price_events.parquet"
DEFAULT_OUTPUT = HERE / "tfidf_output"
LABELS = np.array([-1, 0, 1], dtype=int)
CLASS_NAMES = {-1: "down", 0: "neutral", 1: "up"}
TOKEN_PATTERN = r"(?u)\b(?=\w*[a-zA-Z])\w\w+\b"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boolean_series(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.BooleanDtype) or pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def resolve_event_path(path: Path) -> Path:
    """Accept either an event dataset or its containing output folder."""
    candidate = path
    if path.is_dir():
        candidate = path / "news_price_events.parquet"
        csv_candidate = path / "news_price_events.csv"
        if not candidate.is_file() and csv_candidate.is_file():
            candidate = csv_candidate
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Event dataset does not exist: {candidate}. Run build_dataset.py first "
            "and pass news_price_events.parquet (or its CSV copy), not a "
            ".minute_cache directory."
        )
    return candidate


def read_event_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Event dataset must be Parquet or CSV: {path}")


def prepare_training_frame(
    raw: pd.DataFrame,
    horizon: int,
    include_contaminated: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate and filter event rows for one prediction horizon."""
    suffix = f"{horizon}m"
    label_column = f"label_code_{suffix}"
    available_column = f"label_available_{suffix}"
    clean_column = f"clean_{suffix}"
    required = {
        "news_cluster_id",
        "session_date",
        "cluster_start_utc",
        "combined_text",
        "chronological_split",
        label_column,
        available_column,
        clean_column,
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            "Event dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    frame = raw.copy()
    audit = {"input_events": int(len(frame))}
    frame["chronological_split"] = (
        frame.chronological_split.astype(str).str.strip().str.lower()
    )
    known_splits = {"train", "validation", "test"}
    unknown = sorted(set(frame.chronological_split) - known_splits)
    if unknown:
        raise ValueError(f"Unknown chronological_split values: {unknown}")

    available = boolean_series(frame[available_column])
    clean = boolean_series(frame[clean_column])
    audit["unavailable_events_removed"] = int((~available).sum())
    frame = frame.loc[available].copy()
    clean = clean.loc[available]
    audit["contaminated_events_removed"] = (
        0 if include_contaminated else int((~clean).sum())
    )
    if not include_contaminated:
        frame = frame.loc[clean].copy()

    frame["combined_text"] = frame.combined_text.fillna("").astype(str).str.strip()
    empty_text = frame.combined_text.eq("")
    audit["empty_text_events_removed"] = int(empty_text.sum())
    frame = frame.loc[~empty_text].copy()

    numeric_labels = pd.to_numeric(frame[label_column], errors="coerce")
    invalid_label = numeric_labels.isna() | ~numeric_labels.isin(LABELS)
    audit["invalid_label_events_removed"] = int(invalid_label.sum())
    frame = frame.loc[~invalid_label].copy()
    frame["target_label"] = numeric_labels.loc[~invalid_label].astype(int)
    frame["target_name"] = frame.target_label.map(CLASS_NAMES)

    frame["cluster_start_utc"] = pd.to_datetime(
        frame.cluster_start_utc, utc=True, errors="coerce"
    )
    if frame.cluster_start_utc.isna().any():
        raise ValueError("Eligible rows contain invalid cluster_start_utc values")
    if frame.news_cluster_id.astype(str).duplicated().any():
        raise ValueError("Event dataset contains duplicate news_cluster_id values")

    split_counts_per_date = frame.groupby("session_date").chronological_split.nunique()
    if split_counts_per_date.gt(1).any():
        bad_dates = split_counts_per_date.loc[split_counts_per_date.gt(1)].index.tolist()
        raise ValueError(f"Trading dates cross chronological splits: {bad_dates[:5]}")

    frame = frame.sort_values(
        ["cluster_start_utc", "news_cluster_id"]
    ).reset_index(drop=True)
    split_order = frame.chronological_split.map(
        {"train": 0, "validation": 1, "test": 2}
    )
    if not split_order.is_monotonic_increasing:
        raise ValueError("Chronological splits are not ordered train, validation, test")
    missing_splits = sorted(known_splits - set(frame.chronological_split))
    if missing_splits:
        raise ValueError(f"No eligible rows remain in splits: {missing_splits}")

    audit["eligible_events"] = int(len(frame))
    return frame, audit


def make_model(
    c_value: float,
    min_df: int,
    max_df: float,
    max_features: int,
    ngram_max: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    token_pattern=TOKEN_PATTERN,
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    max_df=max_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=0,
                ),
            ),
        ]
    )


def aligned_probabilities(model: Pipeline, text: Iterable[str]) -> np.ndarray:
    raw = model.predict_proba(text)
    classes = model.named_steps["classifier"].classes_.astype(int)
    aligned = np.zeros((len(raw), len(LABELS)), dtype=float)
    for source_index, label in enumerate(classes):
        target = np.flatnonzero(LABELS == label)
        if len(target) == 1:
            aligned[:, int(target[0])] = raw[:, source_index]
    return aligned


def class_prior_probabilities(labels: np.ndarray, count: int) -> np.ndarray:
    probabilities = np.array([(labels == label).mean() for label in LABELS], dtype=float)
    return np.tile(probabilities, (count, 1))


def score_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = LABELS[np.argmax(probabilities, axis=1)]
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=LABELS)),
    }


def tune_regularization(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    c_grid: Iterable[float],
    min_df: int,
    max_df: float,
    max_features: int,
    ngram_max: int,
) -> tuple[float, pd.DataFrame]:
    """Select regularization on the fixed chronological validation split."""
    rows: list[dict[str, object]] = []
    train_labels = train.target_label.to_numpy(dtype=int)
    validation_labels = validation.target_label.to_numpy(dtype=int)
    for c_value in c_grid:
        model = make_model(c_value, min_df, max_df, max_features, ngram_max)
        model.fit(train.combined_text, train_labels)
        metrics = score_probabilities(
            validation_labels,
            aligned_probabilities(model, validation.combined_text),
        )
        rows.append(
            {
                "c": float(c_value),
                "training_events": int(len(train)),
                "validation_events": int(len(validation)),
                "tfidf_features": int(
                    len(model.named_steps["tfidf"].get_feature_names_out())
                ),
                **metrics,
            }
        )
    metrics_frame = pd.DataFrame(rows)
    ranked = metrics_frame.sort_values(
        ["macro_f1", "balanced_accuracy", "log_loss", "c"],
        ascending=[False, False, True, True],
    )
    return float(ranked.iloc[0].c), metrics_frame


def top_term_rows(model: Pipeline, count: int) -> list[dict[str, object]]:
    terms = model.named_steps["tfidf"].get_feature_names_out()
    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_
    classes = classifier.classes_.astype(int)
    if len(classes) == 2 and len(coefficients) == 1:
        coefficients = np.vstack([-coefficients[0], coefficients[0]])

    rows: list[dict[str, object]] = []
    for class_index, label in enumerate(classes):
        values = coefficients[class_index]
        for rank, term_index in enumerate(np.argsort(values)[::-1][:count], start=1):
            rows.append(
                {
                    "class_label": int(label),
                    "class_name": CLASS_NAMES[int(label)],
                    "rank": rank,
                    "term": str(terms[term_index]),
                    "coefficient": float(values[term_index]),
                }
            )
    return rows


def confusion_rows(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
) -> list[dict[str, object]]:
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    rows: list[dict[str, object]] = []
    for actual_index, actual_label in enumerate(LABELS):
        for predicted_index, predicted_label in enumerate(LABELS):
            rows.append(
                {
                    "model": model_name,
                    "actual_label": int(actual_label),
                    "actual_name": CLASS_NAMES[int(actual_label)],
                    "predicted_label": int(predicted_label),
                    "predicted_name": CLASS_NAMES[int(predicted_label)],
                    "count": int(matrix[actual_index, predicted_index]),
                }
            )
    return rows


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(frame.target_label.eq(label).sum())
        for label in LABELS
    }


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_joblib(model: Pipeline, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    joblib.dump(model, temporary)
    os.replace(temporary, path)


def output_targets(output_dir: Path) -> dict[str, Path]:
    return {
        "model": output_dir / "tfidf_model.joblib",
        "validation": output_dir / "validation_metrics.csv",
        "predictions": output_dir / "test_predictions.csv",
        "confusion": output_dir / "confusion_matrix.csv",
        "terms": output_dir / "top_terms_by_class.csv",
        "metrics": output_dir / "model_metrics.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--include-contaminated", action="store_true")
    parser.add_argument(
        "--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0]
    )
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-features", type=int, default=5_000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=2)
    parser.add_argument("--top-terms", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if any(value <= 0 for value in args.c_grid):
        parser.error("Every --c-grid value must be positive")
    if args.min_df < 1 or not 0 < args.max_df <= 1:
        parser.error("Invalid --min-df or --max-df")
    if args.max_features < 1 or args.top_terms < 1:
        parser.error("--max-features and --top-terms must be positive")

    event_path = resolve_event_path(args.input)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    targets = output_targets(output_dir)
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "Refusing to overwrite existing training outputs; pass --overwrite if "
            "intended:\n" + "\n".join(str(path) for path in existing)
        )

    source_hash_before = file_sha256(event_path)
    raw = read_event_data(event_path)
    events, audit = prepare_training_frame(
        raw, args.horizon, args.include_contaminated
    )
    partitions = {
        name: events.loc[events.chronological_split.eq(name)].copy()
        for name in ("train", "validation", "test")
    }
    train_labels = partitions["train"].target_label.to_numpy(dtype=int)
    missing_train_classes = sorted(set(LABELS) - set(train_labels))
    if missing_train_classes:
        raise ValueError(
            "Training split must contain all three target classes; missing "
            + ", ".join(str(value) for value in missing_train_classes)
        )

    selected_c, validation_metrics = tune_regularization(
        partitions["train"],
        partitions["validation"],
        args.c_grid,
        args.min_df,
        args.max_df,
        args.max_features,
        args.ngram_max,
    )

    development = pd.concat(
        [partitions["train"], partitions["validation"]], ignore_index=True
    ).sort_values(["cluster_start_utc", "news_cluster_id"])
    final_model = make_model(
        selected_c,
        args.min_df,
        args.max_df,
        args.max_features,
        args.ngram_max,
    )
    development_labels = development.target_label.to_numpy(dtype=int)
    final_model.fit(development.combined_text, development_labels)

    test = partitions["test"]
    test_labels = test.target_label.to_numpy(dtype=int)
    model_probabilities = aligned_probabilities(final_model, test.combined_text)
    prior_probabilities = class_prior_probabilities(development_labels, len(test))
    model_predictions = LABELS[np.argmax(model_probabilities, axis=1)]
    prior_predictions = LABELS[np.argmax(prior_probabilities, axis=1)]

    identifier_columns = [
        column
        for column in [
            "news_cluster_id",
            "session_date",
            "cluster_start_utc",
            "decision_time_utc",
            "article_count",
            "combined_text",
            "target_label",
            "target_name",
        ]
        if column in test.columns
    ]
    predictions = test[identifier_columns].copy()
    predictions["tfidf_prediction"] = model_predictions
    predictions["tfidf_prediction_name"] = pd.Series(model_predictions).map(CLASS_NAMES).to_numpy()
    predictions["class_prior_prediction"] = prior_predictions
    for index, label in enumerate(LABELS):
        name = CLASS_NAMES[int(label)]
        predictions[f"tfidf_probability_{name}"] = model_probabilities[:, index]
        predictions[f"class_prior_probability_{name}"] = prior_probabilities[:, index]

    confusion = confusion_rows(test_labels, model_predictions, "tfidf")
    confusion.extend(confusion_rows(test_labels, prior_predictions, "class_prior"))
    terms = pd.DataFrame(top_term_rows(final_model, args.top_terms))

    source_hash_after = file_sha256(event_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Event dataset changed during model training")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_joblib(final_model, targets["model"])
    atomic_csv(validation_metrics, targets["validation"])
    atomic_csv(predictions, targets["predictions"])
    atomic_csv(pd.DataFrame(confusion), targets["confusion"])
    atomic_csv(terms, targets["terms"])

    metrics: dict[str, object] = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "objective": (
            f"Predict the volatility/spread-adjusted {args.horizon}-minute NVDA "
            "event label from clustered news text."
        ),
        "unit_of_analysis": "one fixed-window news cluster",
        "input": {
            "path": str(event_path.resolve()),
            "sha256": source_hash_before,
        },
        "filtering": {
            "horizon_minutes": args.horizon,
            "clean_events_only": not args.include_contaminated,
            **audit,
        },
        "sample": {
            "events": int(len(events)),
            "date_range": [
                str(events.session_date.iloc[0]),
                str(events.session_date.iloc[-1]),
            ],
            "splits": {
                name: {
                    "events": int(len(frame)),
                    "class_counts": class_counts(frame),
                    "date_range": [
                        str(frame.session_date.iloc[0]),
                        str(frame.session_date.iloc[-1]),
                    ],
                }
                for name, frame in partitions.items()
            },
        },
        "tfidf": {
            "text_column": "combined_text",
            "lowercase": True,
            "strip_accents": "unicode",
            "stop_words": "english",
            "ngram_range": [1, args.ngram_max],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_features": args.max_features,
            "final_feature_count": int(
                len(final_model.named_steps["tfidf"].get_feature_names_out())
            ),
            "fit_policy": (
                "C selected with train-only fits on validation; final vocabulary and "
                "classifier fit on train plus validation; test used once."
            ),
        },
        "classifier": {
            "type": "class-balanced L2 logistic regression",
            "c_grid": list(args.c_grid),
            "selected_c": selected_c,
        },
        "held_out_test_metrics": {
            "class_prior": score_probabilities(test_labels, prior_probabilities),
            "tfidf": score_probabilities(test_labels, model_probabilities),
        },
        "package_versions": {
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
        "artifacts": {
            name: str(path.resolve()) for name, path in targets.items()
        },
        "limitations": [
            "TF-IDF coefficients are associations, not sentiment or causal effects.",
            "Labels depend on quote availability and the configured volatility/spread band.",
            "Clean horizons exclude observed later clusters but cannot exclude unobserved news.",
            "The saved joblib model should only be loaded from this trusted project output.",
        ],
    }
    atomic_json(metrics, targets["metrics"])
    print(
        json.dumps(
            {
                "status": "complete",
                "model": str(targets["model"]),
                "selected_c": selected_c,
                "test_metrics": metrics["held_out_test_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

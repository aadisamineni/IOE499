#!/usr/bin/env python3
"""Train chronological NVDA market/news models for impact and continuation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import DEFAULT_OUTPUT, atomic_csv, atomic_json, file_sha256


TOKEN_PATTERN = r"(?u)\b(?=\w*[a-zA-Z])\w\w+\b"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    target: str
    decision_delay: int = 0


DEFAULT_TASKS = [
    TaskSpec("impact_5m", "classification", "impact_label_5m"),
    TaskSpec("impact_15m", "classification", "impact_label_15m"),
    TaskSpec("impact_60m", "classification", "impact_label_60m"),
    TaskSpec("direction_15m", "classification", "direction_label_15m"),
    TaskSpec(
        "material_direction_15m",
        "classification",
        "material_direction_label_15m",
    ),
    TaskSpec(
        "continuation_1m_to_15m",
        "classification",
        "continuation_direction_1m_to_15m",
        decision_delay=1,
    ),
    TaskSpec(
        "continuation_5m_to_60m",
        "classification",
        "continuation_direction_5m_to_60m",
        decision_delay=5,
    ),
    TaskSpec("reaction_return_15m", "regression", "reaction_return_15m"),
    TaskSpec("reaction_return_60m", "regression", "reaction_return_60m"),
    TaskSpec(
        "continuation_return_5m_to_60m",
        "regression",
        "continuation_return_5m_to_60m",
        decision_delay=5,
    ),
]


MARKET_FEATURES = [
    "prior_return_1m",
    "prior_return_5m",
    "prior_return_15m",
    "prior_return_30m",
    "pre_volatility_30m",
    "log_prior_volume_mean_30m",
    "log_prior_trade_count_mean_30m",
    "pre_bar_volume",
    "pre_bar_trade_count",
    "pre_spread_bps",
    "pre_quote_updates",
    "pre_bid_size",
    "pre_ask_size",
    "minutes_from_open",
    "session_progress",
    "minute_of_day_sin",
    "minute_of_day_cos",
]

MARKET_CATEGORICAL_FEATURES = ["market_period"]

NEWS_NUMERIC_FEATURES = [
    "article_count",
    "duplicate_group_size_total",
    "prior_news_count_15m",
    "prior_news_count_60m",
    "prior_news_count_24h",
    "minutes_since_prior_event",
    "text_word_count",
    "text_char_count",
    "contains_number",
    "contains_percent",
    "contains_dollar",
    "max_similarity_prior_30d",
    "mean_similarity_prior_30d",
    "novelty_30d",
    "semantic_tag_count",
    "positive_cue_count",
    "negative_cue_count",
]

NEWS_CATEGORICAL_FEATURES = [
    "event_category",
    "fine_event_type",
    "source",
    "relevance_type",
]


def select_existing(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column in frame.columns and frame[column].notna().any()
    ]


def feature_groups(
    frame: pd.DataFrame,
    task: TaskSpec,
) -> dict[str, list[str]]:
    market = select_existing(frame, MARKET_FEATURES)
    if task.decision_delay:
        causal_post_news = [
            f"initial_return_{task.decision_delay}m",
            f"volume_ratio_{task.decision_delay}m",
            "overnight_gap_return",
        ]
        market.extend(select_existing(frame, causal_post_news))
    tag_features = sorted(column for column in frame.columns if column.startswith("tag_"))
    news_numeric = select_existing(frame, [*NEWS_NUMERIC_FEATURES, *tag_features])
    finbert = select_existing(
        frame,
        ["finbert_negative", "finbert_neutral", "finbert_positive"],
    )
    embeddings = sorted(
        column for column in frame.columns if column.startswith("embedding_")
    )
    return {
        "market": market,
        "market_categorical": select_existing(frame, MARKET_CATEGORICAL_FEATURES),
        "news_numeric": news_numeric,
        "news_categorical": select_existing(frame, NEWS_CATEGORICAL_FEATURES),
        "finbert": finbert,
        "embeddings": embeddings,
    }


def available_variants(groups: dict[str, list[str]]) -> list[str]:
    variants = [
        "market_only",
        "news_structured",
        "market_news_structured",
        "tfidf_only",
        "market_news_tfidf",
    ]
    if groups["finbert"]:
        variants.extend(["market_news_finbert", "market_news_tfidf_finbert"])
    if groups["embeddings"]:
        variants.extend(["market_news_embeddings", "full_text_model"])
    return variants


def make_preprocessor(
    variant: str,
    groups: dict[str, list[str]],
    max_tfidf_features: int,
    embedding_components: int,
) -> ColumnTransformer:
    use_market = variant in {
        "market_only",
        "market_news_structured",
        "market_news_tfidf",
        "market_news_finbert",
        "market_news_tfidf_finbert",
        "market_news_embeddings",
        "full_text_model",
    }
    use_structured_news = variant in {
        "news_structured",
        "market_news_structured",
        "market_news_tfidf",
        "market_news_finbert",
        "market_news_tfidf_finbert",
        "market_news_embeddings",
        "full_text_model",
    }
    use_tfidf = variant in {
        "tfidf_only",
        "market_news_tfidf",
        "market_news_tfidf_finbert",
        "full_text_model",
    }
    use_finbert = variant in {
        "market_news_finbert",
        "market_news_tfidf_finbert",
        "market_news_embeddings",
        "full_text_model",
    }
    use_embeddings = variant in {"market_news_embeddings", "full_text_model"}

    transformers: list[tuple[str, object, object]] = []
    numeric = []
    if use_market:
        numeric.extend(groups["market"])
    if use_structured_news:
        numeric.extend(groups["news_numeric"])
    if use_finbert:
        numeric.extend(groups["finbert"])
    numeric = list(dict.fromkeys(numeric))
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    categorical = []
    if use_market:
        categorical.extend(groups["market_categorical"])
    if use_structured_news:
        categorical.extend(groups["news_categorical"])
    categorical = list(dict.fromkeys(categorical))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if use_tfidf:
        transformers.append(
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    token_pattern=TOKEN_PATTERN,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_tfidf_features,
                    sublinear_tf=True,
                    norm="l2",
                ),
                "combined_text",
            )
        )
    if use_embeddings:
        components = min(embedding_components, len(groups["embeddings"]))
        transformers.append(
            (
                "embeddings",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=components, random_state=0)),
                    ]
                ),
                groups["embeddings"],
            )
        )
    if not transformers:
        raise ValueError(f"Variant {variant} has no usable features")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def make_model(
    task: TaskSpec,
    variant: str,
    groups: dict[str, list[str]],
    regularization: float,
    max_tfidf_features: int,
    embedding_components: int,
) -> Pipeline:
    preprocess = make_preprocessor(
        variant,
        groups,
        max_tfidf_features,
        embedding_components,
    )
    if task.kind == "classification":
        estimator = LogisticRegression(
            C=regularization,
            solver="liblinear",
            max_iter=3_000,
            class_weight=None,
            random_state=0,
        )
    else:
        estimator = Ridge(alpha=regularization)
    return Pipeline([("features", preprocess), ("model", estimator)])


def classification_metrics(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    probabilities = np.clip(probabilities.astype(float), 1e-7, 1 - 1e-7)
    predicted = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "brier": float(brier_score_loss(actual, probabilities)),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
    }
    if len(np.unique(actual)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(actual, probabilities)
        )
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    return metrics


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_series = pd.Series(actual)
    predicted_series = pd.Series(predicted)
    spearman = (
        float(actual_series.corr(predicted_series, method="spearman"))
        if actual_series.nunique() > 1 and predicted_series.nunique() > 1
        else np.nan
    )
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "spearman": spearman,
    }


def score_predictions(
    task: TaskSpec,
    actual: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    if task.kind == "classification":
        return classification_metrics(actual.astype(int), predictions)
    return regression_metrics(actual.astype(float), predictions)


def model_predictions(task: TaskSpec, model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    if task.kind == "classification":
        probabilities = model.predict_proba(frame)
        classes = model.named_steps["model"].classes_
        positive = np.flatnonzero(classes == 1)
        if len(positive) != 1:
            raise ValueError(f"Classifier lacks positive class: {classes}")
        return probabilities[:, int(positive[0])]
    return model.predict(frame)


def baseline_predictions(
    task: TaskSpec,
    train_target: np.ndarray,
    count: int,
) -> np.ndarray:
    value = float(np.mean(train_target))
    return np.full(count, value, dtype=float)


def development_test_split(
    frame: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(frame.outcome_session_date.astype(str).unique())
    if len(dates) < 5:
        raise ValueError("At least five eligible outcome dates are required")
    test_count = max(1, int(np.ceil(len(dates) * test_fraction)))
    test_count = min(test_count, len(dates) - 2)
    test_dates = set(dates[-test_count:])
    development = frame.loc[~frame.outcome_session_date.astype(str).isin(test_dates)].copy()
    test = frame.loc[frame.outcome_session_date.astype(str).isin(test_dates)].copy()
    return development, test


def expanding_date_folds(
    frame: pd.DataFrame,
    splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = np.array(sorted(frame.outcome_session_date.astype(str).unique()))
    if len(dates) < 6:
        return []
    first_validation = max(2, int(np.floor(len(dates) * 0.50)))
    boundaries = np.linspace(first_validation, len(dates), splits + 1, dtype=int)
    boundaries = np.unique(boundaries)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    date_values = frame.outcome_session_date.astype(str).to_numpy()
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        train_dates = set(dates[:start])
        validation_dates = set(dates[start:stop])
        train_index = np.flatnonzero(np.isin(date_values, list(train_dates)))
        validation_index = np.flatnonzero(np.isin(date_values, list(validation_dates)))
        if len(train_index) and len(validation_index):
            folds.append((train_index, validation_index))
    return folds


def choose_regularization(
    task: TaskSpec,
    frame: pd.DataFrame,
    variant: str,
    groups: dict[str, list[str]],
    grid: list[float],
    folds: int,
    max_tfidf_features: int,
    embedding_components: int,
) -> tuple[float, list[dict[str, object]]]:
    target = frame[task.target].to_numpy(
        dtype=int if task.kind == "classification" else float
    )
    rows: list[dict[str, object]] = []
    date_folds = expanding_date_folds(frame, folds)
    if not date_folds:
        raise ValueError(f"Not enough dates for expanding validation in {task.name}")
    for regularization in grid:
        for fold_index, (train_index, validation_index) in enumerate(date_folds, start=1):
            train_target = target[train_index]
            validation_target = target[validation_index]
            if task.kind == "classification" and (
                len(np.unique(train_target)) < 2
                or len(np.unique(validation_target)) < 2
            ):
                continue
            model = make_model(
                task,
                variant,
                groups,
                regularization,
                max_tfidf_features,
                embedding_components,
            )
            model.fit(frame.iloc[train_index], train_target)
            predictions = model_predictions(task, model, frame.iloc[validation_index])
            rows.append(
                {
                    "task": task.name,
                    "kind": task.kind,
                    "variant": variant,
                    "regularization": regularization,
                    "fold": fold_index,
                    "training_rows": len(train_index),
                    "validation_rows": len(validation_index),
                    "training_start": frame.iloc[train_index].outcome_session_date.min(),
                    "training_end": frame.iloc[train_index].outcome_session_date.max(),
                    "validation_start": frame.iloc[
                        validation_index
                    ].outcome_session_date.min(),
                    "validation_end": frame.iloc[
                        validation_index
                    ].outcome_session_date.max(),
                    **score_predictions(task, validation_target, predictions),
                }
            )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError(f"No valid chronological folds for {task.name}/{variant}")
    numeric_metrics = [
        column
        for column in metrics.columns
        if column
        not in {
            "task",
            "kind",
            "variant",
            "fold",
            "training_start",
            "training_end",
            "validation_start",
            "validation_end",
        }
    ]
    summary = metrics.groupby("regularization", as_index=False)[numeric_metrics].mean()
    if task.kind == "classification":
        ranked = summary.sort_values(
            ["roc_auc", "average_precision", "log_loss", "regularization"],
            ascending=[False, False, True, True],
        )
    else:
        ranked = summary.sort_values(
            ["rmse", "spearman", "regularization"],
            ascending=[True, False, True],
        )
    return float(ranked.iloc[0].regularization), rows


def bootstrap_primary_metric(
    task: TaskSpec,
    actual: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    dates: np.ndarray,
    iterations: int,
    comparator_name: str,
) -> dict[str, object]:
    unique_dates = np.unique(dates)
    rng = np.random.default_rng(0)
    model_values: list[float] = []
    differences: list[float] = []
    primary = "roc_auc" if task.kind == "classification" else "spearman"
    for _ in range(iterations):
        sampled = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        indices = np.concatenate([np.flatnonzero(dates == date) for date in sampled])
        sampled_actual = actual[indices]
        if task.kind == "classification" and len(np.unique(sampled_actual)) < 2:
            continue
        model_metric = score_predictions(task, sampled_actual, predictions[indices])[primary]
        baseline_metric = score_predictions(task, sampled_actual, baseline[indices])[primary]
        if np.isfinite(model_metric):
            model_values.append(float(model_metric))
        if np.isfinite(model_metric) and np.isfinite(baseline_metric):
            differences.append(float(model_metric - baseline_metric))

    def interval(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"low": np.nan, "high": np.nan, "valid_resamples": 0}
        return {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
            "valid_resamples": len(values),
        }

    return {
        "primary_metric": primary,
        "comparator": comparator_name,
        "model_95_ci": interval(model_values),
        "model_minus_comparator_95_ci": interval(differences),
    }


def selective_metrics(
    task: TaskSpec,
    actual: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    count = max(1, int(np.ceil(len(actual) * 0.10)))
    if task.kind == "classification":
        confidence = np.abs(predictions - 0.5)
        confident = np.argsort(confidence)[-count:]
        result: dict[str, float | int] = {
            "most_confident_10pct_rows": count,
            "most_confident_10pct_accuracy": float(
                accuracy_score(actual[confident], (predictions[confident] >= 0.5).astype(int))
            ),
        }
        if task.name.startswith("impact_"):
            highest = np.argsort(predictions)[-count:]
            precision = float(np.mean(actual[highest]))
            base_rate = float(np.mean(actual))
            result["highest_probability_10pct_precision"] = precision
            result["highest_probability_10pct_lift"] = (
                precision / base_rate if base_rate > 0 else np.nan
            )
        return result
    highest = np.argsort(predictions)[-count:]
    lowest = np.argsort(predictions)[:count]
    return {
        "tail_rows_each": count,
        "actual_mean_top_prediction_decile": float(np.mean(actual[highest])),
        "actual_mean_bottom_prediction_decile": float(np.mean(actual[lowest])),
        "top_minus_bottom_actual_return": float(
            np.mean(actual[highest]) - np.mean(actual[lowest])
        ),
    }


def atomic_joblib(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    joblib.dump(model, temporary)
    os.replace(temporary, path)


def top_feature_rows(
    task: TaskSpec,
    variant: str,
    model: Pipeline,
    count: int = 30,
) -> list[dict[str, object]]:
    try:
        names = model.named_steps["features"].get_feature_names_out()
        coefficients = np.asarray(model.named_steps["model"].coef_).reshape(-1)
    except (AttributeError, ValueError):
        return []
    if len(names) != len(coefficients):
        return []
    rows: list[dict[str, object]] = []
    for direction, order in (
        ("positive", np.argsort(coefficients)[::-1]),
        ("negative", np.argsort(coefficients)),
    ):
        for rank, index in enumerate(order[:count], start=1):
            rows.append(
                {
                    "task": task.name,
                    "variant": variant,
                    "direction": direction,
                    "rank": rank,
                    "feature": str(names[index]),
                    "coefficient": float(coefficients[index]),
                }
            )
    return rows


def matched_control_rows(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Nearest same-date/time/state no-news comparisons for regular events."""
    if controls.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    matching_features = ["minutes_from_open", "pre_volatility_30m", "prior_return_15m"]
    regular = events.loc[events.market_period.eq("regular")].copy()
    for horizon in horizons:
        outcome = f"reaction_return_{horizon}m"
        event_subset = regular.dropna(subset=[outcome, *matching_features])
        control_subset = controls.dropna(subset=[outcome, *matching_features])
        if event_subset.empty or control_subset.empty:
            continue
        scales = control_subset[matching_features].std().replace(0, 1).fillna(1)
        for event in event_subset.itertuples(index=False):
            same_date = control_subset.loc[
                control_subset.outcome_session_date.astype(str)
                == str(event.outcome_session_date)
            ]
            candidates = same_date if len(same_date) else control_subset
            event_values = np.array(
                [float(getattr(event, feature)) for feature in matching_features]
            )
            candidate_values = candidates[matching_features].to_numpy(dtype=float)
            distances = np.sqrt(
                np.sum(((candidate_values - event_values) / scales.to_numpy()) ** 2, axis=1)
            )
            match = candidates.iloc[int(np.argmin(distances))]
            event_return = float(getattr(event, outcome))
            control_return = float(match[outcome])
            rows.append(
                {
                    "horizon_minutes": horizon,
                    "event_id": event.event_id,
                    "control_id": match.event_id,
                    "outcome_session_date": event.outcome_session_date,
                    "event_return": event_return,
                    "control_return": control_return,
                    "signed_difference": event_return - control_return,
                    "absolute_return_difference": abs(event_return) - abs(control_return),
                    "matching_distance": float(np.min(distances)),
                }
            )
    return pd.DataFrame(rows)


def train_all(
    events_path: Path,
    controls_path: Path | None,
    transformer_path: Path | None,
    output_dir: Path,
    selected_tasks: set[str] | None,
    c_grid: list[float],
    alpha_grid: list[float],
    folds: int,
    test_fraction: float,
    max_tfidf_features: int,
    embedding_components: int,
    bootstrap_iterations: int,
    market_periods: set[str] | None = None,
) -> dict[str, object]:
    source_hash_before = file_sha256(events_path)
    events = pd.read_parquet(events_path)
    events["outcome_session_date"] = events.outcome_session_date.astype(str)
    if transformer_path is not None:
        transformer = pd.read_parquet(transformer_path)
        if transformer.event_id.astype(str).duplicated().any():
            raise ValueError("Transformer feature file contains duplicate event IDs")
        transformer["event_id"] = transformer.event_id.astype(str)
        events["event_id"] = events.event_id.astype(str)
        events = events.merge(
            transformer,
            on="event_id",
            how="left",
            validate="one_to_one",
        )
    if market_periods is not None:
        events = events.loc[events.market_period.astype(str).isin(market_periods)].copy()
        if events.empty:
            raise ValueError(
                "No events remain for market periods: " + ", ".join(sorted(market_periods))
            )

    tasks = [
        task
        for task in DEFAULT_TASKS
        if task.target in events.columns
        and (selected_tasks is None or task.name in selected_tasks)
    ]
    if not tasks:
        raise ValueError("No requested task targets are present in the event dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    task_payload: dict[str, object] = {}

    for task in tasks:
        frame = events.loc[
            events.market_data_available.fillna(False)
            & pd.to_numeric(events[task.target], errors="coerce").notna()
        ].copy()
        frame[task.target] = pd.to_numeric(frame[task.target], errors="raise")
        if task.kind == "classification":
            frame[task.target] = frame[task.target].astype(int)
            if set(frame[task.target].unique()).difference({0, 1}):
                raise ValueError(f"Task {task.name} is not binary")
        frame = frame.sort_values(["analysis_time_utc", "event_id"]).reset_index(drop=True)
        development, test = development_test_split(frame, test_fraction)
        if task.kind == "classification" and (
            development[task.target].nunique() < 2 or test[task.target].nunique() < 2
        ):
            print(f"skip {task.name}: one class in development or test", flush=True)
            continue
        groups = feature_groups(frame, task)
        variants = available_variants(groups)
        target_dtype = int if task.kind == "classification" else float
        development_target = development[task.target].to_numpy(dtype=target_dtype)
        test_target = test[task.target].to_numpy(dtype=target_dtype)
        constant = baseline_predictions(task, development_target, len(test))
        constant_metrics = score_predictions(task, test_target, constant)
        task_results: dict[str, object] = {
            "kind": task.kind,
            "target": task.target,
            "sample": {
                "eligible_rows": len(frame),
                "development_rows": len(development),
                "test_rows": len(test),
                "development_date_range": [
                    str(development.outcome_session_date.min()),
                    str(development.outcome_session_date.max()),
                ],
                "test_date_range": [
                    str(test.outcome_session_date.min()),
                    str(test.outcome_session_date.max()),
                ],
                "development_target_mean": float(np.mean(development_target)),
                "test_target_mean": float(np.mean(test_target)),
            },
            "constant_baseline": constant_metrics,
            "models": {},
        }
        market_predictions: np.ndarray | None = None

        for variant in variants:
            grid = c_grid if task.kind == "classification" else alpha_grid
            try:
                selected, rows = choose_regularization(
                    task,
                    development,
                    variant,
                    groups,
                    grid,
                    folds,
                    max_tfidf_features,
                    embedding_components,
                )
            except ValueError as exc:
                print(f"skip {task.name}/{variant}: {exc}", flush=True)
                continue
            fold_rows.extend(rows)
            model = make_model(
                task,
                variant,
                groups,
                selected,
                max_tfidf_features,
                embedding_components,
            )
            model.fit(development, development_target)
            predictions = model_predictions(task, model, test)
            metrics = score_predictions(task, test_target, predictions)
            comparator = (
                constant
                if variant == "market_only" or market_predictions is None
                else market_predictions
            )
            comparator_name = (
                "constant_baseline"
                if variant == "market_only" or market_predictions is None
                else "market_only"
            )
            bootstrap = bootstrap_primary_metric(
                task,
                test_target,
                predictions,
                comparator,
                test.outcome_session_date.astype(str).to_numpy(),
                bootstrap_iterations,
                comparator_name,
            )
            selective = selective_metrics(task, test_target, predictions)
            if variant == "market_only":
                market_predictions = predictions.copy()
            model_path = output_dir / "models" / f"{task.name}__{variant}.joblib"
            atomic_joblib(model, model_path)
            feature_rows.extend(top_feature_rows(task, variant, model))
            task_results["models"][variant] = {
                "selected_regularization": selected,
                "metrics": metrics,
                "day_block_bootstrap": bootstrap,
                "selective_metrics": selective,
                "model_path": str(model_path.resolve()),
            }
            comparison_rows.append(
                {
                    "task": task.name,
                    "kind": task.kind,
                    "variant": variant,
                    "selected_regularization": selected,
                    "test_rows": len(test),
                    **metrics,
                    **selective,
                }
            )
            for row_index, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "event_id": row.event_id,
                        "outcome_session_date": row.outcome_session_date,
                        "analysis_time_utc": row.analysis_time_utc,
                        "market_period": row.market_period,
                        "event_category": row.event_category,
                        "task": task.name,
                        "kind": task.kind,
                        "variant": variant,
                        "actual": float(test_target[row_index]),
                        "prediction": float(predictions[row_index]),
                        "constant_baseline_prediction": float(constant[row_index]),
                    }
                )
            print(
                f"trained {task.name}/{variant} regularization={selected}",
                flush=True,
            )
        task_payload[task.name] = task_results

    fold_frame = pd.DataFrame(fold_rows)
    comparison = pd.DataFrame(comparison_rows)
    predictions = pd.DataFrame(prediction_rows)
    top_features = pd.DataFrame(feature_rows)
    atomic_csv(fold_frame, output_dir / "walk_forward_metrics.csv")
    atomic_csv(comparison, output_dir / "model_comparison.csv")
    atomic_csv(predictions, output_dir / "test_predictions.csv")
    atomic_csv(top_features, output_dir / "top_features.csv")

    control_matches = pd.DataFrame()
    if controls_path is not None and controls_path.is_file():
        controls = pd.read_parquet(controls_path)
        control_matches = matched_control_rows(events, controls, [5, 15, 60])
        atomic_csv(control_matches, output_dir / "matched_control_comparison.csv")

    source_hash_after = file_sha256(events_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Event dataset changed during model training")
    payload: dict[str, object] = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Compare NVDA market-state, structured-news, TF-IDF, and frozen-transformer "
            "features on immediate impact and post-decision continuation targets."
        ),
        "input": {
            "events": {
                "path": str(events_path.resolve()),
                "sha256": source_hash_before,
            },
            "controls": str(controls_path.resolve()) if controls_path else None,
            "transformer_features": (
                str(transformer_path.resolve()) if transformer_path else None
            ),
            "market_period_filter": (
                sorted(market_periods) if market_periods is not None else None
            ),
        },
        "evaluation": {
            "test_fraction_of_dates": test_fraction,
            "development_validation": f"{folds}-fold expanding-window by outcome date",
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_unit": "outcome trading date",
            "selection": (
                "classification: mean ROC AUC, AP and log loss tie-breakers; "
                "regression: RMSE, Spearman tie-breaker"
            ),
        },
        "feature_policy": {
            "reaction_tasks": "pre-event features only",
            "continuation_tasks": "pre-event features plus initial return/volume available at decision",
            "tfidf_fit": "inside every chronological fold and final development fit",
            "embedding_pca_fit": "inside every chronological fold and final development fit",
        },
        "tasks": task_payload,
        "matched_control_pairs": len(control_matches),
        "package_versions": {
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
        "limitations": [
            "This is a single-security study with a short historical sample.",
            "Probability calibration is evaluated but remains uncertain in a short chronological test.",
            "Publication metadata may lag the first market-visible appearance of an article.",
            "Immediate-reaction targets measure association and are not executable pre-news trades.",
        ],
    }
    atomic_json(payload, output_dir / "model_metrics.json")
    return payload


def parse_float_grid(value: str) -> list[float]:
    try:
        result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError(f"Invalid numeric grid: {value!r}") from exc
    if not result or result[0] <= 0:
        raise ValueError("Regularization grids must contain positive values")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", type=Path, default=DEFAULT_OUTPUT / "news_events.parquet"
    )
    parser.add_argument(
        "--controls", type=Path, default=DEFAULT_OUTPUT / "no_news_controls.parquet"
    )
    parser.add_argument("--transformer-features", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "models_output")
    parser.add_argument(
        "--tasks",
        help="Comma-separated task names; defaults to every target present in the dataset",
    )
    parser.add_argument("--c-grid", default="0.01,0.1,1,10")
    parser.add_argument("--alpha-grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-tfidf-features", type=int, default=5000)
    parser.add_argument("--embedding-components", type=int, default=32)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument(
        "--market-periods",
        help="Optional comma-separated subset: regular,premarket,afterhours,overnight",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if not args.events.is_file():
        parser.error(f"Event dataset does not exist: {args.events}")
    if args.transformer_features is not None and not args.transformer_features.is_file():
        parser.error(f"Transformer feature file does not exist: {args.transformer_features}")
    if args.folds < 1 or not 0 < args.test_fraction < 0.5:
        parser.error("Folds must be positive and test fraction must be between 0 and 0.5")
    if args.max_tfidf_features < 10 or args.embedding_components < 1:
        parser.error("Feature dimensions are invalid")
    if args.bootstrap_iterations < 1:
        parser.error("Bootstrap iterations must be positive")
    try:
        c_grid = parse_float_grid(args.c_grid)
        alpha_grid = parse_float_grid(args.alpha_grid)
    except ValueError as exc:
        parser.error(str(exc))
    selected_tasks = (
        {item.strip() for item in args.tasks.split(",") if item.strip()}
        if args.tasks
        else None
    )
    market_periods = (
        {item.strip() for item in args.market_periods.split(",") if item.strip()}
        if args.market_periods
        else None
    )
    valid_periods = {"regular", "premarket", "afterhours", "overnight"}
    if market_periods is not None and market_periods.difference(valid_periods):
        parser.error(
            "Unknown market periods: "
            + ", ".join(sorted(market_periods.difference(valid_periods)))
        )
    payload = train_all(
        events_path=args.events,
        controls_path=args.controls if args.controls.is_file() else None,
        transformer_path=args.transformer_features,
        output_dir=args.output_dir,
        selected_tasks=selected_tasks,
        c_grid=c_grid,
        alpha_grid=alpha_grid,
        folds=args.folds,
        test_fraction=args.test_fraction,
        max_tfidf_features=args.max_tfidf_features,
        embedding_components=args.embedding_components,
        bootstrap_iterations=args.bootstrap_iterations,
        market_periods=market_periods,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "tasks": list(payload["tasks"]),
                "output": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

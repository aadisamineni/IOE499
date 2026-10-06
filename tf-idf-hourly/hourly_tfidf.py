"""Evaluate whether newly available news text predicts NVDA hourly returns.

Run from the repository root:
    .venv/bin/python tf-idf-hourly/hourly_tfidf.py

One observation is one complete regular-session hourly bar. At each bar start,
the document contains retained representative news published since the preceding
bar start and strictly before the current bar. The outcomes are the current
bar's open-to-close direction and a three-class move using a fixed return band.
All chronological splits are made by trading date, and TF-IDF is fitted only on
the corresponding training portion.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
HOURLY = ROOT / "data/raw/NVDA_hourly_2025-10-01_2026-05-31.csv"
HOURLY_METADATA = ROOT / "data/raw/NVDA_hourly_2025-10-01_2026-05-31_metadata.json"
NEWS = ROOT / "news data pull/processed/all_classified_articles.csv"
COVERAGE = ROOT / "data/matched/2025-09-15_2026-05-15/collection_coverage_by_day.csv"
OUTPUT = Path(__file__).resolve().parent / "output"

EXPLICIT_ZONE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
TOKEN_PATTERN = r"(?u)\b(?=\w*[a-zA-Z])\w\w+\b"
ELIGIBLE_RELEVANCE = {"direct", "indirect"}
MODEL_NAMES = ("market_volume", "tfidf_only", "market_volume_tfidf")
MARKET_FEATURES = [
    "prior_bar_return",
    "prior_volatility",
    "log1p_prior_volume",
    "log1p_article_count",
    "minutes_from_open",
    "minutes_from_open_squared",
]
TARGETS = {
    "direction": {
        "column": "direction_label",
        "classes": np.array([-1, 1], dtype=int),
        "names": {-1: "down", 1: "up"},
    },
    "move_band": {
        "column": "move_band_label",
        "classes": np.array([-1, 0, 1], dtype=int),
        "names": {-1: "down", 0: "neutral", 1: "up"},
    },
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_explicit_timestamp(value: object) -> pd.Timestamp:
    """Parse only timestamps that include a time and an explicit UTC offset."""
    text = str(value).strip()
    if not re.search(r"[T ]\d{2}:\d{2}", text) or not EXPLICIT_ZONE.search(text):
        return pd.NaT
    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if parsed.tzinfo is None:
        return pd.NaT
    return parsed.tz_convert("UTC")


def article_text(row: pd.Series) -> str:
    headline = str(row.get("headline", "")).strip()
    summary = str(row.get("summary", "")).strip()
    if headline and summary:
        return f"{headline}. {summary}"
    return headline or summary


def complete_coverage_days(coverage: pd.DataFrame) -> tuple[set[str], list[str]]:
    required = {"coverage_date_utc", "ticker", "coverage_state"}
    missing = required.difference(coverage.columns)
    if missing:
        raise ValueError(f"Coverage input is missing columns: {sorted(missing)}")
    if coverage.duplicated(["coverage_date_utc", "ticker"]).any():
        raise ValueError("Coverage input contains duplicate date/ticker rows")
    tickers = sorted(coverage.ticker.unique())
    good_days = set()
    for date, group in coverage.groupby("coverage_date_utc", sort=False):
        if sorted(group.ticker.unique()) == tickers and group.coverage_state.eq("success").all():
            good_days.add(str(date))
    if not good_days:
        raise ValueError("Coverage input contains no completely successful UTC dates")
    return good_days, tickers


def prepare_articles(news: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "article_id",
        "headline",
        "summary",
        "published_at_utc",
        "relevance_type",
        "excluded",
        "event_representative",
        "duplicate_group_id",
    }
    missing = required.difference(news.columns)
    if missing:
        raise ValueError(f"News input is missing columns: {sorted(missing)}")

    selected = news.loc[
        news.relevance_type.isin(ELIGIBLE_RELEVANCE)
        & news.excluded.str.lower().eq("false")
        & news.event_representative.str.lower().eq("true")
    ].drop_duplicates().copy()
    selected["published"] = selected.published_at_utc.map(parse_explicit_timestamp)
    rejected = int(selected.published.isna().sum())
    selected = selected.loc[selected.published.notna()].copy()
    if selected.empty:
        raise ValueError("No retained representative direct/indirect articles remain")
    if selected.duplicate_group_id.eq("").any():
        raise ValueError("A retained article is missing its duplicate-group identifier")
    if selected.duplicated("duplicate_group_id").any():
        raise ValueError("More than one retained representative exists in a duplicate group")
    selected["article_text"] = selected.apply(article_text, axis=1)
    selected = selected.sort_values(["published", "duplicate_group_id"]).reset_index(drop=True)
    audit = {
        "retained_representative_articles": int(len(selected)),
        "rejected_publication_timestamps": rejected,
    }
    return selected, audit


def interval_has_complete_coverage(
        start: pd.Timestamp, end: pd.Timestamp, good_days: set[str]) -> bool:
    if pd.isna(start) or pd.isna(end) or start >= end:
        return False
    last_included = end - pd.Timedelta(nanoseconds=1)
    days = pd.date_range(start.floor("D"), last_included.floor("D"), freq="D", tz="UTC")
    return all(str(day.date()) in good_days for day in days)


def prepare_bars(bars: pd.DataFrame, volatility_window: int) -> pd.DataFrame:
    required = {
        "trading_timestamp", "ticker", "open", "high", "low", "close",
        "adjusted_close", "volume",
    }
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Hourly input is missing columns: {sorted(missing)}")
    result = bars.copy()
    result["bar_start_utc"] = result.trading_timestamp.map(parse_explicit_timestamp)
    if result.bar_start_utc.isna().any():
        raise ValueError("Every hourly bar needs an explicit, valid timezone")
    result = result.sort_values("bar_start_utc").reset_index(drop=True)
    if result.bar_start_utc.duplicated().any():
        raise ValueError("Hourly timestamps must be unique")
    if not result.ticker.eq("NVDA").all():
        raise ValueError("Every hourly row must be for NVDA")

    result["bar_start_new_york"] = result.bar_start_utc.dt.tz_convert("America/New_York")
    result["trading_date"] = result.bar_start_new_york.dt.strftime("%Y-%m-%d")
    first_clock = result.groupby("trading_date").bar_start_new_york.first().dt.strftime("%H:%M")
    if not first_clock.eq("09:30").all():
        raise ValueError("Every session must start at 09:30 America/New_York")

    next_start = result.groupby("trading_date").bar_start_utc.shift(-1)
    observed_duration = (next_start - result.bar_start_utc).dt.total_seconds() / 60
    is_session_final = next_start.isna()
    if not observed_duration.loc[~is_session_final].eq(60).all():
        raise ValueError("Non-final bars must be spaced exactly 60 minutes apart")
    final_clock = result.loc[is_session_final, "bar_start_new_york"].dt.strftime("%H:%M")
    if not final_clock.isin(["12:30", "15:30"]).all():
        raise ValueError("A session has an unexpected final bar timestamp")
    result["bar_duration_minutes"] = np.where(is_session_final, 30, 60)
    result["is_partial_final_bar"] = is_session_final

    numeric = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    positive = result[["open", "close"]].gt(0).all(axis=1)
    result["bar_return"] = np.where(positive, result.close / result.open - 1, np.nan)
    result["previous_prediction_cutoff_utc"] = result.bar_start_utc.shift(1)
    result["prior_bar_return"] = result.bar_return.shift(1)
    result["prior_volatility"] = (
        result.bar_return.shift(1).rolling(volatility_window, min_periods=volatility_window)
        .std(ddof=0)
    )
    prior_volume = result.volume.shift(1)
    result["log1p_prior_volume"] = np.where(prior_volume.ge(0), np.log1p(prior_volume), np.nan)
    session_open = result.bar_start_new_york.dt.normalize() + pd.Timedelta(hours=9, minutes=30)
    result["minutes_from_open"] = (
        result.bar_start_new_york - session_open).dt.total_seconds() / 60
    result["minutes_from_open_squared"] = result.minutes_from_open ** 2
    return result


def build_hourly_examples(
        bars: pd.DataFrame,
        news: pd.DataFrame,
        good_days: set[str],
        move_threshold: float = 0.005,
        volatility_window: int = 6,
) -> tuple[pd.DataFrame, dict]:
    """Construct leakage-safe, non-overlapping news documents for full hourly bars."""
    if not 0 < move_threshold < 1:
        raise ValueError("move_threshold must be between zero and one")
    if volatility_window < 2:
        raise ValueError("volatility_window must be at least two bars")
    hourly = prepare_bars(bars, volatility_window)
    articles, article_audit = prepare_articles(news)

    full = ~hourly.is_partial_final_bar
    target_price_ok = np.isfinite(hourly.bar_return)
    cutoff_ok = hourly.previous_prediction_cutoff_utc.notna()
    controls_ok = np.isfinite(hourly[
        ["prior_bar_return", "prior_volatility", "log1p_prior_volume",
         "minutes_from_open", "minutes_from_open_squared"]
    ].to_numpy(dtype=float)).all(axis=1)
    base_eligible = full & target_price_ok & cutoff_ok & controls_ok
    coverage_ok = pd.Series(False, index=hourly.index)
    for index in hourly.index[base_eligible]:
        coverage_ok.loc[index] = interval_has_complete_coverage(
            hourly.at[index, "previous_prediction_cutoff_utc"],
            hourly.at[index, "bar_start_utc"],
            good_days,
        )
    eligible = hourly.loc[base_eligible & coverage_ok].copy()
    zero_return_bars = int(eligible.bar_return.eq(0).sum())
    eligible = eligible.loc[eligible.bar_return.ne(0)].copy()
    if eligible.empty:
        raise ValueError("No hourly examples remain after price, control, and coverage checks")

    # Timestamp.value is always nanoseconds. Pandas may otherwise expose a
    # lower-resolution integer dtype, which would make search cutoffs incomparable.
    published_ns = np.array(
        [timestamp.value for timestamp in articles.published], dtype=np.int64)
    texts = articles.article_text.to_numpy(dtype=object)
    documents = []
    article_counts = []
    for row in eligible.itertuples(index=False):
        lower = int(row.previous_prediction_cutoff_utc.value)
        upper = int(row.bar_start_utc.value)
        first = int(np.searchsorted(published_ns, lower, side="left"))
        last = int(np.searchsorted(published_ns, upper, side="left"))
        window_text = [str(value) for value in texts[first:last] if str(value).strip()]
        documents.append("\n".join(window_text))
        article_counts.append(last - first)

    eligible["document"] = documents
    eligible["representative_article_count"] = article_counts
    eligible["log1p_article_count"] = np.log1p(eligible.representative_article_count)
    eligible["direction_label"] = np.where(eligible.bar_return > 0, 1, -1).astype(int)
    eligible["move_band_label"] = np.select(
        [eligible.bar_return <= -move_threshold, eligible.bar_return >= move_threshold],
        [-1, 1],
        default=0,
    ).astype(int)
    if not np.isfinite(eligible[MARKET_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("An eligible market or news-volume feature is not finite")
    if not eligible.bar_start_utc.is_monotonic_increasing:
        raise ValueError("Eligible examples are not chronological")

    audit = {
        "input_bars": int(len(hourly)),
        "partial_final_bars_excluded": int((~full).sum()),
        "full_bars_with_missing_target_price": int((full & ~target_price_ok).sum()),
        "otherwise_usable_bars_without_prior_controls": int(
            (full & target_price_ok & (~cutoff_ok | ~controls_ok)).sum()),
        "bars_excluded_for_incomplete_news_coverage": int((base_eligible & ~coverage_ok).sum()),
        "zero_return_bars_excluded": zero_return_bars,
        "eligible_examples": int(len(eligible)),
        "examples_with_news": int(eligible.representative_article_count.gt(0).sum()),
        "articles_assigned_to_eligible_examples": int(
            eligible.representative_article_count.sum()),
        **article_audit,
    }
    columns = [
        "trading_date", "bar_start_utc", "bar_start_new_york",
        "previous_prediction_cutoff_utc", "bar_duration_minutes", "bar_return",
        "direction_label", "move_band_label", "prior_bar_return", "prior_volatility",
        "log1p_prior_volume", "minutes_from_open", "minutes_from_open_squared",
        "representative_article_count", "log1p_article_count", "document",
    ]
    return eligible[columns].reset_index(drop=True), audit


def make_vectorizer(min_df: int, max_df: float, max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float64,
    )


def build_model(name: str, c_value: float, min_df: int, max_df: float,
                max_features: int) -> Pipeline:
    transformers = []
    if name in {"tfidf_only", "market_volume_tfidf"}:
        transformers.append(("text", make_vectorizer(min_df, max_df, max_features), "document"))
    if name in {"market_volume", "market_volume_tfidf"}:
        transformers.append(("market", StandardScaler(), MARKET_FEATURES))
    if not transformers:
        raise ValueError(f"Unknown model name: {name}")
    features = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def aligned_probabilities(model: Pipeline, frame: pd.DataFrame,
                          classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(frame)
    model_classes = model.named_steps["classifier"].classes_.astype(int)
    aligned = np.zeros((len(frame), len(classes)), dtype=float)
    for source, label in enumerate(model_classes):
        matches = np.where(classes == label)[0]
        if not len(matches):
            raise ValueError(f"Model produced unexpected class {label}")
        aligned[:, int(matches[0])] = raw[:, source]
    return aligned


def class_prior_probabilities(y_train: np.ndarray, n_rows: int,
                              classes: np.ndarray) -> np.ndarray:
    counts = np.array([(y_train == label).sum() for label in classes], dtype=float)
    if not counts.all():
        raise ValueError("Training data must contain every target class")
    return np.tile(counts / counts.sum(), (n_rows, 1))


def score_probabilities(y_true: np.ndarray, probabilities: np.ndarray,
                        classes: np.ndarray, class_names: dict[int, str]) -> dict:
    predictions = classes[np.argmax(probabilities, axis=1)]
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(recall_score(
            y_true, predictions, labels=classes, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(
            y_true, predictions, labels=classes, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
    }
    present = np.unique(y_true)
    if len(classes) == 2 and len(present) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
    elif len(classes) > 2 and len(present) == len(classes):
        result["macro_ovr_auc"] = float(roc_auc_score(
            y_true, probabilities, labels=classes, multi_class="ovr", average="macro"))
    else:
        result["roc_auc" if len(classes) == 2 else "macro_ovr_auc"] = None

    precision = precision_score(
        y_true, predictions, labels=classes, average=None, zero_division=0)
    recall = recall_score(y_true, predictions, labels=classes, average=None, zero_division=0)
    f1 = f1_score(y_true, predictions, labels=classes, average=None, zero_division=0)
    result["per_class"] = {
        class_names[int(label)]: {
            "support": int((y_true == label).sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(classes)
    }
    return result


def split_by_trading_date(frame: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(frame.trading_date.unique()))
    split = int(len(dates) * train_fraction)
    if split < 2 or split >= len(dates):
        raise ValueError("Date split leaves too few training or test dates")
    train_dates = set(dates[:split])
    train = frame.loc[frame.trading_date.isin(train_dates)].copy()
    test = frame.loc[~frame.trading_date.isin(train_dates)].copy()
    if set(train.trading_date) & set(test.trading_date):
        raise AssertionError("A trading date appears in both train and test")
    return train, test


def chronological_date_folds(frame: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = np.array(sorted(frame.trading_date.unique()))
    if n_splits < 2 or len(dates) <= n_splits + 1:
        raise ValueError("Too few dates for the requested chronological folds")
    test_size = max(2, len(dates) // (n_splits + 2))
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    folds = []
    for train_date_index, validation_date_index in splitter.split(dates):
        train_dates = set(dates[train_date_index])
        validation_dates = set(dates[validation_date_index])
        train_rows = np.flatnonzero(frame.trading_date.isin(train_dates).to_numpy())
        validation_rows = np.flatnonzero(frame.trading_date.isin(validation_dates).to_numpy())
        folds.append((train_rows, validation_rows))
    return folds


def tune_models(
        train: pd.DataFrame,
        target_name: str,
        c_grid: list[float],
        n_splits: int,
        min_df: int,
        max_df: float,
        max_features: int,
) -> tuple[dict, pd.DataFrame]:
    target = TARGETS[target_name]
    classes = target["classes"]
    class_names = target["names"]
    labels = train[target["column"]].to_numpy(dtype=int)
    rows = []
    for fold_number, (train_index, validation_index) in enumerate(
            chronological_date_folds(train, n_splits), start=1):
        y_fit = labels[train_index]
        y_validation = labels[validation_index]
        if len(np.unique(y_fit)) != len(classes):
            raise ValueError(
                f"{target_name} training fold {fold_number} does not contain every class")
        context = {
            "target": target_name,
            "fold": fold_number,
            "train_start": train.trading_date.iloc[train_index[0]],
            "train_end": train.trading_date.iloc[train_index[-1]],
            "validation_start": train.trading_date.iloc[validation_index[0]],
            "validation_end": train.trading_date.iloc[validation_index[-1]],
        }
        prior = class_prior_probabilities(y_fit, len(validation_index), classes)
        prior_scores = score_probabilities(y_validation, prior, classes, class_names)
        rows.append({
            "model": "class_prior", "c": "", **context,
            **{key: value for key, value in prior_scores.items() if key != "per_class"},
        })
        for name in MODEL_NAMES:
            for c_value in c_grid:
                model = build_model(name, c_value, min_df, max_df, max_features)
                model.fit(train.iloc[train_index], y_fit)
                probabilities = aligned_probabilities(
                    model, train.iloc[validation_index], classes)
                scores = score_probabilities(
                    y_validation, probabilities, classes, class_names)
                rows.append({
                    "model": name, "c": c_value, **context,
                    **{key: value for key, value in scores.items() if key != "per_class"},
                })

    fold_metrics = pd.DataFrame(rows)
    selected = {}
    for name in MODEL_NAMES:
        candidates = fold_metrics.loc[fold_metrics.model.eq(name)].copy()
        candidates["c_numeric"] = pd.to_numeric(candidates.c)
        aggregate = candidates.groupby("c_numeric").agg(
            mean_macro_f1=("macro_f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_log_loss=("log_loss", "mean"),
        ).reset_index()
        best = aggregate.sort_values(
            ["mean_macro_f1", "mean_balanced_accuracy", "mean_log_loss", "c_numeric"],
            ascending=[False, False, True, True],
        ).iloc[0]
        selected[name] = {
            "c": float(best.c_numeric),
            "mean_cv_macro_f1": float(best.mean_macro_f1),
            "mean_cv_balanced_accuracy": float(best.mean_balanced_accuracy),
            "mean_cv_log_loss": float(best.mean_log_loss),
        }
    return selected, fold_metrics


def bootstrap_intervals(
        y_true: np.ndarray,
        probabilities: np.ndarray,
        trading_dates: np.ndarray,
        classes: np.ndarray,
        class_names: dict[int, str],
        samples: int,
        seed: int,
) -> dict:
    if samples < 1:
        return {}
    unique_dates = np.array(sorted(np.unique(trading_dates)))
    positions = {date: np.flatnonzero(trading_dates == date) for date in unique_dates}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(samples):
        chosen = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        index = np.concatenate([positions[date] for date in chosen])
        scores = score_probabilities(
            y_true[index], probabilities[index], classes, class_names)
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss",
                    "roc_auc", "macro_ovr_auc"):
            value = scores.get(key)
            if value is not None:
                values.setdefault(key, []).append(float(value))
    return {
        key: {
            "low": float(np.quantile(metric_values, 0.025)),
            "high": float(np.quantile(metric_values, 0.975)),
        }
        for key, metric_values in values.items()
        if metric_values
    }


def paired_bootstrap_difference(
        y_true: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        trading_dates: np.ndarray,
        classes: np.ndarray,
        class_names: dict[int, str],
        samples: int,
        seed: int,
) -> dict:
    unique_dates = np.array(sorted(np.unique(trading_dates)))
    positions = {date: np.flatnonzero(trading_dates == date) for date in unique_dates}
    rng = np.random.default_rng(seed)
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]
    metric_names.append("roc_auc" if len(classes) == 2 else "macro_ovr_auc")
    original_first = score_probabilities(y_true, first, classes, class_names)
    original_second = score_probabilities(y_true, second, classes, class_names)
    values = {metric: [] for metric in metric_names}
    for _ in range(samples):
        chosen = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        index = np.concatenate([positions[date] for date in chosen])
        score_first = score_probabilities(
            y_true[index], first[index], classes, class_names)
        score_second = score_probabilities(
            y_true[index], second[index], classes, class_names)
        for metric in metric_names:
            if score_first.get(metric) is not None and score_second.get(metric) is not None:
                values[metric].append(score_first[metric] - score_second[metric])
    return {
        metric: {
            "estimate": float(original_first[metric] - original_second[metric]),
            "low": float(np.quantile(metric_values, 0.025)),
            "high": float(np.quantile(metric_values, 0.975)),
        }
        for metric, metric_values in values.items()
        if metric_values and original_first.get(metric) is not None
    }


def top_term_rows(target_name: str, model_name: str, model: Pipeline,
                  classes: np.ndarray, class_names: dict[int, str],
                  count: int = 20) -> list[dict]:
    features = model.named_steps["features"]
    vectorizer = features.named_transformers_["text"]
    terms = vectorizer.get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[:, :len(terms)]
    if len(classes) == 2:
        class_coefficients = {
            int(classes[0]): -coefficients[0],
            int(classes[1]): coefficients[0],
        }
    else:
        fitted_classes = model.named_steps["classifier"].classes_.astype(int)
        class_coefficients = {
            int(label): coefficients[index]
            for index, label in enumerate(fitted_classes)
        }
    rows = []
    for label in classes:
        values = class_coefficients[int(label)]
        for rank, index in enumerate(np.argsort(values)[::-1][:count], start=1):
            rows.append({
                "target": target_name,
                "model": model_name,
                "class_label": int(label),
                "class_name": class_names[int(label)],
                "rank": rank,
                "term": terms[index],
                "coefficient_toward_class": float(values[index]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.90)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--move-threshold", type=float, default=0.005)
    parser.add_argument("--volatility-window", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction <= 0.9:
        parser.error("--train-fraction must be between 0.5 and 0.9")
    if args.cv_splits < 2:
        parser.error("--cv-splits must be at least two")
    if any(value <= 0 for value in args.c_grid):
        parser.error("Every --c-grid value must be positive")
    if args.min_df < 1 or not 0 < args.max_df <= 1 or args.max_features < 1:
        parser.error("Invalid TF-IDF document-frequency or feature setting")
    if not 0 < args.move_threshold < 1:
        parser.error("--move-threshold must be between zero and one")
    if args.volatility_window < 2 or args.bootstrap_samples < 1:
        parser.error("Invalid volatility window or bootstrap sample count")

    source_paths = [HOURLY, HOURLY_METADATA, NEWS, COVERAGE]
    hashes_before = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    hourly_metadata = json.loads(HOURLY_METADATA.read_text(encoding="utf-8"))
    if hourly_metadata.get("interval") != "1h":
        raise ValueError("Hourly metadata does not describe one-hour bars")
    if hourly_metadata.get("bar_timestamp_convention") != (
            "Provider timestamp marks the beginning of each hourly bar."):
        raise ValueError("Unexpected hourly timestamp convention")

    good_days, coverage_tickers = complete_coverage_days(read_csv(COVERAGE))
    examples, build_audit = build_hourly_examples(
        read_csv(HOURLY),
        read_csv(NEWS),
        good_days,
        move_threshold=args.move_threshold,
        volatility_window=args.volatility_window,
    )
    train, test = split_by_trading_date(examples, args.train_fraction)
    test_dates = test.trading_date.to_numpy()

    all_selected = {}
    fold_frames = []
    final_models: dict[str, dict[str, Pipeline]] = {}
    all_probabilities: dict[str, dict[str, np.ndarray]] = {}
    all_metrics = {}
    paired_differences = {}

    for target_number, (target_name, target) in enumerate(TARGETS.items(), start=1):
        classes = target["classes"]
        class_names = target["names"]
        y_train = train[target["column"]].to_numpy(dtype=int)
        y_test = test[target["column"]].to_numpy(dtype=int)
        if len(np.unique(y_train)) != len(classes) or len(np.unique(y_test)) != len(classes):
            raise ValueError(f"Both {target_name} partitions must contain every class")

        selected, fold_metrics = tune_models(
            train, target_name, args.c_grid, args.cv_splits,
            args.min_df, args.max_df, args.max_features)
        all_selected[target_name] = selected
        fold_frames.append(fold_metrics)
        probabilities = {
            "class_prior": class_prior_probabilities(y_train, len(test), classes)
        }
        target_models = {}
        for name in MODEL_NAMES:
            model = build_model(
                name, selected[name]["c"], args.min_df, args.max_df, args.max_features)
            model.fit(train, y_train)
            target_models[name] = model
            probabilities[name] = aligned_probabilities(model, test, classes)
        final_models[target_name] = target_models
        all_probabilities[target_name] = probabilities

        target_metrics = {}
        for model_number, (name, values) in enumerate(probabilities.items(), start=1):
            scores = score_probabilities(y_test, values, classes, class_names)
            scores["day_block_bootstrap_95_ci"] = bootstrap_intervals(
                y_test, values, test_dates, classes, class_names,
                args.bootstrap_samples, seed=1000 * target_number + model_number)
            target_metrics[name] = scores
        all_metrics[target_name] = target_metrics
        paired_differences[target_name] = paired_bootstrap_difference(
            y_test,
            probabilities["market_volume_tfidf"],
            probabilities["market_volume"],
            test_dates,
            classes,
            class_names,
            args.bootstrap_samples,
            seed=9000 + target_number,
        )

    predictions = test[[
        "trading_date", "bar_start_utc", "bar_start_new_york", "bar_return",
        "direction_label", "move_band_label", "representative_article_count",
    ]].copy()
    confusion_rows = []
    term_rows = []
    for target_name, target in TARGETS.items():
        classes = target["classes"]
        class_names = target["names"]
        y_test = test[target["column"]].to_numpy(dtype=int)
        for name, values in all_probabilities[target_name].items():
            predicted = classes[np.argmax(values, axis=1)]
            predictions[f"{target_name}__{name}__prediction"] = predicted
            for index, label in enumerate(classes):
                predictions[
                    f"{target_name}__{name}__probability_{class_names[int(label)]}"
                ] = values[:, index]
            matrix = confusion_matrix(y_test, predicted, labels=classes)
            for actual_index, actual in enumerate(classes):
                for predicted_index, predicted_label in enumerate(classes):
                    confusion_rows.append({
                        "target": target_name,
                        "model": name,
                        "actual_label": int(actual),
                        "actual_name": class_names[int(actual)],
                        "predicted_label": int(predicted_label),
                        "predicted_name": class_names[int(predicted_label)],
                        "count": int(matrix[actual_index, predicted_index]),
                    })
        for name in ("tfidf_only", "market_volume_tfidf"):
            term_rows.extend(top_term_rows(
                target_name, name, final_models[target_name][name],
                classes, class_names))

    hashes_after = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    if hashes_before != hashes_after:
        raise AssertionError("A source file changed during modeling")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output / "hourly_documents.csv", index=False)
    predictions.to_csv(output / "held_out_predictions.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrix.csv", index=False)
    pd.DataFrame(term_rows).to_csv(output / "top_terms_by_class.csv", index=False)

    target_samples = {}
    final_feature_counts = {}
    for target_name, target in TARGETS.items():
        classes = target["classes"]
        column = target["column"]
        target_samples[target_name] = {
            "all_class_counts": {
                str(label): int((examples[column] == label).sum()) for label in classes},
            "train_class_counts": {
                str(label): int((train[column] == label).sum()) for label in classes},
            "test_class_counts": {
                str(label): int((test[column] == label).sum()) for label in classes},
        }
        final_feature_counts[target_name] = {
            name: int(model.named_steps["classifier"].n_features_in_)
            for name, model in final_models[target_name].items()
        }

    metadata = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "objective": "Predict NVDA hourly open-to-close movement using newly available news.",
        "unit_of_analysis": "one complete regular-session hourly bar",
        "prediction_timing": {
            "cutoff": "start of the target hourly bar",
            "news_window": (
                "previous provider bar start inclusive to target bar start exclusive; "
                "overnight news is assigned to the next market-open bar"
            ),
            "target": "target bar close / target bar open - 1",
            "partial_bar_policy": "exclude each session's final 30-minute provider bar",
        },
        "targets": {
            "direction": {"encoding": {"-1": "return < 0", "1": "return > 0"}},
            "move_band": {
                "threshold": args.move_threshold,
                "encoding": {
                    "-1": f"return <= -{args.move_threshold}",
                    "0": f"-{args.move_threshold} < return < {args.move_threshold}",
                    "1": f"return >= {args.move_threshold}",
                },
            },
        },
        "news_filter": (
            "Non-excluded direct/indirect event representatives with valid explicit publication times"
        ),
        "coverage_policy": (
            "Every UTC date touched by a news window must have successful collection for all tickers"
        ),
        "coverage_tickers": coverage_tickers,
        "source_sha256": hashes_before,
        "source_files_unchanged": True,
        "package_versions": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
        "build_audit": build_audit,
        "sample": {
            "eligible_examples": int(len(examples)),
            "eligible_trading_dates": int(examples.trading_date.nunique()),
            "date_range": [examples.trading_date.iloc[0], examples.trading_date.iloc[-1]],
            "train_examples": int(len(train)),
            "train_trading_dates": int(train.trading_date.nunique()),
            "train_date_range": [train.trading_date.iloc[0], train.trading_date.iloc[-1]],
            "test_examples": int(len(test)),
            "test_trading_dates": int(test.trading_date.nunique()),
            "test_date_range": [test.trading_date.iloc[0], test.trading_date.iloc[-1]],
            "targets": target_samples,
        },
        "tfidf": {
            "text": "headline and summary concatenated within each non-overlapping news window",
            "lowercase": True,
            "strip_accents": "unicode",
            "stop_words": "english",
            "token_pattern": TOKEN_PATTERN,
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_features": args.max_features,
            "sublinear_tf": True,
            "norm": "l2",
            "fit_policy": "fit independently inside each training fold and final training data",
        },
        "market_and_volume_features": MARKET_FEATURES,
        "classifier": "class-balanced L2 logistic regression",
        "chronological_evaluation": {
            "split_unit": "trading date",
            "train_fraction_of_dates": args.train_fraction,
            "cv_splits": args.cv_splits,
            "c_grid": args.c_grid,
            "selection_metric": "mean macro F1; balanced accuracy and log loss break ties",
            "bootstrap": (
                f"{args.bootstrap_samples} held-out resamples of complete trading-day blocks"
            ),
        },
        "selected_hyperparameters": all_selected,
        "final_feature_counts": final_feature_counts,
        "held_out_metrics": all_metrics,
        "paired_difference_market_volume_tfidf_minus_market_volume": paired_differences,
        "limitations": [
            "Hourly bars from the same market regime remain statistically dependent even with date-blocked evaluation.",
            "The final provider bar is only 30 minutes and is excluded rather than modeled as a full hour.",
            "The price source contains documented provider gaps; affected targets and lagged-control rows are excluded.",
            "Publication times are retrospective metadata and do not prove when an article first became tradable information.",
            "Class balancing and macro-F1 model selection prioritize class decisions rather than calibrated probabilities.",
            "TF-IDF coefficients are associations, not linguistic sentiment, causality, or trading recommendations.",
            "Top terms are selected on the same training sample and are exploratory.",
        ],
    }
    (output / "model_metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_directory": str(output.relative_to(ROOT)),
        "sample": metadata["sample"],
        "selected_hyperparameters": all_selected,
        "held_out_metrics": all_metrics,
        "paired_difference_market_volume_tfidf_minus_market_volume": paired_differences,
    }, indent=2))


if __name__ == "__main__":
    main()

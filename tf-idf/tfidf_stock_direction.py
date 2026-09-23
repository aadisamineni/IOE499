"""Evaluate whether previous-day news text predicts three-class NVDA movement.

Run from the repository root:
    .venv/bin/python tf-idf/tfidf_stock_direction.py

The unit of analysis is one stock session. Eligible representative article
headlines and summaries from the preceding New York calendar date are combined
into one document. TF-IDF is fitted inside each chronological training fold so
validation and test text never influence its vocabulary or inverse-document
frequencies.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

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
DAILY = ROOT / "matched_one_day/output/daily_stock_news_summary.csv"
MATCHES = ROOT / "matched_one_day/output/stock_news_matches.csv"
MATCH_VALIDATION = ROOT / "matched_one_day/output/validation_report.json"
OUTPUT = Path(__file__).resolve().parent / "output"
CLASSES = np.array([-1, 0, 1], dtype=int)
CLASS_NAMES = {-1: "down", 0: "neutral", 1: "up"}
MARKET_FEATURES = ["prior_return", "prior_volatility_20d", "log1p_article_count"]
MODEL_NAMES = ("market_only", "tfidf_only", "market_plus_tfidf")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def article_text(row: pd.Series) -> str:
    headline = str(row.get("headline", "")).strip()
    summary = str(row.get("summary", "")).strip()
    if headline and summary:
        return f"{headline}. {summary}"
    return headline or summary


def build_session_documents(daily: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Create one leakage-safe modeling document per fully covered stock day."""
    required_daily = {
        "trading_date", "matched_news_date", "price_move_2pct",
        "adjusted_close_return_1d", "adjusted_close_return_1d_lag1_session",
        "daily_return_volatility_20d_lag1_session",
        "within_configured_utc_collection_range",
    }
    required_matches = {
        "trading_date", "matched_news_date", "article_match_key", "headline", "summary",
        "event_representative", "publication_timestamp_utc", "duplicate_group_id",
    }
    missing_daily = required_daily.difference(daily.columns)
    missing_matches = required_matches.difference(matches.columns)
    if missing_daily:
        raise ValueError(f"Daily input is missing columns: {sorted(missing_daily)}")
    if missing_matches:
        raise ValueError(f"Article input is missing columns: {sorted(missing_matches)}")
    if daily.trading_date.duplicated().any() or not daily.trading_date.is_monotonic_increasing:
        raise ValueError("Daily input must contain unique, chronologically sorted sessions")

    sessions = daily.loc[
        pd.to_numeric(daily.within_configured_utc_collection_range, errors="coerce").eq(1)
    ].copy()
    sessions = sessions.sort_values("trading_date").reset_index(drop=True)
    labels = pd.to_numeric(sessions.price_move_2pct, errors="coerce")
    if labels.isna().any() or not labels.isin(CLASSES).all():
        raise ValueError("price_move_2pct must contain only -1, 0, and 1")
    sessions["price_move_2pct"] = labels.astype(int)

    representative = matches.loc[
        matches.event_representative.str.lower().eq("true") &
        matches.trading_date.isin(sessions.trading_date)
    ].copy()
    if representative.duplicated(["trading_date", "article_match_key"]).any():
        raise ValueError("Duplicate stock-session/article pairs remain after filtering")
    if representative.duplicated(["trading_date", "duplicate_group_id"]).any():
        raise ValueError("More than one representative remains for a duplicate group")
    representative = representative.sort_values(
        ["trading_date", "publication_timestamp_utc", "article_match_key"])
    representative["article_text"] = representative.apply(article_text, axis=1)

    documents = representative.groupby("trading_date", sort=False).agg(
        document=("article_text", lambda values: "\n".join(v for v in values if v)),
        representative_article_count=("article_match_key", "size"),
    ).reset_index()
    sessions = sessions.merge(documents, on="trading_date", how="left", validate="one_to_one")
    sessions["document"] = sessions.document.fillna("")
    sessions["representative_article_count"] = (
        pd.to_numeric(sessions.representative_article_count, errors="coerce").fillna(0).astype(int))
    sessions["prior_return"] = pd.to_numeric(
        sessions.adjusted_close_return_1d_lag1_session, errors="coerce")
    sessions["prior_volatility_20d"] = pd.to_numeric(
        sessions.daily_return_volatility_20d_lag1_session, errors="coerce")
    sessions["log1p_article_count"] = np.log1p(sessions.representative_article_count)
    if not np.isfinite(sessions[MARKET_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("A market control is missing or non-finite")
    if not (sessions.matched_news_date == (
            pd.to_datetime(sessions.trading_date) - pd.Timedelta(days=1)
            ).dt.strftime("%Y-%m-%d")).all():
        raise ValueError("A session does not use the previous calendar date")
    return sessions


def make_vectorizer(min_df: int, max_df: float, max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
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
    if name in {"tfidf_only", "market_plus_tfidf"}:
        transformers.append(("text", make_vectorizer(min_df, max_df, max_features), "document"))
    if name in {"market_only", "market_plus_tfidf"}:
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


def aligned_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    model_classes = model.named_steps["classifier"].classes_.astype(int)
    aligned = np.zeros((len(frame), len(CLASSES)), dtype=float)
    for source, label in enumerate(model_classes):
        aligned[:, int(np.where(CLASSES == label)[0][0])] = raw[:, source]
    return aligned


def score_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = CLASSES[np.argmax(probabilities, axis=1)]
    scores = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(recall_score(
            y_true, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, labels=CLASSES,
                                    average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=CLASSES)),
    }
    if len(np.unique(y_true)) == len(CLASSES):
        scores["macro_ovr_auc"] = float(roc_auc_score(
            y_true, probabilities, labels=CLASSES, multi_class="ovr", average="macro"))
    else:
        scores["macro_ovr_auc"] = None
    precision = precision_score(y_true, predictions, labels=CLASSES,
                                average=None, zero_division=0)
    recall = recall_score(y_true, predictions, labels=CLASSES,
                          average=None, zero_division=0)
    f1 = f1_score(y_true, predictions, labels=CLASSES, average=None, zero_division=0)
    scores["per_class"] = {
        CLASS_NAMES[int(label)]: {
            "support": int((y_true == label).sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(CLASSES)
    }
    return scores


def class_prior_probabilities(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    counts = np.array([(y_train == label).sum() for label in CLASSES], dtype=float)
    probabilities = counts / counts.sum()
    return np.tile(probabilities, (n_rows, 1))


def chronological_folds(n_rows: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("At least two chronological CV splits are required")
    # Reserve roughly one third for the initial fit and split the remainder
    # into broader validation windows. Narrower windows can contain only the
    # neutral class in this small, imbalanced time series.
    test_size = max(5, n_rows // (n_splits + 2))
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    folds = list(splitter.split(np.arange(n_rows)))
    for train_index, validation_index in folds:
        # The caller checks actual label coverage because this function only
        # receives row positions.
        if not len(train_index) or not len(validation_index):
            raise ValueError("Chronological CV produced an empty fold")
    return folds


def tune_models(train: pd.DataFrame, c_grid: list[float], n_splits: int,
                min_df: int, max_df: float, max_features: int) -> tuple[dict, pd.DataFrame]:
    folds = chronological_folds(len(train), n_splits)
    y = train.price_move_2pct.to_numpy(dtype=int)
    fold_rows = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        y_fold_train, y_validation = y[train_index], y[validation_index]
        if len(np.unique(y_fold_train)) != len(CLASSES):
            raise ValueError(f"Training fold {fold_number} does not contain all three classes")
        fold_context = {
            "fold": fold_number,
            "train_start": train.trading_date.iloc[train_index[0]],
            "train_end": train.trading_date.iloc[train_index[-1]],
            "validation_start": train.trading_date.iloc[validation_index[0]],
            "validation_end": train.trading_date.iloc[validation_index[-1]],
            **{f"train_{CLASS_NAMES[int(label)]}": int((y_fold_train == label).sum())
               for label in CLASSES},
            **{f"validation_{CLASS_NAMES[int(label)]}": int((y_validation == label).sum())
               for label in CLASSES},
        }
        baseline = class_prior_probabilities(y_fold_train, len(validation_index))
        baseline_scores = score_probabilities(y_validation, baseline)
        fold_rows.append({
            "model": "class_prior", "c": "", **fold_context,
            **{key: value for key, value in baseline_scores.items() if key != "per_class"},
        })
        for name in MODEL_NAMES:
            for c_value in c_grid:
                model = build_model(name, c_value, min_df, max_df, max_features)
                model.fit(train.iloc[train_index], y_fold_train)
                probabilities = aligned_probabilities(model, train.iloc[validation_index])
                scores = score_probabilities(y_validation, probabilities)
                fold_rows.append({
                    "model": name, "c": c_value, **fold_context,
                    **{key: value for key, value in scores.items() if key != "per_class"},
                })

    fold_metrics = pd.DataFrame(fold_rows)
    selected = {}
    for name in MODEL_NAMES:
        candidates = fold_metrics.loc[fold_metrics.model.eq(name)].copy()
        candidates["c_numeric"] = pd.to_numeric(candidates.c)
        aggregate = candidates.groupby("c_numeric").agg(
            mean_macro_f1=("macro_f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_log_loss=("log_loss", "mean"),
        ).reset_index()
        aggregate = aggregate.sort_values(
            ["mean_macro_f1", "mean_balanced_accuracy", "mean_log_loss", "c_numeric"],
            ascending=[False, False, True, True],
        )
        best = aggregate.iloc[0]
        selected[name] = {
            "c": float(best.c_numeric),
            "mean_cv_macro_f1": float(best.mean_macro_f1),
            "mean_cv_balanced_accuracy": float(best.mean_balanced_accuracy),
            "mean_cv_log_loss": float(best.mean_log_loss),
        }
    return selected, fold_metrics


def top_term_rows(name: str, model: Pipeline, count: int = 20) -> list[dict]:
    features = model.named_steps["features"]
    vectorizer = features.named_transformers_["text"]
    terms = vectorizer.get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[:, :len(terms)]
    rows = []
    for class_index, label in enumerate(model.named_steps["classifier"].classes_.astype(int)):
        values = coefficients[class_index]
        for direction, indices in (
            ("positive", np.argsort(values)[::-1][:count]),
            ("negative", np.argsort(values)[:count]),
        ):
            for rank, index in enumerate(indices, start=1):
                rows.append({
                    "model": name,
                    "class_label": int(label),
                    "class_name": CLASS_NAMES[int(label)],
                    "association": direction,
                    "rank": rank,
                    "term": terms[index],
                    "coefficient": float(values[index]),
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
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction <= 0.9:
        parser.error("--train-fraction must be between 0.5 and 0.9")
    if args.cv_splits < 2:
        parser.error("--cv-splits must be at least 2")
    if any(value <= 0 for value in args.c_grid):
        parser.error("Every --c-grid value must be positive")
    if args.min_df < 1 or not 0 < args.max_df <= 1 or args.max_features < 1:
        parser.error("Invalid TF-IDF document-frequency or feature setting")

    source_paths = [DAILY, MATCHES, MATCH_VALIDATION]
    hashes_before = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    sessions = build_session_documents(read_csv(DAILY), read_csv(MATCHES))
    split = int(len(sessions) * args.train_fraction)
    train, test = sessions.iloc[:split].copy(), sessions.iloc[split:].copy()
    y_train = train.price_move_2pct.to_numpy(dtype=int)
    y_test = test.price_move_2pct.to_numpy(dtype=int)
    if len(np.unique(y_train)) != len(CLASSES) or len(np.unique(y_test)) != len(CLASSES):
        raise ValueError("Both final chronological partitions must contain all three classes")

    selected, fold_metrics = tune_models(
        train, args.c_grid, args.cv_splits, args.min_df, args.max_df, args.max_features)

    final_models = {}
    probabilities = {"class_prior": class_prior_probabilities(y_train, len(test))}
    for name in MODEL_NAMES:
        model = build_model(name, selected[name]["c"], args.min_df, args.max_df,
                            args.max_features)
        model.fit(train, y_train)
        final_models[name] = model
        probabilities[name] = aligned_probabilities(model, test)

    metrics = {name: score_probabilities(y_test, values)
               for name, values in probabilities.items()}
    final_feature_counts = {
        name: int(model.named_steps["classifier"].n_features_in_)
        for name, model in final_models.items()
    }
    predictions = test[[
        "trading_date", "matched_news_date", "adjusted_close_return_1d",
        "price_move_2pct", "representative_article_count",
    ]].copy()
    confusion_rows = []
    for name, values in probabilities.items():
        predicted = CLASSES[np.argmax(values, axis=1)]
        predictions[f"{name}_prediction"] = predicted
        for index, label in enumerate(CLASSES):
            predictions[f"{name}_probability_{CLASS_NAMES[int(label)]}"] = values[:, index]
        matrix = confusion_matrix(y_test, predicted, labels=CLASSES)
        for actual_index, actual in enumerate(CLASSES):
            for predicted_index, predicted_label in enumerate(CLASSES):
                confusion_rows.append({
                    "model": name,
                    "actual_label": int(actual),
                    "actual_name": CLASS_NAMES[int(actual)],
                    "predicted_label": int(predicted_label),
                    "predicted_name": CLASS_NAMES[int(predicted_label)],
                    "count": int(matrix[actual_index, predicted_index]),
                })

    term_rows = []
    for name in ("tfidf_only", "market_plus_tfidf"):
        term_rows.extend(top_term_rows(name, final_models[name]))

    hashes_after = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    assert hashes_before == hashes_after, "Source files changed during modeling"
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    document_columns = [
        "trading_date", "matched_news_date", "adjusted_close_return_1d",
        "price_move_2pct", "prior_return", "prior_volatility_20d",
        "representative_article_count", "log1p_article_count", "document",
    ]
    sessions[document_columns].to_csv(output / "session_documents.csv", index=False)
    predictions.to_csv(output / "session_predictions.csv", index=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrix.csv", index=False)
    pd.DataFrame(term_rows).to_csv(output / "top_terms_by_class.csv", index=False)

    metadata = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "objective": "Predict -1/0/+1 NVDA adjusted-close movement using previous-day news.",
        "target": {
            "column": "price_move_2pct",
            "encoding": {"-1": "return <= -2%", "0": "-2% < return < 2%",
                         "1": "return >= 2%"},
        },
        "unit_of_analysis": "one NVDA trading session",
        "news_rule": "Representative direct/indirect articles from the previous New York calendar date.",
        "source_sha256": hashes_before,
        "source_files_unchanged": True,
        "package_versions": {
            "numpy": version("numpy"), "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
        "sample": {
            "eligible_sessions": len(sessions),
            "date_range": [sessions.trading_date.iloc[0], sessions.trading_date.iloc[-1]],
            "class_counts": {str(label): int((sessions.price_move_2pct == label).sum())
                             for label in CLASSES},
            "train_sessions": len(train),
            "train_date_range": [train.trading_date.iloc[0], train.trading_date.iloc[-1]],
            "train_class_counts": {str(label): int((y_train == label).sum()) for label in CLASSES},
            "test_sessions": len(test),
            "test_date_range": [test.trading_date.iloc[0], test.trading_date.iloc[-1]],
            "test_class_counts": {str(label): int((y_test == label).sum()) for label in CLASSES},
        },
        "tfidf": {
            "text": "headline and summary concatenated per session",
            "lowercase": True,
            "strip_accents": "unicode",
            "stop_words": "english",
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_features": args.max_features,
            "sublinear_tf": True,
            "norm": "l2",
            "fit_policy": "fit independently inside each training fold and once on final training data",
        },
        "market_features": MARKET_FEATURES,
        "classifier": "class-balanced L2 multinomial logistic regression",
        "chronological_evaluation": {
            "train_fraction": args.train_fraction,
            "cv_splits": args.cv_splits,
            "c_grid": args.c_grid,
            "selection_metric": "mean macro F1; balanced accuracy and log loss break ties",
        },
        "selected_hyperparameters": selected,
        "final_feature_counts": final_feature_counts,
        "held_out_metrics": metrics,
        "limitations": [
            "Only 146 fully covered sessions are available; estimates have high sampling uncertainty.",
            "The neutral class is substantially larger than the up and down classes.",
            "TF-IDF measures term importance, not linguistic sentiment or causal impact.",
            "Previous-calendar-day matching excludes same-day premarket news and, for Monday sessions, Friday and Saturday news.",
            "Retrospectively collected article metadata cannot guarantee historical text availability or completeness.",
            "Top coefficients are exploratory associations selected on the same training sample.",
        ],
    }
    (output / "model_metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_directory": str(output.relative_to(ROOT)),
        "sample": metadata["sample"],
        "selected_hyperparameters": selected,
        "held_out_metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

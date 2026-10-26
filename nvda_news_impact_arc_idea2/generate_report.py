#!/usr/bin/env python3
"""Generate a compact Markdown report from completed idea-2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT


def format_number(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "—"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows available."
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def generate_report(output_dir: Path, destination: Path) -> str:
    dataset_audit = json.loads((output_dir / "dataset_audit.json").read_text())
    study = pd.read_csv(output_dir / "event_study_summary.csv")
    comparison_path = output_dir / "models_output/model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.is_file() else pd.DataFrame()
    metrics_path = output_dir / "models_output/model_metrics.json"
    model_metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    matches_path = output_dir / "models_output/matched_control_comparison.csv"
    matches = pd.read_csv(matches_path) if matches_path.is_file() else pd.DataFrame()

    overall = study.loc[study.grouping.eq("overall")].copy()
    overall_table = pd.DataFrame(
        {
            "horizon": overall.horizon_minutes.astype(int).astype(str) + "m",
            "events": overall.available_returns.astype(int),
            "mean return": overall.mean_return.map(lambda value: format_number(value, 6)),
            "mean absolute return": overall.mean_absolute_return.map(
                lambda value: format_number(value, 6)
            ),
            "positive rate": overall.positive_rate.map(format_number),
            "impact rate": overall.impact_rate.map(format_number),
        }
    )

    period = study.loc[
        study.grouping.eq("market_period") & study.horizon_minutes.eq(15)
    ].copy()
    period_table = pd.DataFrame(
        {
            "period": period.market_period,
            "events": period.available_returns.astype(int),
            "mean absolute 15m return": period.mean_absolute_return.map(
                lambda value: format_number(value, 6)
            ),
            "15m impact rate": period.impact_rate.map(format_number),
        }
    )

    best_rows: list[dict[str, object]] = []
    if not comparison.empty:
        for task, group in comparison.groupby("task", sort=False):
            kind = str(group.kind.iloc[0])
            if kind == "classification":
                eligible = group.dropna(subset=["roc_auc"])
                best = eligible.sort_values("roc_auc", ascending=False).iloc[0]
                market = group.loc[group.variant.eq("market_only")]
                market_score = float(market.roc_auc.iloc[0]) if len(market) else np.nan
                interval = (
                    model_metrics.get("tasks", {})
                    .get(task, {})
                    .get("models", {})
                    .get(str(best.variant), {})
                    .get("day_block_bootstrap", {})
                    .get("model_minus_comparator_95_ci", {})
                )
                difference_interval = (
                    "—"
                    if str(best.variant) == "market_only"
                    else (
                        f"[{format_number(interval.get('low'))}, "
                        f"{format_number(interval.get('high'))}]"
                    )
                )
                best_rows.append(
                    {
                        "task": task,
                        "best model": best.variant,
                        "metric": "ROC AUC",
                        "best score": format_number(best.roc_auc),
                        "market only": format_number(market_score),
                        "difference": format_number(float(best.roc_auc) - market_score),
                        "paired difference 95% CI": difference_interval,
                    }
                )
            else:
                eligible = group.dropna(subset=["rmse"])
                best = eligible.sort_values("rmse").iloc[0]
                market = group.loc[group.variant.eq("market_only")]
                market_score = float(market.rmse.iloc[0]) if len(market) else np.nan
                interval = (
                    model_metrics.get("tasks", {})
                    .get(task, {})
                    .get("models", {})
                    .get(str(best.variant), {})
                    .get("day_block_bootstrap", {})
                    .get("model_minus_comparator_95_ci", {})
                )
                best_rows.append(
                    {
                        "task": task,
                        "best model": best.variant,
                        "metric": "RMSE",
                        "best score": format_number(best.rmse, 6),
                        "market only": format_number(market_score, 6),
                        "difference": format_number(float(best.rmse) - market_score, 6),
                        "paired difference 95% CI": (
                            "—"
                            if str(best.variant) == "market_only"
                            else (
                                "Spearman delta "
                                f"[{format_number(interval.get('low'))}, "
                                f"{format_number(interval.get('high'))}]"
                            )
                        ),
                    }
                )
    best_table = pd.DataFrame(best_rows)

    segment_rows: list[dict[str, object]] = []
    for segment_path in sorted(output_dir.glob("models_output_*/model_comparison.csv")):
        segment = segment_path.parent.name.removeprefix("models_output_")
        segment_comparison = pd.read_csv(segment_path)
        for task, group in segment_comparison.groupby("task", sort=False):
            market = group.loc[group.variant.eq("market_only")]
            if market.empty:
                continue
            if str(group.kind.iloc[0]) == "classification":
                best = group.dropna(subset=["roc_auc"]).sort_values(
                    "roc_auc", ascending=False
                ).iloc[0]
                market_score = float(market.roc_auc.iloc[0])
                metric = "ROC AUC"
                score = float(best.roc_auc)
            else:
                best = group.dropna(subset=["rmse"]).sort_values("rmse").iloc[0]
                market_score = float(market.rmse.iloc[0])
                metric = "RMSE"
                score = float(best.rmse)
            segment_rows.append(
                {
                    "period": segment,
                    "task": task,
                    "best model": best.variant,
                    "metric": metric,
                    "best score": format_number(score, 6 if metric == "RMSE" else 4),
                    "market only": format_number(
                        market_score, 6 if metric == "RMSE" else 4
                    ),
                    "difference": format_number(
                        score - market_score, 6 if metric == "RMSE" else 4
                    ),
                }
            )
    segment_table = pd.DataFrame(segment_rows)

    match_rows: list[dict[str, object]] = []
    if not matches.empty:
        rng = np.random.default_rng(0)
        for horizon, group in matches.groupby("horizon_minutes"):
            daily = group.groupby(group.outcome_session_date.astype(str)).agg(
                difference_sum=("absolute_return_difference", "sum"),
                rows=("absolute_return_difference", "size"),
            )
            sampled_indices = rng.integers(
                0, len(daily), size=(1_000, len(daily))
            )
            sums = daily.difference_sum.to_numpy(dtype=float)[sampled_indices].sum(axis=1)
            counts = daily.rows.to_numpy(dtype=float)[sampled_indices].sum(axis=1)
            bootstrapped = sums / counts
            match_rows.append(
                {
                    "horizon": f"{int(horizon)}m",
                    "pairs": len(group),
                    "mean news-minus-control absolute return": format_number(
                        group.absolute_return_difference.mean(), 6
                    ),
                    "median difference": format_number(
                        group.absolute_return_difference.median(), 6
                    ),
                    "day-block 95% CI": (
                        f"[{format_number(np.quantile(bootstrapped, 0.025), 6)}, "
                        f"{format_number(np.quantile(bootstrapped, 0.975), 6)}]"
                    ),
                }
            )
    match_table = pd.DataFrame(match_rows)

    event_info = dataset_audit["events"]
    news_info = dataset_audit["news_audit"]
    lines = [
        "# NVDA news impact — ARC idea 2 report",
        "",
        "## Coverage",
        "",
        f"- Retained source articles: {news_info['retained_articles']:,}",
        f"- Modeled information events: {event_info['rows']:,}",
        f"- Events with an available market anchor: {event_info['market_data_available']:,}",
        f"- Outcome sessions: {event_info['date_range'][0]} through {event_info['date_range'][1]}",
        f"- Event periods: `{json.dumps(event_info['market_period_counts'], sort_keys=True)}`",
        "",
        "## Immediate event study",
        "",
        markdown_table(overall_table),
        "",
        "### Fifteen-minute response by market period",
        "",
        markdown_table(period_table),
        "",
        "## Held-out model comparison",
        "",
        markdown_table(best_table),
        "",
        "For classification, a useful news result must improve chronologically on the "
        "NVDA market-only ROC AUC, not merely exceed the constant classifier's accuracy. "
        "For return regression, lower RMSE is better and rank correlation should also be "
        "checked in `model_comparison.csv`.",
        "The displayed best model is a descriptive comparison across the held-out results, "
        "not another model-selection step. Intervals spanning zero do not establish an "
        "incremental improvement over the market-only comparator.",
        "",
        "## Separately fitted market periods",
        "",
        markdown_table(segment_table),
        "",
        "## Matched no-news controls",
        "",
        markdown_table(match_table),
        "",
        "Controls may be reused across event matches; the uncertainty interval resamples "
        "complete outcome dates.",
        "",
        "## Interpretation safeguards",
        "",
        "- Immediate-reaction returns are attribution outcomes, not executable pre-news trades.",
        "- Continuation tasks begin only after their stated one- or five-minute decision delay.",
        "- News published while the market is closed is batched into the next opening information set.",
        "- All held-out results use the last block of complete outcome dates.",
        "- Publication timestamps may lag the first appearance of information elsewhere.",
        "- FinBERT/embedding results appear only after the frozen feature stage has been run.",
        "",
    ]
    report = "\n".join(lines)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or args.output_dir / "research_report.md"
    required = [
        args.output_dir / "dataset_audit.json",
        args.output_dir / "event_study_summary.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required artifacts: " + ", ".join(map(str, missing)))
    generate_report(args.output_dir, destination)
    print(json.dumps({"status": "complete", "report": str(destination.resolve())}, indent=2))


if __name__ == "__main__":
    main()

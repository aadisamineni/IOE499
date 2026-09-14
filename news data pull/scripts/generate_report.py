#!/usr/bin/env python3
"""Generate coverage, audit, duplicate, exclusion, and sample reports."""

from __future__ import annotations

import argparse
import calendar
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from common import (
    PROJECT_ROOT,
    atomic_write_json,
    load_config,
    normalize_text,
    read_csv,
    read_jsonl,
    term_present,
    utc_now_iso,
    write_csv,
)


MANIFEST_PATH = PROJECT_ROOT / "raw" / "request_manifest.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "processed"

REQUEST_FIELDS = [
    "window_id",
    "ticker",
    "from_date",
    "to_date",
    "status",
    "http_status",
    "article_count",
    "retrieved_at_utc",
    "raw_file",
    "error_kind",
]


def latest_windows() -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(MANIFEST_PATH):
        if row.get("record_type") == "request_window" and row.get("window_id"):
            latest[str(row["window_id"])] = row
    return sorted(
        latest.values(), key=lambda row: (row.get("ticker", ""), row.get("from_date", ""), row.get("to_date", ""))
    )


def month_keys(start: date, end: date) -> list[str]:
    cursor = start.replace(day=1)
    keys = []
    while cursor <= end:
        keys.append(cursor.strftime("%Y-%m"))
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
    return keys


def monthly_counts(
    articles: list[dict[str, str]], start: date, end: date
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {
        month: {
            "month": month,
            "unique_articles": 0,
            "classified_direct": 0,
            "classified_indirect": 0,
            "excluded_articles": 0,
            "duplicate_event_articles_suppressed": 0,
            "retained_direct_signals": 0,
            "retained_indirect_signals": 0,
        }
        for month in month_keys(start, end)
    }
    for article in articles:
        published = article.get("published_at_utc", "")
        month = published[:7] if len(published) >= 7 else "unknown"
        if month not in rows:
            continue
        row = rows[month]
        row["unique_articles"] += 1
        relevance_type = article.get("relevance_type")
        if relevance_type == "direct":
            row["classified_direct"] += 1
        elif relevance_type == "indirect":
            row["classified_indirect"] += 1
        excluded = article.get("excluded") == "true"
        if excluded:
            row["excluded_articles"] += 1
        if "duplicate_event_coverage" in article.get("exclusion_reason", ""):
            row["duplicate_event_articles_suppressed"] += 1
        if not excluded and relevance_type == "direct":
            row["retained_direct_signals"] += 1
        if not excluded and relevance_type == "indirect":
            row["retained_indirect_signals"] += 1
    return list(rows.values())


def exclusion_counts(articles: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for article in articles:
        for reason in article.get("exclusion_reason", "").split(";"):
            if reason:
                counts[reason] += 1
    return [
        {"exclusion_reason": reason, "article_count": count}
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def terminal_status(row: dict[str, Any]) -> bool:
    return row.get("status") not in {"split_potential_truncation", "transient_retry"}


def coverage_periods(
    windows: list[dict[str, Any]], tickers: list[str], start: date, end: date
) -> tuple[list[dict[str, str]], bool]:
    day_state: dict[str, dict[date, str]] = {
        ticker: {
            start + timedelta(days=offset): "missing_no_terminal_request"
            for offset in range((end - start).days + 1)
        }
        for ticker in tickers
    }
    for row in windows:
        ticker = str(row.get("ticker") or "")
        if ticker not in day_state or not terminal_status(row):
            continue
        try:
            row_start = max(start, date.fromisoformat(str(row["from_date"])))
            row_end = min(end, date.fromisoformat(str(row["to_date"])))
        except (KeyError, ValueError):
            continue
        status = str(row.get("status") or "unknown")
        if status == "success":
            state = "covered_success"
        elif status == "success_empty_unverified":
            state = "empty_response_unverified"
        elif status == "success_truncation_suspected":
            state = "potential_truncation"
        else:
            state = status
        cursor = row_start
        while cursor <= row_end:
            day_state[ticker][cursor] = state
            cursor += timedelta(days=1)

    periods: list[dict[str, str]] = []
    fully_verified = True
    for ticker in tickers:
        cursor = start
        while cursor <= end:
            state = day_state[ticker][cursor]
            if state == "covered_success":
                cursor += timedelta(days=1)
                continue
            fully_verified = False
            period_start = cursor
            while cursor + timedelta(days=1) <= end and day_state[ticker][cursor + timedelta(days=1)] == state:
                cursor += timedelta(days=1)
            periods.append(
                {
                    "ticker": ticker,
                    "from_date": period_start.isoformat(),
                    "to_date": cursor.isoformat(),
                    "coverage_state": state,
                    "interpretation": coverage_interpretation(state),
                }
            )
            cursor += timedelta(days=1)
    return periods, fully_verified


def coverage_interpretation(state: str) -> str:
    if state == "empty_response_unverified":
        return "HTTP 200 returned an empty array; this is not silently treated as proof of no news."
    if state == "potential_truncation":
        return "Minimum-size window still met the configured truncation threshold."
    if state == "missing_no_terminal_request":
        return "No completed terminal request covers this period; collection may be interrupted."
    if state.startswith("unavailable"):
        return "Finnhub did not provide an analyzable response for this period."
    return "The request failed after configured retries."


def choose_evenly(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (row.get("published_at_utc", ""), row.get("article_id", "")))
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(position * (len(ordered) - 1) / (count - 1)) for position in range(count)]
    return [ordered[index] for index in indexes]


def sample_quality(row: dict[str, str]) -> tuple[Any, ...]:
    headline = normalize_text(row.get("headline", ""))
    direct_in_headline = term_present(headline, "nvidia") or term_present(headline, "nvda")
    category_priority = {
        "export_controls_market_access": 6,
        "manufacturing_packaging_hbm": 5,
        "competing_accelerators_custom_chips": 5,
        "product_delay_supply_constraint": 5,
        "regulatory_action": 4,
        "partnership_contract": 4,
        "product_announcement": 4,
        "ai_infrastructure_spending": 3,
        "other_nvidia_development": 1,
    }.get(row.get("event_category", ""), 0)
    matched_count = len([term for term in row.get("matched_relevance_terms", "").split("|") if term])
    return (
        direct_in_headline if row.get("relevance_type") == "direct" else True,
        category_priority,
        matched_count,
        bool(row.get("summary")),
        len(row.get("summary", "")),
        row.get("published_at_utc", ""),
    )


def choose_stratified(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return sorted(rows, key=lambda row: (row.get("published_at_utc", ""), row.get("article_id", "")))
    selected: list[dict[str, str]] = []
    selected_ids: set[int] = set()
    category_order = [
        "product_announcement",
        "export_controls_market_access",
        "ai_infrastructure_spending",
        "manufacturing_packaging_hbm",
        "competing_accelerators_custom_chips",
        "partnership_contract",
        "product_delay_supply_constraint",
        "regulatory_action",
        "other_nvidia_development",
    ]
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row.get("event_category", "")].append(row)
    for category in category_order:
        if by_category[category]:
            best = max(by_category[category], key=sample_quality)
            selected.append(best)
            selected_ids.add(id(best))
            if len(selected) == count:
                break

    by_month: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if id(row) not in selected_ids:
            by_month[row.get("published_at_utc", "")[:7]].append(row)
    for month in sorted(by_month):
        if len(selected) == count:
            break
        best = max(by_month[month], key=sample_quality)
        selected.append(best)
        selected_ids.add(id(best))
    if len(selected) < count:
        remaining = sorted(
            (row for row in rows if id(row) not in selected_ids),
            key=sample_quality,
            reverse=True,
        )
        selected.extend(remaining[: count - len(selected)])
    return sorted(
        selected,
        key=lambda row: (row.get("published_at_utc", ""), row.get("article_id", "")),
    )


def write_sample_reports(signals: list[dict[str, str]]) -> dict[str, Any]:
    direct_pool = [row for row in signals if row.get("relevance_type") == "direct"]
    indirect_pool = [row for row in signals if row.get("relevance_type") == "indirect"]
    direct = choose_stratified(direct_pool, 10)
    indirect = choose_stratified(indirect_pool, 10)
    samples = direct + indirect
    sample_path = REPORTS_DIR / "sample_retained_articles.csv"
    fieldnames = list(signals[0].keys()) if signals else [
        "article_id",
        "queried_tickers",
        "headline",
        "summary",
        "source",
        "url",
        "published_at_utc",
        "retrieved_at_utc",
        "relevance_type",
        "event_category",
        "relevance_reason",
        "excluded",
        "exclusion_reason",
        "duplicate_group_id",
    ]
    write_csv(sample_path, samples, fieldnames)

    lines = [
        "# Sample retained articles",
        "",
        f"Generated at: {utc_now_iso()}",
        "",
        f"Selected {len(direct)} direct and {len(indirect)} indirect retained articles.",
        "",
    ]
    for group_name, group_rows in (("Direct Nvidia news", direct), ("Indirect Nvidia relevance", indirect)):
        lines.extend([f"## {group_name}", ""])
        if not group_rows:
            lines.extend(["No retained articles were available for this group.", ""])
        for index, row in enumerate(group_rows, start=1):
            headline = row.get("headline") or "(missing headline)"
            lines.extend(
                [
                    f"{index}. **{headline}**",
                    f"   - Published: {row.get('published_at_utc', '')}; source: {row.get('source', '')}",
                    f"   - Category: {row.get('event_category', '')}",
                    f"   - Relevance: {row.get('relevance_reason', '')}",
                    f"   - URL: {row.get('url', '')}",
                    "",
                ]
            )
    (REPORTS_DIR / "sample_retained_articles.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "requested_direct": 10,
        "actual_direct": len(direct),
        "requested_indirect": 10,
        "actual_indirect": len(indirect),
        "total": len(samples),
    }


def write_coverage_markdown(report: dict[str, Any]) -> None:
    summary = report["summary"]
    preflight = report["preflight"]
    lines = [
        "# Finnhub company-news coverage report",
        "",
        f"Generated at: {report['generated_at_utc']}",
        "",
        "## Scope",
        "",
        f"- Requested period: {report['requested_start_date']} through {report['requested_end_date']}, inclusive.",
        f"- Tickers: {', '.join(report['tickers'])}.",
        "- NVDA is the primary dataset; the other tickers are contextual association queries.",
        "",
        "## Access and request coverage",
        "",
        f"- Historical endpoint access verified at both ends: {preflight.get('coverage_verified', False)}.",
        f"- Fully verified terminal-window coverage: {summary['fully_verified_window_coverage']}.",
        f"- Successful terminal windows: {summary['successful_terminal_windows']}.",
        f"- Failed terminal windows: {summary['failed_terminal_windows']}.",
        f"- Unavailable terminal windows: {summary['unavailable_terminal_windows']}.",
        f"- Empty 200 responses kept as unverified: {summary['empty_unverified_terminal_windows']}.",
        f"- Potentially truncated minimum windows: {summary['potentially_truncated_terminal_windows']}.",
        "",
        "See `request_windows.csv` for every latest request-window state and "
        "`unavailable_periods.csv` for failed, unavailable, empty-unverified, potentially truncated, "
        "or not-yet-collected dates.",
        "",
        "## Article audit summary",
        "",
        f"- Raw ticker observations: {summary['raw_observations']}.",
        f"- Unique articles after ID/URL deduplication: {summary['unique_articles']}.",
        f"- Exact duplicate observations removed: {summary['exact_duplicate_observations_removed']}.",
        f"- Multi-article event groups: {summary['multi_article_event_groups']}.",
        f"- Event-duplicate articles suppressed: {summary['event_duplicate_articles_suppressed']}.",
        f"- Retained direct signals: {summary['retained_direct']}.",
        f"- Retained indirect signals: {summary['retained_indirect']}.",
        "",
        "## Interpretation limitations",
        "",
        "- Finnhub company news supplies headlines, summaries, URLs, sources, related symbols, and metadata—not guaranteed full article text.",
        "- `retrieved_at_utc` records this collection run. It does not establish when an article was historically available to a trading system.",
        "- A response associated with a queried ticker is not treated as proof that the article discusses or materially affects that company.",
        "- Related-ticker collection can miss broad policy news that Finnhub does not associate with one of the configured companies.",
        "- Empty HTTP 200 arrays remain explicitly labeled unverified rather than being interpreted as zero-news proof.",
        "- Relevance and exclusions are deterministic configuration rules. No sentiment or confidence score is invented.",
        "- This project creates a news dataset only; it contains no trading or portfolio-rebalancing logic.",
        "",
        "## Official documentation checked",
        "",
        "- Company-news endpoint and response schema: https://finnhub.io/docs/api/company-news",
        "- Current plan history and request limits: https://finnhub.io/pricing",
        "- Official OpenAPI schema: https://github.com/Finnhub-Stock-API/finnhub-go/blob/master/api/openapi.yaml",
        "",
    ]
    (REPORTS_DIR / "coverage_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to config JSON (defaults to project config.json)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    start = date.fromisoformat(config["collection"]["start_date"])
    end = date.fromisoformat(config["collection"]["end_date"])
    tickers = config["collection"]["tickers"]
    windows = latest_windows()
    articles = read_csv(PROCESSED_DIR / "all_classified_articles.csv")
    signals = read_csv(PROCESSED_DIR / "signal_articles.csv")
    with (PROCESSED_DIR / "processing_summary.json").open("r", encoding="utf-8") as handle:
        processing_summary = json.load(handle)
    preflight_path = PROJECT_ROOT / "raw" / "preflight_report.json"
    preflight = {}
    if preflight_path.exists():
        with preflight_path.open("r", encoding="utf-8") as handle:
            preflight = json.load(handle)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORTS_DIR / "request_windows.csv", windows, REQUEST_FIELDS)
    monthly = monthly_counts(articles, start, end)
    write_csv(REPORTS_DIR / "monthly_counts.csv", monthly, list(monthly[0].keys()))
    exclusions = exclusion_counts(articles)
    write_csv(
        REPORTS_DIR / "exclusion_counts.csv",
        exclusions,
        ["exclusion_reason", "article_count"],
    )
    unavailable, fully_verified = coverage_periods(windows, tickers, start, end)
    write_csv(
        REPORTS_DIR / "unavailable_periods.csv",
        unavailable,
        ["ticker", "from_date", "to_date", "coverage_state", "interpretation"],
    )
    sample_summary = write_sample_reports(signals)

    terminal = [row for row in windows if terminal_status(row)]
    event_groups: dict[str, int] = defaultdict(int)
    for article in articles:
        event_groups[article.get("duplicate_group_id", "")] += 1
    status_counts = Counter(str(row.get("status") or "unknown") for row in windows)
    summary = {
        "fully_verified_window_coverage": fully_verified,
        "successful_terminal_windows": sum(str(row.get("status", "")).startswith("success") for row in terminal),
        "failed_terminal_windows": sum(str(row.get("status", "")).startswith("failed") for row in terminal),
        "unavailable_terminal_windows": sum(str(row.get("status", "")).startswith("unavailable") for row in terminal),
        "empty_unverified_terminal_windows": sum(row.get("status") == "success_empty_unverified" for row in terminal),
        "potentially_truncated_terminal_windows": sum(row.get("status") == "success_truncation_suspected" for row in terminal),
        "raw_observations": processing_summary["raw_observations"],
        "unique_articles": processing_summary["unique_articles_after_id_url_deduplication"],
        "exact_duplicate_observations_removed": processing_summary["exact_duplicate_observations_removed"],
        "multi_article_event_groups": sum(size > 1 for group_id, size in event_groups.items() if group_id),
        "event_duplicate_articles_suppressed": sum(
            "duplicate_event_coverage" in article.get("exclusion_reason", "") for article in articles
        ),
        "retained_direct": sum(row.get("relevance_type") == "direct" for row in signals),
        "retained_indirect": sum(row.get("relevance_type") == "indirect" for row in signals),
    }
    report = {
        "generated_at_utc": utc_now_iso(),
        "requested_start_date": start.isoformat(),
        "requested_end_date": end.isoformat(),
        "tickers": tickers,
        "preflight": preflight,
        "request_status_counts": dict(sorted(status_counts.items())),
        "summary": summary,
        "sample_summary": sample_summary,
        "monthly_counts": monthly,
        "exclusion_counts": exclusions,
        "unavailable_periods": unavailable,
    }
    atomic_write_json(REPORTS_DIR / "coverage_report.json", report)
    write_coverage_markdown(report)
    print(json.dumps(summary, indent=2))
    print(json.dumps(sample_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

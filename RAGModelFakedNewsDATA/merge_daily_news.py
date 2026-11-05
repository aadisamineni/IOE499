#!/usr/bin/env python3
"""Merge provider news on UTC calendar days with synthetic portfolio weights."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYMBOLS = ("NVDA", "SPY", "META", "PFE")
PROVIDERS = {"alpaca": ROOT / "alpaca" / "processed", "finnhub": ROOT / "processed"}
OUTPUT = ROOT / "daily_merged"
WEIGHT_COLUMNS = tuple(f"weight_{symbol}" for symbol in SYMBOLS)
ARTICLE_COLUMNS = ("provider", "article_id", "query_symbols", "published_at_utc", "updated_at_utc",
                   "headline", "summary", "content_html", "author", "publisher", "url",
                   "related_symbols", "category", "retrieved_at_utc", "collection_origin")


def weights_for_day(day, seed):
    """Generate positive weights in 1/10,000 units, adding exactly to one."""
    digest = hashlib.sha256(f"{seed}|{day.isoformat()}".encode()).digest()
    cuts = sorted(random.Random(int.from_bytes(digest, "big")).sample(range(1, 10000), 3))
    boundaries = (0, *cuts, 10000)
    return {name: f"{(boundaries[i + 1] - boundaries[i]) / 10000:.4f}"
            for i, name in enumerate(WEIGHT_COLUMNS)}


def utc_day(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Article timestamp must include a timezone")
    return timestamp.astimezone(timezone.utc).date()


def load_articles(start, end):
    records = {}
    observations = Counter()
    for provider, folder in PROVIDERS.items():
        for symbol in SYMBOLS:
            with (folder / f"{symbol}_news.csv").open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    day = utc_day(row["published_at_utc"])
                    if not start <= day <= end:
                        continue
                    observations[provider] += 1
                    article_id = row.get("article_id", "").strip()
                    if not article_id:
                        raise ValueError(f"Missing article ID in {provider} {symbol}")
                    key = (provider, article_id)
                    candidate = {"provider": provider, "article_id": article_id,
                        "published_at_utc": row["published_at_utc"],
                        "updated_at_utc": row.get("updated_at_utc", ""),
                        "headline": row.get("headline", ""), "summary": row.get("summary", ""),
                        "content_html": row.get("content_html", ""), "author": row.get("author", ""),
                        "publisher": row.get("source", ""), "url": row.get("url", ""),
                        "related_symbols": row.get("related_symbols", row.get("related", "")),
                        "category": row.get("category", ""), "retrieved_at_utc": row.get("retrieved_at_utc", ""),
                        "collection_origin": row.get("collection_origin", ""), "_day": day}
                    if key not in records:
                        records[key] = (candidate, {symbol})
                    else:
                        prior, symbols = records[key]
                        symbols.add(symbol)
                        if candidate["retrieved_at_utc"] > prior["retrieved_at_utc"]:
                            records[key] = (candidate, symbols)
    by_day = defaultdict(list)
    for article, symbols in records.values():
        article["query_symbols"] = ",".join(symbol for symbol in SYMBOLS if symbol in symbols)
        by_day[article.pop("_day")].append(article)
    for articles in by_day.values():
        articles.sort(key=lambda row: (row["published_at_utc"], row["provider"], row["article_id"]))
    return by_day, observations


def merge(start, end, seed, output=OUTPUT):
    if end < start:
        raise ValueError("End date must not precede start date")
    articles_by_day, observations = load_articles(start, end)
    output.mkdir(parents=True, exist_ok=True)
    article_fields = ("date_utc", *WEIGHT_COLUMNS, *ARTICLE_COLUMNS)
    day_fields = ("date_utc", *WEIGHT_COLUMNS, "article_count", "alpaca_article_count", "finnhub_article_count")
    provider_counts = Counter()
    empty_days = 0
    with (output / "news_with_daily_weights.csv").open("w", newline="", encoding="utf-8") as article_handle, \
         (output / "daily_weights.csv").open("w", newline="", encoding="utf-8") as day_handle:
        article_writer = csv.DictWriter(article_handle, fieldnames=article_fields)
        day_writer = csv.DictWriter(day_handle, fieldnames=day_fields)
        article_writer.writeheader()
        day_writer.writeheader()
        day = start
        while day <= end:
            weights = weights_for_day(day, seed)
            articles = articles_by_day.get(day, [])
            counts = Counter(article["provider"] for article in articles)
            provider_counts.update(counts)
            base = {"date_utc": day.isoformat(), **weights}
            day_writer.writerow({**base, "article_count": len(articles),
                                 "alpaca_article_count": counts["alpaca"],
                                 "finnhub_article_count": counts["finnhub"]})
            if not articles:
                empty_days += 1
                article_writer.writerow(base)
            for article in articles:
                article_writer.writerow({**base, **article})
            day += timedelta(days=1)
    summary = {"start_date_utc": str(start), "end_date_utc": str(end),
               "calendar_days": (end - start).days + 1, "days_without_articles": empty_days,
               "articles_by_provider": dict(provider_counts), "input_symbol_rows_by_provider": dict(observations),
               "weight_seed": seed, "weight_precision": "0.0001", "weights_are_synthetic": True,
               "deduplication": "Within each provider by article ID; queried symbols are combined. Cross-provider records are kept separately.",
               "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--seed", type=int, default=499)
    args = parser.parse_args()
    summaries = [json.loads((path / "export_summary.json").read_text(encoding="utf-8")) for path in PROVIDERS.values()]
    start = args.start or min(date.fromisoformat(item["requested_start_utc"]) for item in summaries)
    end = args.end or max(date.fromisoformat(item["requested_end_utc"]) for item in summaries)
    print(json.dumps(merge(start, end, args.seed), indent=2))


if __name__ == "__main__":
    main()

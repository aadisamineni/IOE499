"""Create transparent word-frequency CSVs from the classified news corpus."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "news data pull/processed/all_classified_articles.csv"
STOCK = ROOT / "data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv"
OUT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokens(text):
    return TOKEN.findall(str(text or "").lower())


def parse_date(value):
    stamp = str(value).strip()
    if not stamp:
        return "", ""
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    utc = parsed.astimezone(timezone.utc)
    ny = utc.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
    return utc.date().isoformat(), ny.date().isoformat()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    news = [r for r in read_rows(NEWS) if r.get("relevance_type") in {"direct", "indirect"}]
    stock = {r["trading_date"]: r for r in read_rows(STOCK)}
    overall = Counter()
    overall_docs = defaultdict(set)
    overall_direct = Counter()
    overall_indirect = Counter()
    overall_categories = defaultdict(Counter)
    daily = defaultdict(lambda: {"occurrences": Counter(), "docs": defaultdict(set), "direct": Counter(), "indirect": Counter(), "categories": defaultdict(Counter), "article_ids": set()})

    for article in news:
        utc_date, ny_date = parse_date(article.get("published_at_utc", ""))
        if not utc_date:
            continue
        article_id = article.get("article_id") or article.get("url") or f"row-{len(overall_docs)}"
        words = tokens(f"{article.get('headline', '')} {article.get('summary', '')}")
        counts = Counter(words)
        category = article.get("event_category") or "uncategorized"
        relevance = article.get("relevance_type") or "unknown"
        day = daily[(utc_date, ny_date)]
        day["article_ids"].add(article_id)
        for word, count in counts.items():
            overall[word] += count
            overall_docs[word].add(article_id)
            overall_direct[word] += count if relevance == "direct" else 0
            overall_indirect[word] += count if relevance == "indirect" else 0
            overall_categories[word][category] += count
            day["occurrences"][word] += count
            day["docs"][word].add(article_id)
            day["direct"][word] += count if relevance == "direct" else 0
            day["indirect"][word] += count if relevance == "indirect" else 0
            day["categories"][word][category] += count

    with (OUT / "overall_word_frequency.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["word", "total_occurrences", "article_count", "direct_occurrences", "indirect_occurrences", "first_utc_date", "last_utc_date", "top_event_category"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        dates_by_word = defaultdict(list)
        for (utc_date, _), day in daily.items():
            for word in day["occurrences"]:
                dates_by_word[word].append(utc_date)
        for word, count in sorted(overall.items(), key=lambda x: (-x[1], x[0])):
            top_category = overall_categories[word].most_common(1)[0][0]
            writer.writerow({"word": word, "total_occurrences": count, "article_count": len(overall_docs[word]), "direct_occurrences": overall_direct[word], "indirect_occurrences": overall_indirect[word], "first_utc_date": min(dates_by_word[word]), "last_utc_date": max(dates_by_word[word]), "top_event_category": top_category})

    daily_fields = ["article_utc_date", "article_new_york_date", "word", "word_occurrences", "articles_containing_word", "direct_occurrences", "indirect_occurrences", "top_event_category", "article_count_for_day", "nvda_price_move_1pct_same_date", "nvda_price_move_5pct_same_date", "same_date_event_context_note"]
    with (OUT / "daily_word_frequency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=daily_fields)
        writer.writeheader()
        for (utc_date, ny_date), day in sorted(daily.items()):
            stock_row = stock.get(utc_date, {})
            for word, count in sorted(day["occurrences"].items(), key=lambda x: (-x[1], x[0])):
                top_category = day["categories"][word].most_common(1)[0][0]
                writer.writerow({"article_utc_date": utc_date, "article_new_york_date": ny_date, "word": word, "word_occurrences": count, "articles_containing_word": len(day["docs"][word]), "direct_occurrences": day["direct"][word], "indirect_occurrences": day["indirect"][word], "top_event_category": top_category, "article_count_for_day": len(day["article_ids"]), "nvda_price_move_1pct_same_date": stock_row.get("price_move_1pct", ""), "nvda_price_move_5pct_same_date": stock_row.get("price_move_5pct", ""), "same_date_event_context_note": "same publication-date context only; not evidence of causality"})

    print(f"Selected articles: {len(news)}")
    print(f"Unique words: {len(overall)}")
    print(f"Article dates: {len(daily)}")
    print(f"Wrote: {OUT / 'overall_word_frequency.csv'}")
    print(f"Wrote: {OUT / 'daily_word_frequency.csv'}")


if __name__ == "__main__":
    main()

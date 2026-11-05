#!/usr/bin/env python3
"""Download resumable, paginated Alpaca news, including article content when available."""

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent / "alpaca"
SYMBOLS = ("NVDA", "SPY", "META", "PFE")
START = date(2025, 5, 1)
FIELDS = ("article_id", "query_symbol", "published_at_utc", "updated_at_utc", "headline",
          "summary", "content_html", "author", "source", "url", "related_symbols", "retrieved_at_utc")


def secret(env, path):
    value = os.environ.get(env, "").strip()
    if not value and path:
        value = Path(path).expanduser().read_text(encoding="utf-8").strip()
        if "=" in value:
            value = value.split("=", 1)[1].strip()
        value = value.strip("\"'")
    if not value:
        raise SystemExit(f"Set {env} or provide its credential file.")
    return value


def months(end):
    start = START
    while start <= end:
        next_month = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
        yield start, min(end, next_month - timedelta(days=1))
        start = next_month


def fetch(params, headers):
    url = "https://data.alpaca.markets/v1beta1/news?" + urlencode(params)
    for attempt in range(7):
        try:
            with urlopen(Request(url, headers=headers), timeout=45) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or not isinstance(payload.get("news"), list) or "next_page_token" not in payload:
                raise RuntimeError("Unexpected Alpaca news response format")
            return payload
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 6:
                raise RuntimeError(f"Alpaca news request failed: HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            if attempt == 6:
                raise RuntimeError("Alpaca news request failed: network or timeout") from None
        time.sleep(min(2 ** attempt, 60))
    raise RuntimeError("Alpaca news request failed")


def export(end):
    articles = {}
    for path in sorted((ROOT / "raw").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for article in payload["response"]["news"]:
            try:
                published = datetime.fromisoformat(article["created_at"].replace("Z", "+00:00"))
                if published.tzinfo is None or not START <= published.astimezone(timezone.utc).date() <= end:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            key = str(article.get("id") or "")
            if not key:
                continue
            candidate = (article, payload["retrieved_at_utc"])
            if key not in articles or candidate[1] > articles[key][1]:
                articles[key] = candidate
    out = ROOT / "processed"
    out.mkdir(parents=True, exist_ok=True)
    coverage = []
    for first, last in months(end):
        pages = sorted((ROOT / "raw" / f"{first}_{last}").glob("page_*.json"))
        status = "not_requested"
        if pages:
            complete_sequence = all(path.name == f"page_{i:05}.json" for i, path in enumerate(pages, 1))
            final = json.loads(pages[-1].read_text(encoding="utf-8"))
            status = "complete" if complete_sequence and final["response"].get("next_page_token") is None else "in_progress"
        coverage.append({"start_utc": str(first), "end_utc": str(last), "status": status, "saved_pages": len(pages)})
    with (out / "monthly_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("start_utc", "end_utc", "status", "saved_pages"))
        writer.writeheader()
        writer.writerows(coverage)
    summary = {}
    for symbol in SYMBOLS:
        records = []
        for article, retrieved in articles.values():
            symbols = article.get("symbols") or []
            if symbol not in symbols:
                continue
            records.append({"article_id": article["id"], "query_symbol": symbol,
                "published_at_utc": article.get("created_at", ""), "updated_at_utc": article.get("updated_at", ""),
                "headline": article.get("headline", ""), "summary": article.get("summary", ""),
                "content_html": article.get("content", ""), "author": article.get("author", ""),
                "source": article.get("source", ""), "url": article.get("url", ""),
                "related_symbols": ",".join(symbols), "retrieved_at_utc": retrieved})
        records.sort(key=lambda row: (row["published_at_utc"], str(row["article_id"])))
        with (out / f"{symbol}_news.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        summary[symbol] = {"article_count": len(records), "with_content": sum(bool(r["content_html"]) for r in records),
                           "first_publication_utc": records[0]["published_at_utc"] if records else None,
                           "last_publication_utc": records[-1]["published_at_utc"] if records else None}
        print(f"Alpaca {symbol}: {len(records)} articles, {summary[symbol]['with_content']} with content", flush=True)
    (out / "export_summary.json").write_text(json.dumps({"requested_start_utc": str(START), "requested_end_utc": str(end),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "symbols": summary}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file")
    parser.add_argument("--secret-file")
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.end < START:
        parser.error("--end must be on or after 2025-05-01")
    if args.export_only:
        export(args.end)
        return
    headers = {"APCA-API-KEY-ID": secret("APCA_API_KEY_ID", args.key_file),
               "APCA-API-SECRET-KEY": secret("APCA_API_SECRET_KEY", args.secret_file)}
    try:
        for first, last in months(args.end):
            folder = ROOT / "raw" / f"{first}_{last}"
            folder.mkdir(parents=True, exist_ok=True)
            params = {"symbols": ",".join(SYMBOLS), "start": f"{first}T00:00:00Z",
                      "end": f"{last}T23:59:59.999999Z", "sort": "asc", "limit": 50,
                      "include_content": "true"}
            page = 1
            while True:
                path = folder / f"page_{page:05}.json"
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    data = saved["response"]
                else:
                    data = fetch(params, headers)
                    saved = {"request": params.copy(), "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                             "response": data}
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
                    temporary.replace(path)
                    print(f"Alpaca {first}..{last} page {page}: {len(data['news'])} articles", flush=True)
                    time.sleep(0.5)
                token = data["next_page_token"]
                if not token:
                    break
                if token == params.get("page_token"):
                    raise RuntimeError("Alpaca returned a repeated pagination token")
                params["page_token"] = token
                page += 1
            export(args.end)
    finally:
        export(args.end)


if __name__ == "__main__":
    main()

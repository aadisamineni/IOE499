# News data for RAG

This folder collects news records for NVDA, SPY, META, and Pfizer (PFE),
published from May 1, 2025 through the current UTC date. Alpaca and Finnhub
are kept in separate folders because they are different provider feeds.

The large merged-news and Alpaca NVDA CSVs use Git LFS. Install Git LFS before
cloning or run `git lfs pull` after installing it to retrieve their contents.

## Daily merged data and synthetic portfolio weights

`daily_merged/news_with_daily_weights.csv` is ordered by UTC calendar day and
article publication time. Each provider article is one row, with the same four
weights repeated on every row for a given day. If a day has no articles, it
still has one row with empty article fields. The four weights are nonnegative,
stored to four decimal places, and add exactly to 1. They are arbitrary
synthetic values, not calculated portfolio allocations or market data.

`daily_merged/daily_weights.csv` has exactly one row per calendar day, including
weekends, with the weights and article counts. `daily_merged/merge_summary.json`
records the input period, counts and seed. Articles returned for more than one
symbol are deduplicated within each provider by ID and their query symbols
are combined. Records from different providers remain separate, so the same
underlying story can appear more than once. The article timestamp, provider,
publisher, link, summary and any available article content remain in the merge.

Rebuild after refreshing the source files:

```sh
python3 RAGModelFakedNewsDATA/merge_daily_news.py
```

The default seed is `499`; `--seed`, `--start`, and `--end` can be supplied
to choose another deterministic series or date range. The same date and seed
always receive the same weights regardless of article count.

## Alpaca news and article content

`alpaca/processed/{NVDA,SPY,META,PFE}_news.csv` contains Alpaca articles,
including `content_html` where the provider supplies article text. Content may
be empty, and the same article can appear in several symbol files. Raw API
pages and their retrieval times are in `alpaca/raw/`. The API filters by
associated symbol, and `related_symbols` preserves those associations. Its
pagination is sorted by update time; publication and update timestamps are
both retained. Raw content can contain HTML and should be treated as untrusted
input by applications displaying it.

To collect or resume, set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` and run:

```sh
python3 RAGModelFakedNewsDATA/pull_alpaca_news.py
```

Alternatively, provide `--key-file` and `--secret-file` paths. Do not commit
credentials. Pages are saved as they are downloaded; rerunning resumes from
the last saved page. A completed current-day window is a snapshot, so rerun
on a later UTC date to fetch an updated current-month window. To rebuild CSVs
from existing pages, use `--export-only`. `alpaca/processed/export_summary.json`
counts the saved records and records with nonempty content; an export alone is
not evidence that every month's pagination has finished. Check
`alpaca/processed/monthly_coverage.csv` for pagination status. A completed
window confirms the provider returned its final page, not that its news feed
contains every article published by other sources.

## Finnhub news

Run from the repository root:

```sh
export FINNHUB_API_KEY="your-key"
python3 RAGModelFakedNewsDATA/pull_news.py
```

Alternatively, pass `--key-file` with a local credential file containing either
the token or `api_key=...`. Do not commit credentials. Collection can take a
long time because requests are paced and busy windows are subdivided. It is
resumable: rerunning skips successful saved windows, except for today's window,
which is refreshed. To export already saved responses without making requests:

```sh
python3 RAGModelFakedNewsDATA/pull_news.py --export-only
```

`processed/{NVDA,SPY,META,PFE}_news.csv` contains one record per Finnhub article
ID per query symbol, sorted by UTC publication time. Articles can appear in
more than one symbol's file. The exports include relevant existing NVDA and
META snapshots from the repository's `news data pull/raw` directory, as well
as newly downloaded snapshots. `collection_origin` and `retrieved_at_utc`
identify the source and retrieval time of the selected snapshot. These are
provider-associated results, not independently verified relevance labels.

The collector stores raw responses and a resumable request manifest under
`raw/`. Status `success_empty_unverified` means Finnhub returned an empty list;
it does not establish that no news existed. A one-day window marked
`success_truncation_suspected` may be incomplete. Historical availability may
vary by account and date. The current UTC date is incomplete until it ends.
`processed/export_summary.json` describes the records actually exported; the
request manifest contains per-window collection status. The
`processed/daily_coverage.csv` export shows the latest status for each symbol
and date, separately for the new and previously saved pulls. An interrupted
run leaves untouched dates marked `not_requested`.

Finnhub provides headlines, summaries, links, and metadata here, not guaranteed
full article text. These are real provider records, not fabricated news, and
retrieval timestamps do not prove what was available at publication time.

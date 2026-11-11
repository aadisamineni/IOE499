# Signal Desk — IOE 499 results dashboard

**Current status: frontend/input-data preview only. The actual project pipeline
is not connected. Portfolio allocations and model briefs show pending states;
generation is disabled. No random or synthetic allocation values are displayed,
exported, or exposed by the daily API.**

The initial prototype displayed demonstration weights already present in the
remote data branch; that behavior has been removed at the user's request.
The implementation notes below describe the initial adapter and available
source files, not completed project pipeline results. Inference helpers remain
unconnected for future integration.

A local presentation frontend for the RAG research workflow: daily portfolio
weights, provider news activity, searchable article evidence, and optional
Qwen briefings with source IDs. Uses Python's standard library and browser-native
HTML/CSS/JavaScript; no frontend build, cloud API key, or paid inference is needed.

## Run

From the repository root:

```sh
python3 rag_dashboard/server.py
```

Open **http://127.0.0.1:8765**. The first launch indexes available news into an
ignored SQLite FTS5 cache. Subsequent launches reuse the cache. Stop with Ctrl+C.

By default the app reads `RAGModelFakedNewsDATA` directly from the fetched
`origin/rag-pipeline` Git commit. It does not switch branches or change source
files. Fetch that branch first if it is not present locally.

For a local checkout with fully downloaded Git LFS files:

```sh
python3 rag_dashboard/server.py --data-dir /path/to/RAGModelFakedNewsDATA --refresh
```

Use `--refresh` after changing local CSVs; local-folder caches are not monitored
for edits. For remote-ref mode, a different commit automatically selects a new
cache. `--port 8766` selects another port. `--git-ref` selects another data ref.

## What is connected

- `rag-pipeline`: saved `daily_weights.csv`, `merge_summary.json`, and individual
  Alpaca/Finnhub symbol exports. The merged 133 MB LFS article CSV is unnecessary.
- `RAG_Pipelines`: compatible with its Ollama/Qwen model and local endpoint;
  the dashboard implements its own filtered evidence and inference adapter.
- Overview: daily synthetic allocation, all-provider daily news counts, and
  collection totals. Asset filters apply to the evidence feed and brief; global
  portfolio/activity metrics remain explicitly labeled as all assets.
- Evidence explorer: date, asset, provider, keyword search, pagination, complete
  saved summaries, source links, and JSON snapshot export.
- Pipeline & provenance: exact data commit, indexed counts, missing inputs,
  retrieval method, and interpretation limits.

In the initial fetched snapshot, **108,278** provider-deduplicated records are
reported in the collection summary, while **101,577** can be indexed from the
available CSVs. `alpaca/processed/NVDA_news.csv` is an unresolved Git LFS pointer.
Cross-listed stories can appear in other symbol exports, so the difference is
not the full size of that NVDA export. Search completeness is marked explicitly.
The source has **554 UTC days**, May 1, 2025–November 5, 2026. The latest day is
a partial-day historical snapshot, not current live coverage.

## Optional evidence brief

Start Ollama with the `qwen3:30b-instruct` model installed. The status indicator
checks availability; model inference occurs only when Generate is clicked.
`--model NAME` uses another installed Ollama model. No model is downloaded by
the app. All model traffic goes to `127.0.0.1:11434` with proxies disabled.

The adapter sends up to five saved headline/summary records matching the selected
UTC day, symbol and provider. Search keywords rank results with SQLite FTS5 BM25;
without keywords the newest matching records are used. This is lexical retrieval,
not an embedding index or trained return predictor. The question does not silently
change search filters. Requests exceeding the context byte guard are rejected
rather than truncating articles. Calls can take up to three minutes.

The answer displays its supplied evidence and checks bracketed provider/article
IDs against that evidence. It flags unknown IDs, missing valid citations, and
truncated responses. This checks citation identity, not whether each claim is
entailed by a source. Article content is rendered as text, never provider HTML.
Changing the selection prevents an in-flight answer appearing against a new date.

Daily weights are arbitrary seeded demonstration values. They are not generated
by Qwen, optimized holdings, measured returns, backtested strategies, or hedge
recommendations. No P&L, sentiment score, or predictive accuracy is fabricated.

## Checks

```sh
python3 -m unittest discover -s rag_dashboard -p 'test_*.py' -v
node --check rag_dashboard/app.js
```

The tests use temporary synthetic fixtures to verify provider deduplication,
symbol union, date boundaries, keyword escaping, multiline CSV parsing,
unavailable LFS handling, and inference evidence/citation checks. They do not
call Ollama or alter source data.

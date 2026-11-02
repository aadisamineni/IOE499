# April 2026 hourly price moves and news timing

Run from the repository root with the existing project environment:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python hourly_models/april_news_timing/analyze.py
```

Dependencies: root requirements.txt. Reuses hourly_models' NASDAQ calendar, price validation, timestamp parser and collection-coverage logic. Inputs remain unchanged and SHA-256 hashes are checked.

Rank all valid full 60-minute April bars by absolute (close/open - 1). Export all moves >=1%; show the top 10 in the report and graph. This is an exploratory descriptive selection, not a predictive evaluation. Overnight gaps and final 30-minute bars are not ranked. Start/end and article timestamps are exported in UTC and America/New_York.

Equal comparison windows: before [start-1h,start), during [start,end), after [end,end+1h). Also count [start-24h,start) and [end,end+24h). An article exactly at bar end belongs to after. Within-bar price timing is unknown. The same article may match multiple moves and phases across different events; aggregate counts are event–article links, not unique articles.

Use existing direct/indirect relevance classifications, excluding rows already marked excluded. Deduplicate exact-copy groups and article IDs; retain distinct coverage of the same topic. Direct-only counts are also exported. All event windows have successful recorded collection coverage, which does not guarantee all publishers/news were captured. No explicit historical revision timestamps are available; current headline/summary versions may differ from original versions.

Files: event_summary.csv (rank, times, OHLC endpoints, return, threshold, window counts/direct counts/coverage); event_article_matches.csv (one event–article pair, source headline/summary/link/features, phase, window booleans and time distances); april_bar_audit.csv (all original April bars with calendar/validity fields); timestamp_review.csv (ambiguous timestamps across source); results.json (counts, hashes, checks); news_timing.png; report.html (top-10 article lists and all event counts).

Content interpretation: after a move does not mean reporting that move. Review headlines AND summaries. For example, April 24 article 139882100 claims a closing market-cap milestone but is timestamped 12:23 New York, before the session closes. This is an availability/version warning, not proof of the true original publication time. Live-coverage pages also require caution. No comprehensive semantic labeling or causal inference is performed.

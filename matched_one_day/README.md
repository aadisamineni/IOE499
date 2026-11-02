# Previous-day stock/news matching

This workflow matches every NVDA trading session to relevant news published on
the immediately preceding **New York calendar date**. For example, the stock
row for December 2 is matched to articles dated December 1 in New York.

This is deliberately different from both a rolling 24-hour window and the
previous NASDAQ session. A Monday session therefore matches Sunday news only.

## Two-percent movement feature

`price_move_2pct` is derived from `adjusted_close_return_1d`:

- `-1` when the adjusted close-to-close return is at most -2%;
- `1` when the adjusted close-to-close return is at least +2%;
- `0` otherwise.

The thresholds are inclusive and the feature is present in both output tables.

## Run

From the repository root:

```sh
.venv/bin/python matched_one_day/match_previous_day.py
```

Optional `--start`, `--end`, `--threshold`, and `--output` arguments are
available. Inputs are read without modification from the processed NVDA stock
and classified-news files.

## Outputs

Files are written to `matched_one_day/output/`:

- `daily_stock_news_summary.csv`: one row for every stock session, including
  dates with no eligible news;
- `stock_news_matches.csv`: one row per stock-session/article match;
- `timestamp_review.csv`: eligible articles rejected because their publication
  timestamp was missing or ambiguous;
- `validation_report.json`: matching rules, counts, source hashes, limitations,
  and validation results.

Only articles already classified as `direct` or `indirect` are matched.
Publication timestamps must contain an explicit timezone and are converted to
`America/New_York` before their calendar date is assigned.

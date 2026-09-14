# NVDA daily stock research dataset

The existing entry point, `NvidiaDatapull.py`, downloads Yahoo Finance daily data
through yfinance for **2025-10-01 through 2026-05-31 inclusive**. Work belongs on
`ZAS`; incorporate upstream updates from `origin/Developing` and push with
`git push origin HEAD:ZAS`.

## Run

From the repository root (tested with Python 3.13):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python NvidiaDatapull.py
```

`requirements.txt` pins the installed environment, including transitive packages.
Direct dependencies are yfinance (download), pandas and numpy (features and
validation), and pandas_market_calendars (NASDAQ sessions). The environment and
yfinance cache remain local and are ignored by Git. Internet access to the Python
package index is needed for installation; Yahoo access is needed for each run.

## Outputs and provenance

- `data/raw/NVDA_daily_2025-10-01_2026-05-31.csv`: requested provider columns.
- `data/raw/NVDA_daily_warmup_2025-07-03_2025-09-30.csv`: separate 90-calendar-day
  warm-up request; weekends/holidays do not receive rows.
- `data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv`: source columns
  plus fourteen derived columns, trimmed after computing features and lags.
- `data/raw/NVDA_daily_2025-10-01_2026-05-31_metadata.json`: successful retrieval
  times (UTC), versions, request settings, actual coverage, counts, validation,
  corporate-action and unusual-return review records, and conventions.

The script prints paths, first/last dates, count, first five source rows,
missing-value counts, and validation findings. Reruns refresh the files; Yahoo
may revise history. On failure it exits nonzero and writes a separate
`data/raw/NVDA_daily_2025-10-01_2026-05-31_failure.json` with the exact error and
available validation. Existing successful outputs may be from an older run;
check timestamps/status. Successfully received raw requests are retained even
if later validation or downloading fails. No replacement data is synthesized.

The [current yfinance history documentation](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html)
specifies inclusive `start` and exclusive `end`. Main download uses `2026-06-01`
as the exclusive end; warm-up uses `2025-10-01`. Explicit settings are daily
interval, `auto_adjust=False`, `back_adjust=False`, `actions=True`, `repair=False`,
`keepna=True`, `rounding=False`, `prepost=False`, `timeout=30`, and
`yf.config.debug.hide_exceptions=False` to surface errors (the installed yfinance
version deprecates the older `raise_errors=True` argument).

“Raw” means provider fields preserved, **not original pre-split trade prices**.
Yahoo's historical OHLC is generally split-adjusted, while adjusted close also
reflects dividend distributions. See [Yahoo's adjustment explanation](https://help.yahoo.com/kb/SLN28256.html)
and the [yfinance project discussion of split-adjusted close](https://github.com/ranaroussi/yfinance/discussions/1682).
The script applies no further adjustments to saved source columns or volume,
and does not reconstruct pre-split prices. The adjustment factor is used only
internally for the open-to-close feature.

## Data dictionary

`t` denotes a NASDAQ session; `A` is adjusted close and `r` its simple daily
return. Prices/dividends are USD per share; returns are fractions (0.01 = 1%).
CSV missing values are blank, never filled with zero.

| Column | Definition |
| --- | --- |
| `trading_date` | Exchange-local session date, America/New_York, `YYYY-MM-DD`. |
| `ticker` | `NVDA`. |
| `open` | Provider daily opening price; split-adjusted convention described above. |
| `high` | Provider daily highest price, same basis as open. |
| `low` | Provider daily lowest price, same basis as open. |
| `close` | Provider daily closing price, same basis as open. |
| `adjusted_close` | Provider closing price adjusted for splits and dividends. |
| `volume` | Provider daily share volume, preserved as returned. |
| `dividends` | Provider cash distribution per share on the ex-dividend date; zero means none reported. |
| `stock_splits` | Provider new-shares/old-shares ratio on the action date; zero means none reported. |
| `adjusted_close_return_1d` | `A[t] / A[t-1] - 1`; requires previous session. |
| `adjusted_open_to_close_return` | `A[t] / (open[t] * A[t] / close[t]) - 1`; same adjustment factor on both prices, algebraically `close/open - 1`. |
| `adjusted_close_return_5d` | `A[t] / A[t-5] - 1`; five-session trailing return. |
| `daily_return_volatility_20d` | Sample standard deviation (`ddof=1`) of 20 daily returns ending at t, no annualization. |
| `average_volume_20d` | Mean volume over 20 sessions ending at t, in shares. |
| `price_move_5pct` | Adjusted close change from the previous session: `1` for at least +5%, `-1` for at most -5%, otherwise `0`. Inclusive thresholds; unavailable changes remain blank. Known after today's close, not a future target. |
| `price_move_1pct` | Same adjusted close comparison at inclusive ±1% thresholds: `1` for at least +1%, `-1` for at most -1%, otherwise `0`. Unavailable changes remain blank; known after today's close. |
| `adjusted_close_return_1d_lag1_session` | Previous session's adjusted daily return. |
| `adjusted_open_to_close_return_lag1_session` | Previous session's adjusted intraday return. |
| `adjusted_close_return_5d_lag1_session` | Previous session's trailing five-session return. |
| `daily_return_volatility_20d_lag1_session` | Previous session's trailing daily-return volatility. |
| `average_volume_20d_lag1_session` | Previous session's trailing average volume. |
| `price_move_5pct_lag1_session` | Previous session's 5% movement classification; available for today's premarket use. |
| `price_move_1pct_lag1_session` | Previous session's 1% movement classification; available for today's premarket use. |

Rolling windows require all observations. Warm-up supplies the earlier closes,
returns and volumes, including the first output row's lagged features. The script
fails validation if warm-up is incomplete, rather than silently treating adjacent
downloaded rows across missing sessions as consecutive sessions.

## Timing and validation

Same-day high, low, close, volume and **every unlagged derived feature** are
available only after that session ends. For premarket predictions on date t,
use the explicitly named `_lag1_session` features; do not feed date t's completed
bar into the predictor. Open itself becomes available at the opening trade.
This is a retrospectively adjusted research snapshot, not a point-in-time
archive of provider revisions. A one-session lag handles feature timing but
does not turn revised historical prices into vintage data. No news is joined
and no future prediction targets are created.

Both requests are checked against the packaged NASDAQ exchange calendar,
including early closes. Checks cover sorted/unique dates, missing and unexpected
sessions, null/nonfinite fields, positive prices, nonnegative volume, and OHLC
bounds. Source failures stop processed output. Nonzero corporate actions and
absolute adjusted daily returns of at least 10% are retained and listed for
review (including warm-up); they are not automatically deleted. The calendar is
versioned package data, not a live exchange-status feed.

Run the focused offline checks with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

# Match existing NVDA shifts to preceding news

From the repository root, use the existing project-local environment:

```sh
.venv/bin/python matching/match_news.py --start 2025-09-15 --end 2026-05-15
.venv/bin/python -m unittest discover -s tests -v
```

If needed, create `.venv` with `python3 -m venv .venv` and install the root
`requirements.txt`. This program uses pandas and pandas_market_calendars already
pinned there. Matching is offline; it does not call Finnhub, download prices,
recalculate returns, generate labels, or train a model.

## Inputs and limitations

- Stock: `data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv`.
  Dates actually span October 1, 2025–May 29, 2026. Match only available flagged
  sessions inside the requested range. The 12 September 15–30 sessions lack
  original flags; report these as missing, without inventing flags from warm-up.
- News: `news data pull/processed/all_classified_articles.csv`, with UTC
  publication dates October 1, 2025–May 1, 2026. Use the 9,740 articles already
  classified `direct` or `indirect` by the existing pipeline. Preserve all their
  features, including original exclusion reasons and event-group information.
  The 43,333 records marked `not_relevant` are recorded in `excluded_articles.csv`.
- Use the full audit CSV rather than `signal_articles.csv`: the latter suppresses
  distinct coverage of an event. No headline/event-group deduplication is done
  here, and existing earnings/listicle/other content exclusions are not reapplied.
  Only identical input records are removed. Upstream ID/URL deduplication has
  already occurred; terminal raw responses are audited for collapsed content
  variants, listed in `article_revision_review.csv`.

Original files are read-only inputs. SHA-256 hashes of primary inputs are
verified before/after the run and recorded in `validation_report.json`.

## Existing indicators and timing

There are **two signed source columns**, not four independent binary columns:

| Output direction view | Exact existing value used |
| --- | --- |
| `shift_1pct_up` | `price_move_1pct == "1"` |
| `shift_1pct_down` | `price_move_1pct == "-1"` |
| `shift_5pct_up` | `price_move_5pct == "1"` |
| `shift_5pct_down` | `price_move_5pct == "-1"` |

The two source columns are copied unchanged into every event and match. Four
binary direction views (0/1) merely decode those values. No threshold or price
calculation occurs. Five-percent categories overlap with same-direction
one-percent categories in the actual file: four up days and one down day.
The filtered review CSVs preserve that overlap; their counts must not be summed
to obtain unique events. The `_lag1_session` source columns are not today's flags.

`NvidiaDatapull.py`, root `README.md`, and stock metadata establish that the flags
represent adjusted **close-to-close** price movement. The matching cutoff is
therefore the **previous NASDAQ session's close**, not today's close or midnight.
NASDAQ's versioned calendar handles holidays, early closes and DST. The event
stores UTC and America/New_York cutoffs, the previous session date, and the end
of the price interval. This timing is documented, not a provisional assumption.

Articles must satisfy `window_start <= published < cutoff`. The windows are:

- 24 elapsed hours and 72 elapsed hours before the cutoff.
- Seven **calendar days** before the cutoff, preserving the New York wall-clock
  time; this spans 167 or 169 hours across DST, otherwise 168.

Store one row per event and exact article record (`article_match_key`), with
`in_24h`, `in_72h`, `in_7d`, elapsed `hours_before_cutoff`, and mutually exclusive
`lookback_bucket`. Exact 24/72-hour boundaries belong to the shorter bucket;
the seven-day lower boundary is included. Articles may link to multiple events.

## Timestamps, revisions, and coverage

Publication strings must have an explicit time and UTC offset/Z. Missing,
date-only, invalid, or timezone-ambiguous publications are excluded from matching
and saved in `timestamp_review.csv`. No times or zones are assigned to articles.
Original timestamp strings remain alongside normalized UTC/New York columns.

The current news has no update/revision timestamps, and was retrieved in
September 2026. Thus **all matches have uncertain historical availability**.
Retrieval timestamps are not treated as publication dates or proof of a later
revision. Raw content variants flag possible revisions/identity conflicts;
they do not establish when a change happened. These uncertain cases are retained
and explicitly marked, not declared historically verified.

If an article contains `updated_at_utc`, `modified_at_utc`, `revised_at_utc`, or
`first_available_at_utc`, the matcher excludes the event–article pair when that
time is at/after the cutoff. A separate earlier version can still match if its
own timestamps qualify; no earlier text is reconstructed. Ambiguous revision
times are reviewed and excluded from strict matches. Current source data has
no such known revision timestamps, so there are zero known-revision exclusions.

Collection coverage is evaluated against the latest terminal request-manifest
entries for all nine configured tickers. Since the collector explicitly filters
UTC publication dates, coverage-day bounds are UTC (not article timestamp
guesses). Failed, missing, empty-unverified, and potentially truncated requests
are distinguished from successful collection. Outside-range dates are known
collection gaps. Windows near the start or after the end receive explicit flags.
Successful request coverage does not establish complete publisher coverage.

Every flagged day remains in the summary and HTML even with no matches. Zero
matches with coverage gaps are not interpreted as “no news.” Current collection
also flags potential NVDA truncation on February 26 and March 17–18, 2026.

## Outputs

All files are under `data/matched/2025-09-15_2026-05-15/` (dates follow CLI args):

| File | Contents |
| --- | --- |
| `event_summary.csv` | One row per flagged session, both original signed flags, four direction views, cutoff/timing, nested article counts, coverage and quality flags. |
| `event_article_matches.csv` | Unique event–article links, original news fields/features, normalized timestamps, hours/bucket/window flags, and availability/revision quality flags. |
| `1pct_up_articles.csv`, `1pct_down_articles.csv`, `5pct_up_articles.csv`, `5pct_down_articles.csv` | Overlapping filtered views of the matches. |
| `report.html` | Each event's categories, cutoff, counts and expandable linked headlines; includes zero-match days and coverage warnings. |
| `timestamp_review.csv` | Ambiguous/missing timestamps, original fields and review reason; header-only when none. |
| `article_revision_review.csv` | Known revision fields or raw content variants requiring review. |
| `excluded_articles.csv` | Out-of-scope relevance, exact copies, unusable timestamps, or articles with no eligible event window; preserves the original news fields. |
| `excluded_event_article_pairs.csv` | Pair-specific known revision/availability exclusions; header-only when none. |
| `missing_stock_sessions.csv`, `stock_date_review.csv` | Requested sessions without flags and invalid stock-date records. |
| `collection_coverage_by_day.csv` | UTC date/ticker collection state, including outside-range and truncation flags. |
| `validation_report.json` | Input hashes, versions, counts, overlaps, coverage limitations, exclusions and validation findings. |

Validation checks all available flagged days, original indicator equality,
strict pre-cutoff timing, unique pairs, and all window memberships/counts.
Tests cover holiday/early-close cutoffs, DST, exact boundaries, ambiguous times,
revisions, duplicate records, overlapping categories and zero-match days.
Reruns overwrite only this matching output directory. These links are temporal
associations and do not establish that news caused a price movement.

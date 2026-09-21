# Distinctive words before 1% up/down days

Run from the repository root:

```sh
.venv/bin/python News_word_frequency/distinctive_terms.py
.venv/bin/python -m unittest discover -s tests -p test_distinctive_terms.py -v
```

Uses only Python's standard library and existing matched articles; no download,
return recalculation, source modification, sentiment labels, or model training.

## Rules

- Existing signed 1% movement flags define up/down event days. Use articles in
  the existing strict 24-hour window before the previous session's close.
- Primary comparison includes only events marked `collection_window_verified_24h`.
  All 103 events remain in the audit and presence matrix; 11 have insufficient
  coverage and are excluded from the primary denominator. Verified collection
  does not prove complete source coverage or historical article availability.
- Extract headline and summary separately, lowercase, preserve original token
  adjacency within punctuation-delimited sentences. Remove URLs and numeric-only
  tokens as barriers; keep alphanumeric tokens such as H200. Single-character
  tokens are also barriers. No stemming or domain-term exclusion.
- Expand negative contractions to preserve `not`. Keep `no`, `not`, and `nor`.
  Drop common English words as standalone candidates and phrases containing only
  stopwords. Phrases such as `the future` can remain because they contain a
  substantive word. Full stopwords are saved in metadata.json.
- Each word or consecutive two-token phrase counts once per event day, regardless
  of article count or repetitions. Require five combined distinct event days.
- Rank by up-day prevalence minus down-day prevalence (percentage points).
  Up and down groups use their own denominators. Negative differences favor down.
  Ties are resolved by support, then alphabetically in top-20 views.
- Reuse all existing matched direct/indirect news, including upstream excluded
  rows. No additional editorial, earnings, or same-event suppression is applied.

## Outputs

- `report.html`: top 20 terms for each direction, counts, percentages and expandable
  supporting articles from both directions (up to three distinct articles each).
- `term_prevalence.csv`: every term passing the five-day minimum, not just winners.
- `top20_up.csv`, `top20_down.csv`: ranked differences.
- `supporting_articles.csv`: original headlines, summaries, IDs, URLs, publication
  times, cutoffs and event dates for the examples.
- `event_term_presence.csv`: all events, inclusion status and binary term columns
  prefixed with `term::`. The vocabulary is selected from the primary sample.
- `event_audit.csv`: per-event coverage status and matched-article counts.
- `sensitivity_top20.csv`: rankings for minimum support of 3, 5 and 10 days, with
  verified windows only and with all events (the latter has known coverage bias).
- `metadata.json`: rules, denominators, overlap counts, source hashes and checks.

The primary sample has 92 days (51 up, 41 down) and 5,115 unique articles. None
of these articles are matched to both directions in the primary 24-hour sample,
though distinct stories may cover the same episode. Some terms are boilerplate,
commentary or generic phrases; inspect examples before interpreting them.

8,795 candidates pass the minimum. Ranking so many candidates can produce large
chance differences. These full-sample results are exploratory, not significance
claims or validated predictive features. Minimum-support sensitivity is not a
substitute for later-period validation. No temporal holdout was performed here;
future model vocabulary must be selected on training data only. Coverage gaps,
retrospective availability, and correlated news episodes remain limitations.

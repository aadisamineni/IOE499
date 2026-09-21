# Words before 1% and 3% up/down days

Run from the repository root with the existing local environment (standard
library only; no new dependencies):

```sh
.venv/bin/python News_word_frequency/group_word_frequency.py
```

Open `report.html` for four top-10 tables with bars. `top10_words.csv` contains
the same results; `top10_words_unfiltered.csv` includes common words and numbers.
`all_word_counts.csv` contains the full filtered vocabulary. Counts rank total
occurrences, with alphabetical ties. Article prevalence and occurrences per
1,000 filtered tokens are supplied for comparing groups of different sizes.

Rules:

- Use the existing matcher’s seven-calendar-day windows, strictly before the
  previous session’s close, for events within September 15–May 15.
- Reuse `price_move_1pct` unchanged for the 1% groups. No 3% flag exists in the
  source: new analysis-only groups use the saved `adjusted_close_return_1d`
  at inclusive `>=0.03` and `<=-0.03`. No prices or returns are recomputed and
  no source files or flags are overwritten.
- Count headline plus summary, lowercased, once per `article_match_key` within
  each group even when the article matches multiple days. Distinct coverage is
  retained. All matched direct/indirect articles are used, including articles
  marked excluded by the older signal-filtering pipeline.
- The main view removes an explicit common-English stopword list, numbers,
  nonalphabetic tokens and single letters. No stemming, domain-word removal,
  sentiment labels, or topic model is used. Exact rules are in `metadata.json`.
- The 3% groups are subsets of same-direction 1% groups. Articles can also
  appear in both upward and downward groups through different matched days.
- `group_events.csv` audits event membership and saved source values;
  `group_articles.csv` audits the unique articles counted within each group.
  `group_summary.csv` reports event/article counts, unmatched days, and coverage
  warnings. Original input hashes are verified before/after the run.

These are exploratory word counts, not evidence of predictive value or causality.
Stock flags begin October 1, news ends May 1, and all news matches have uncertain
historical availability. Larger groups naturally generate larger raw counts.

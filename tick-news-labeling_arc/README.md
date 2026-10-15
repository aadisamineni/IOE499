# Tick/news event-dataset builder

This folder contains preprocessing code only. It does not train a model.

The pipeline converts deduplicated news into fixed, non-chaining five-minute
event clusters and joins each cluster to SIP quote midpoints. It produces
continuous returns and volatility/spread-adjusted `up`, `neutral`, and `down`
labels at 5, 15, and 60-minute horizons.

## Design

For each NASDAQ regular-session news cluster, the pipeline:

1. keeps non-excluded representative articles;
2. uses direct Nvidia news by default (`--include-indirect` is optional);
3. opens a fixed five-minute window at the first article and combines all news
   inside that window without allowing the cluster to extend indefinitely;
4. sets the decision time to the first complete minute after the cluster
   window closes;
5. uses the most recent valid SIP bid/ask midpoint at that decision time;
6. measures forward returns at the requested horizons;
7. derives a neutral band from trailing one-minute volatility and the observed
   bid/ask spread; and
8. flags horizons containing later news rather than silently attributing their
   entire return to the original cluster.

The source quote files are streamed one UTC day at a time. The 90 GB combined
CSV is deliberately not read. Small per-day Parquet files containing the final
valid midpoint of each minute are cached under the output directory.

## Inputs

Defaults point at the existing repository data:

- `news data pull/processed/all_classified_articles.csv`
- `alpaca data pull/raw/sip/quotes/NVDA_quotes_YYYY-MM-DD.jsonl.gz`

`published_at_utc` is used as the news time. `retrieved_at_utc` is never used as
historical availability time.

An optional benchmark quote directory can be supplied. It must use the same
Alpaca partition format for a symbol such as QQQ or SOXX. When present, the
label return is:

```text
NVDA log return - benchmark_beta * benchmark log return
```

Without a benchmark, the label return is the raw NVDA log return and the
output records that limitation.

## Cluster usage

Every row is one event cluster, not one tick or one article. A cluster can have
different labels at different horizons—for example, `up` at five minutes and
`down` at sixty minutes after a reversal.

Use `clean_5m == true` for the cleanest initial experiment. A clean horizon has
no subsequent cluster starting between the current cluster's window end and
the horizon endpoint. Longer horizons are retained with contamination counts
so they can be studied as persistence/reversal outcomes.

## Run on the compute environment

Nothing in this folder is run automatically. On the compute environment:

```sh
python -m pip install -r tick-news-labeling_arc/requirements.txt
python tick-news-labeling_arc/build_dataset.py \
  --output-dir tick-news-labeling_arc/output
```

To produce benchmark-adjusted returns after collecting QQQ SIP quotes in the
same daily format:

```sh
python tick-news-labeling_arc/build_dataset.py \
  --benchmark-symbol QQQ \
  --benchmark-quotes-dir /path/to/raw/sip/quotes \
  --output-dir tick-news-labeling_arc/output
```

Useful options:

```text
--include-indirect
--cluster-minutes 5
--latency-minutes 1
--horizons 5,15,60
--pre-vol-minutes 30
--z-cutoff 1.0
--spread-multiple 1.0
--write-csv
--force-rebuild-cache
--allow-missing-quote-days
```

The build stops on a missing required quote partition by default. The final
option permits incomplete price coverage and should be used only for a
deliberately partial research run.

## Outputs

- `news_price_events.parquet`: one row per cluster with text, returns, labels,
  price-quality fields, contamination flags, and chronological split labels.
- `cluster_articles.parquet`: auditable article-to-cluster membership.
- `build_audit.json`: filters, counts, input paths, settings, missing quote
  partitions, and output counts.
- `.minute_cache/`: restartable per-day quote-minute Parquet files.

With `--write-csv`, CSV copies of the two Parquet datasets are also created.
Parquet is strongly preferred for the training cluster.

## Label definition

For horizon `h`, the return threshold is:

```text
max(
    z_cutoff * pre_event_one_minute_volatility * sqrt(h),
    spread_multiple * decision_spread_bps / 10,000
)
```

The label is `up` above the positive threshold, `down` below the negative
threshold, and `neutral` inside the band. Threshold inputs use only information
available before the outcome. Continuous raw, benchmark, label, maximum, and
minimum returns are retained so a different labeling rule can be applied later
without rebuilding the merge.

# Hourly TF-IDF stock-movement experiment

This experiment tests whether newly available Nvidia-related news text helps
predict NVDA's next complete regular-session hourly bar. It is separate from
the previous-day daily model in `tf-idf/`.

## Observation timing

One observation is one complete one-hour provider bar. At the start of a target
bar, the text document contains retained news published from the preceding bar
start (inclusive) to the target bar start (exclusive). Consequently:

- an article is assigned to no more than one prediction document;
- an article timestamped at the target cutoff is not used at that cutoff;
- news after the final bar start and overnight is assigned to the next market
  open; and
- every article in a document precedes the return it is used to predict.

Only non-excluded `direct` and `indirect` event representatives are used. Every
UTC date touched by a document window must have successful collection coverage
for every configured ticker.

The provider's final bar in each session is only 30 minutes long, so it is not a
target. Documented missing-price bars and rows without complete lagged controls
are also excluded.

## Targets and models

The primary target is hourly direction:

- `-1`: target bar open-to-close return below zero;
- `1`: target bar open-to-close return above zero.

The secondary target is a fixed 0.5% movement band:

- `-1`: return at or below -0.5%;
- `0`: return strictly between -0.5% and +0.5%;
- `1`: return at or above +0.5%.

For each target, the experiment compares:

1. a training-class-prior baseline;
2. lagged market controls plus news volume;
3. TF-IDF only; and
4. market controls, news volume, and TF-IDF.

The market/volume features are the preceding bar return, six-bar trailing
volatility, preceding volume, current time of day, and the number of newly
available articles. Text uses word unigrams and bigrams with at most 500
features. Tokens must contain at least one letter, preventing years and bare
price figures from acting as unstable time proxies while retaining mixed tokens
such as `H100`. All logistic regressions use class-balanced L2 regularization.

## Leakage controls and evaluation

The last 30% of trading dates—not individual rows—is held out. Earlier
expanding-window folds also split on complete dates and select the regularization
strength. Vectorizers, inverse-document frequencies, scalers, and models are fit
inside each training fold. Held-out uncertainty intervals resample complete
trading-day blocks. Because regularization is selected using macro-F1 and the
models use class balancing, classification balance is the priority; raw
probabilities may not be well calibrated.

This workflow measures an exploratory association. It is not a sentiment model,
a causal estimate, or a trading strategy.

## Run

From the repository root:

```sh
.venv/bin/python -m pip install -r tf-idf-hourly/requirements.txt
.venv/bin/python tf-idf-hourly/hourly_tfidf.py
```

Run `--help` for options controlling the movement band, date split, folds,
regularization grid, TF-IDF settings, volatility window, bootstrap count, and
output directory.

## Outputs

Files are written to `tf-idf-hourly/output/`:

- `hourly_documents.csv`: eligible bars, lagged controls, targets, and text;
- `held_out_predictions.csv`: held-out probabilities and predictions;
- `fold_metrics.csv`: chronological validation scores for model selection;
- `confusion_matrix.csv`: held-out confusion matrices;
- `top_terms_by_class.csv`: strongest training-sample term associations; and
- `model_metrics.json`: source hashes, audit counts, settings, held-out metrics,
  trading-day bootstrap intervals, and limitations.

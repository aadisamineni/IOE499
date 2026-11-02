# Previous-day TF-IDF stock-direction experiment

This experiment tests whether words in the prior New York calendar day's news
help predict NVDA's next adjusted close-to-close movement class:

- `-1`: return at or below -2%;
- `0`: return strictly between -2% and +2%;
- `1`: return at or above +2%.

It is a stock-movement model, not a labeled sentiment model. TF-IDF identifies
terms that help distinguish return classes in this sample; it does not determine
whether an article is linguistically positive or negative.

## Leakage controls and unit of analysis

The unit of analysis is one trading session. The script reads the audited
`matched_one_day` outputs and concatenates the headlines and summaries of
eligible articles into one document per session. Only existing `direct` and
`indirect` duplicate-group representatives are retained so syndicated copies do
not dominate the term weights.

Sessions outside the complete configured news-collection range are excluded.
The data is kept in chronological order. The last 30% is held out, and earlier
expanding-window folds select regularization strength. Every TF-IDF vocabulary
and inverse-document-frequency calculation is fitted only on the corresponding
training data.

## Models

The script compares:

1. Training-class-prior baseline;
2. market controls only;
3. TF-IDF only;
4. market controls plus TF-IDF.

Market controls are the lagged daily return, lagged 20-session volatility, and
log-transformed representative article count. Logistic models use class-balanced
L2 regularization. TF-IDF uses lowercased word unigrams and bigrams, English stop
words, `min_df=3`, `max_df=0.90`, and at most 500 features.

## Run

From the repository root:

```sh
.venv/bin/python -m pip install -r tf-idf/requirements.txt
.venv/bin/python tf-idf/tfidf_stock_direction.py
```

The train fraction, CV fold count, regularization grid, document-frequency
limits, feature cap, and output directory can be changed with command-line
arguments. Run `--help` for details.

## Outputs

Files are written to `tf-idf/output/`:

- `session_documents.csv`: one document and target per eligible session;
- `session_predictions.csv`: held-out probabilities and predictions;
- `model_metrics.json`: configuration, source hashes, selected settings,
  held-out metrics, and limitations;
- `fold_metrics.csv`: chronological validation results for every model and
  regularization candidate;
- `confusion_matrix.csv`: held-out confusion matrices;
- `top_terms_by_class.csv`: strongest positive and negative TF-IDF coefficients
  for each class.

Balanced accuracy and macro F1 are emphasized because the neutral class is much
larger than the up and down classes. Results are exploratory and should not be
interpreted as a trading signal or evidence that news caused a price movement.

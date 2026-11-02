# Article content review against earlier saved coverage

The review reads the full **saved headline and summary** for each assessed article,
identifies the substantive reported fact, and compares it with preceding saved
articles. It uses no web searches, similarity-derived labels, price/tick outcomes,
or imported judgments from the earlier provisional review. Unavailable full article
bodies are not inferred. The source collection is unchanged.

## Files

- **`new_articles.csv`**: the requested list of new-information candidates and
  mixed articles with an identified new fact. Read `new_information` to see exactly
  what is considered new; do not treat the entire mixed article as new.
- **`classified_articles.csv`**: every completed content assessment, including
  original fields/text, label, claim, supporting current passage, explanation,
  earlier article IDs and limitations.
- **`earlier_article_evidence.csv`**: the current article/claim, earlier article
  ID/headline/time/source/URL, an exact supporting passage, and why it matters.
  `repeated_fact` supports a recap judgment. `prior_event_context` supports the
  older part of a mixed article; it does not establish the new detail was old.
  `possible_context_only` is explicitly not proof of factual duplication.
- **`progress.json`**: exact assessed/pending counts and the next article.
- `review_status.csv`: status for all 53,073 articles. Pending articles are explicitly
  unreviewed and are absent from the completed-classification and new-article lists.
- `assessments/*.json`: the explicit article-by-article judgments and evidence.

The existing `relevance_type` and `excluded` fields are preserved. This review covers
all saved articles; the new list therefore includes peripheral company releases,
research announcements and other-company news as well as NVIDIA stories. A new fact
does not establish NVIDIA relevance, materiality, a price effect, or a trading signal.

## Labels and evidence

| Label | Meaning |
|---|---|
| `new_information_candidate` | A specific factual update, disclosure, report finding, or attributable statement is not established in the earlier saved coverage reviewed. |
| `old_news_recap` | The substantive event is repeated from an earlier saved article, or the target explicitly recaps a previously dated announcement without an identifiable update. |
| `price_reaction` | Mainly describes/explains an observed price move or market session; no specific incremental business fact is established in the saved text. |
| `opinion_background` | Mainly recommendations, forecasts, valuation, strategy or background; no distinct new factual disclosure is identified. |
| `mixed` | Includes a specifically identified new fact alongside older-event coverage or observed-price reporting. |
| `insufficient_information` | Saved text is too incomplete, mismatched or unclear for a reliable substantive decision. |

New articles state the actual candidate fact rather than merely saying their wording
is different. Comparators must have a **strictly earlier saved publication time**;
same-time reports cannot prove one came first. Quotes in the evidence CSV are checked
against the cited original headline or summary. No earlier evidence is invented
when none is found.

Self-dated historical recaps have `evidence_basis` explaining that their own text
dates the announcement earlier; they may have no earlier-article evidence row because
the collection does not contain that older period. An old underlying event can still
be newly disclosed: this is considered separately from an explicit announcement recap.

The collection begins **October 1, 2025**. Initial-day candidates have almost no
historical coverage to check. They are **new within available dataset coverage**,
not verified first public releases. Saved text may also contain later revisions;
publication-history authenticity remains unverified. Earlier provisional assessments
remain in their original folder and are not counted as completed under this review.

`save_review.py` only validates and exports the recorded judgments. It does **not**
classify articles or decide novelty. To regenerate the CSVs after explicit assessments:

```sh
python3 research/news_novelty_review/dataset_content_review/save_review.py
```

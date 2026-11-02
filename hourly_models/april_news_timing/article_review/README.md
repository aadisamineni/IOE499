# Provisional content review: April's top ten hourly moves

Run from repository root:

```sh
.venv/bin/python hourly_models/april_news_timing/article_review/build_review.py
```

Requires pandas from the existing project requirements. Source: ../event_article_matches.csv, limited to ranks 1–10 and the equal-hour before/during/after windows. Eighty distinct article IDs produce 92 event–article pairs. Existing exact-copy filtering was performed in the upstream timing analysis. Retain distinct coverage; an article can belong to more than one timing group when compared with adjacent price bars.

## Review rubric

These are provisional assistant assessments of locally saved headlines and summaries, explicitly authorized for this exploratory review. No external article retrieval or additional LLM API call was used. These labels are not independent human consensus and should not yet become validated training labels.

Each flag is assessed independently and categories overlap:

- concrete_development_reported: identifies a specific business, regulatory, product, economic, or other development. Does NOT mean first announcement, new information at publication, confirmed rumor, or material relevance to NVDA. Recounting an earlier announcement still qualifies.
- existing_price_movement_any_company: explicitly describes an already observed share-price movement, performance, or market-cap milestone, including historical performance. A price level alone or predicted breakout does not suffice.
- existing_nvda_price_movement: the observed movement explicitly concerns Nvidia; subset of any-company movement. Does not establish that it describes our selected hourly bar or has the same direction.
- opinion_or_analysis: includes an investment recommendation, valuation, technical setup, or interpretation. Can coexist with factual announcements and price reporting.
- insufficient_or_ambiguous_content: missing or vague text, or uncertain interpretation, limits classification. Other supported flags can still be true.

False means no clear affirmative evidence in the available text, not a verified absence in the full article. Supporting headline and summary preserve all available textual evidence; assessment_note explains the decision. Text-only assessments do not prove event occurrence. development_first_release_time distinguishes explicit earlier-development wording in selected cases from unverified timing. No record's original release time or historical article version was independently verified. timestamp_or_revision_warning flags live coverage and specific inconsistent-looking metadata; unflagged timestamps are not necessarily reliable.

## Findings and limits

Explicit NVDA price reporting: before 5/40 (12.5%), during 8/26 (30.8%), after 13/26 (50%). Concrete-development reports: 28/40, 17/26, 12/26. Both development and NVDA movement: 2, 5, 5 respectively. Categories overlap and group membership overlaps, so neither totals nor inferential tests should assume independent samples.

The pattern is descriptive and conditional on selecting the largest observed moves. It is not evidence of predictive power or causation. Before articles may describe premarket, previous-day, or prior-week movements. General price commentary is not necessarily commentary on the selected bar. Some reports focus on other companies; retaining existing direct/indirect relevance is not a new relevance audit. Further semantic and publication-history review is needed before analysis across all hours.

## Files and independent review

- provisional_article_review.csv: one record per article, source text/link/timestamp, overlapping flags, evidence, rationale and warnings.
- reviewed_event_article_pairs.csv: every selected event–article link with labels attached, retaining phase and exact timing.
- group_summary.csv: distinct article counts by timing group and flag; overlaps allowed.
- report.html: group table plus all reviewed articles and explanations.
- independent_review_ZAS.csv and independent_review_AVS.csv: shuffled, unlabeled source records. Enter yes/no/unclear for each flag, provide a supporting excerpt and notes. Review independently before opening the provisional labels or discussing disagreements. Labels, event returns and phases are withheld; publication timestamps remain visible. Existing review sheets are never overwritten by reruns.
- metadata.json: input hash, scope, label origin, validation and summary counts.

After both independent sheets are completed, compare each flag, preserve disagreements, and adjudicate with explicit reasons. No agreement statistic or human consensus is claimed yet.

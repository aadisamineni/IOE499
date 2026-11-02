# News novelty pilot

Completed 2026-10-26. Open `report.html` for a readable article-by-article review or `review.csv` in a spreadsheet.

## What was done

Selected 50 articles uniformly without replacement (seed 499) from 2,038 direct-NVDA, non-excluded, representative regular-session articles. Eligibility used `prepare_news` from origin/AVS c99cb7a with its default October 1, 2025–May 1, 2026 range. This samples articles, not clusters; the results cannot be treated as the exact composition of the ARC training set. Source text and input SHA-256 are preserved in `sample.csv` and `sample_metadata.json`.

The assistant read all 50 saved headlines/summaries and performed targeted external searches covering 23 sampled records, primarily announcement candidates, apparent recaps and timestamp conflicts. Some evidence is search-visible publication metadata; not every original full article was retrievable. The remaining 27 records have content-only judgments and explicitly lack independently verified publication histories. No subsequent return files were read during this review. Prices described within the articles were visible and used to identify reactive reporting.

## Provisional results

| Main assessment | Articles |
|---|---:|
| Opinion/background | 14 |
| Recap | 14 |
| Reactive reporting | 13 |
| Announcement candidate | 6 |
| Unclear | 3 |

These are mutually exclusive main-content judgments; actual articles can mix categories. No sampled article was established as a verified first release. Six candidates match concrete developments but first availability remains unresolved. This is not evidence that none was new. Confidence describes confidence in the stated content assessment, not a probability of novelty or price movement. New opinion, newly released research, fresh analyst actions and other-company news can themselves be information; these labels address the narrower NVDA business-announcement experiment and should not be used to automatically discard every noncandidate.

## Important examples

- Rows 17–18: January 6 coverage of Supermicro and CoreWeave developments for which company releases are dated January 5.
- Row 20: January 8 article whose saved summary dates the analyst action December 19.
- Row 45: March 27 HBM4 coverage, versus Micron's March 16 announcement.
- Row 46: April 1 Marvell partnership coverage, versus NVIDIA's March 31 announcement.
- Rows 31–32: saved February 25 times of 17:33:58 and 18:15 UTC for CFO headlines, versus the company's scheduled conference call at 22:00 UTC. This is a provenance conflict, not proof of a specific feed bug. Verify original wire history and article versions before assigning a corrected time. The call schedule alone does not prove that every detail was first released at the call.
- Row 5: Qualcomm's current page displays October 28 despite October 27 sample/reporting. Do not silently force source dates to agree.

Source URLs and date precision/caveats appear in each reviewed row. Earliest located evidence means a source found during this bounded search, not the guaranteed first public release. Exact delays should only be calculated where both timestamps have verified offsets and historical meaning.

## Next use

1. Have a team member label `independent_review.csv` before reading assistant assessments. It contains no subsequent outcomes and no assistant decisions. Resolve disagreements with evidence; preserve both judgments.
2. Investigate the two CFO timestamp conflicts in raw feed records and original wire publication history. Do not alter source data or replace timestamps with guessed release times.
3. Establish usable original-release timestamps for the six announcement candidates. Add first-source URLs and precise offset-aware times only when supported.
4. Build a separate first-release event study, maintaining the distinction between an original announcement and a later tradable article. Keep the existing delayed-prediction experiment separately defined.

These are assistant-generated exploratory assessments, not independently human-validated training labels. This pilot does not establish price reaction, forecasting performance or causality. No existing filtering rules or datasets were changed.

## Files

- `sample.csv`: unchanged sampled source fields.
- `sample_metadata.json`: selection and source provenance.
- `review.csv`: 50 assessments, rationales, external sources and warnings.
- `report.html`: readable report with source links.
- `independent_review.csv`: blank human-review sheet; report reruns preserve it.
- `summary.json`: aggregate counts and limitations flags.
- `build_report.py`: renders the explicit assessments; does not query an LLM or perform web searches automatically.

Regenerate rendered outputs with:

```sh
.venv/bin/python research/news_novelty_review/build_report.py
```

## Prompt for a separate ChatGPT review

Provide the sampled CSV and use a chat with browsing available:

> Review these articles for information novelty at the saved publication time. Do not inspect subsequent returns or use remembered price performance. For each article, identify the concrete development, distinguish business announcements, new details, recaps, price-reactive reporting and opinion/background. Search earlier reporting and original company/regulatory sources for factual developments. Record supporting URLs, the earliest source you actually find, its timestamp and timezone or date-only precision, and any revision conflicts. Never equate an event date with public disclosure time. Never label an article a confirmed first release merely because no earlier source appears in your search. Use unclear where text or timing is insufficient. Preserve mixed content and weak company relevance in notes. Return one row per article with evidence, confidence in the assessment, and explicit verification limits. These are provisional assistant labels requiring independent human review.

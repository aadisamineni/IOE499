# Theme Analysis

This folder extracts interpretable news themes from the matched NVDA event/article data.

The workflow:

1. Reads `data/matched/2025-09-15_2026-05-15/event_article_matches.csv`.
2. Removes excluded rows and exact duplicate article-event matches.
3. Builds TF-IDF features from each article's headline and summary.
4. Uses NMF to discover recurring topics.
5. Compares topic prevalence across 1% up and 1% down event windows.

Run from the project root:

```bash
python3 Theme_analysis/find_themes.py
```

Outputs are written to `Theme_analysis/output/`:

- `topic_words.csv`: top words and weights for each discovered topic.
- `article_topic_assignments.csv`: dominant topic and topic weights per article-event pair.
- `theme_event_summary.csv`: topic prevalence by event direction.
- `category_event_summary.csv`: existing rule-based category prevalence by event direction.

The topics are exploratory associations, not evidence that a theme caused a stock movement. Because an article can be matched to more than one event window, interpretation should focus on event-level prevalence and should eventually be checked against non-event control days.

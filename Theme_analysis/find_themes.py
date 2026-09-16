"""Discover news themes in matched NVDA event/article data using TF-IDF + NMF."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/matched/2025-09-15_2026-05-15/event_article_matches.csv"
OUTPUT = Path(__file__).resolve().parent / "output"
N_TOPICS = 8


def main():
    OUTPUT.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT, low_memory=False)
    df = df[df["excluded"].astype(str).str.lower().eq("false")].copy()
    df = df.drop_duplicates(subset=["event_date", "article_id"])
    df["direction"] = np.select(
        [df["shift_1pct_up"].eq(1), df["shift_1pct_down"].eq(1)],
        ["up", "down"], default="other"
    )
    df = df[df["direction"].isin(["up", "down"])].copy()
    df["text"] = (df["headline"].fillna("") + " " + df["summary"].fillna(""))
    df = df[df["text"].str.strip().ne("")].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=5,
        max_df=0.90, sublinear_tf=True, max_features=5000
    )
    X = vectorizer.fit_transform(df["text"])
    n_topics = min(N_TOPICS, max(2, min(X.shape[0] - 1, X.shape[1] - 1)))
    model = NMF(n_components=n_topics, init="nndsvda", random_state=42,
                max_iter=500)
    W = model.fit_transform(X)
    terms = vectorizer.get_feature_names_out()

    topic_rows = []
    for topic_id, weights in enumerate(model.components_):
        top_idx = weights.argsort()[::-1][:15]
        for rank, idx in enumerate(top_idx, 1):
            topic_rows.append({"topic": topic_id, "rank": rank,
                               "word": terms[idx], "weight": weights[idx]})
    pd.DataFrame(topic_rows).to_csv(OUTPUT / "topic_words.csv", index=False)

    for topic_id in range(n_topics):
        df[f"topic_{topic_id}_weight"] = W[:, topic_id]
    df["dominant_topic"] = W.argmax(axis=1)
    keep = ["event_date", "direction", "article_id", "headline", "event_category", "dominant_topic"]
    keep += [f"topic_{i}_weight" for i in range(n_topics)]
    df[keep].to_csv(OUTPUT / "article_topic_assignments.csv", index=False)

    event_topic = df.groupby(["event_date", "direction", "dominant_topic"]).size().rename("article_count").reset_index()
    event_topic["event_article_share"] = event_topic["article_count"] / event_topic.groupby("event_date")["article_count"].transform("sum")
    summary = event_topic.groupby(["direction", "dominant_topic"]).agg(
        events_with_topic=("event_date", "nunique"),
        mean_article_share=("event_article_share", "mean"),
        total_article_matches=("article_count", "sum")
    ).reset_index()
    summary.to_csv(OUTPUT / "theme_event_summary.csv", index=False)

    cats = df.groupby(["event_date", "direction", "event_category"]).size().rename("article_count").reset_index()
    cats["event_article_share"] = cats["article_count"] / cats.groupby("event_date")["article_count"].transform("sum")
    cats.groupby(["direction", "event_category"]).agg(
        events_with_category=("event_date", "nunique"),
        mean_article_share=("event_article_share", "mean"),
        total_article_matches=("article_count", "sum")
    ).reset_index().to_csv(OUTPUT / "category_event_summary.csv", index=False)
    print(f"Analyzed {len(df):,} article-event rows across {df['event_date'].nunique():,} event dates")
    print(f"Wrote {n_topics} topics to {OUTPUT}")


if __name__ == "__main__":
    main()

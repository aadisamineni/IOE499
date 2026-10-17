"""Deterministic tests for the event-level TF-IDF training workflow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_tfidf_arc import (  # noqa: E402
    aligned_probabilities,
    prepare_training_frame,
    tune_regularization,
)


class EventTfidfTests(unittest.TestCase):
    @staticmethod
    def event_frame() -> pd.DataFrame:
        rows = []
        tokens = {-1: "decline weak sell", 0: "steady unchanged neutral", 1: "rally strong buy"}
        index = 0
        for split_index, split in enumerate(("train", "validation", "test")):
            for repetition in range(6):
                for label in (-1, 0, 1):
                    index += 1
                    rows.append(
                        {
                            "news_cluster_id": f"cluster-{index}",
                            "session_date": f"2026-0{split_index + 1}-{repetition + 1:02d}",
                            "cluster_start_utc": (
                                f"2026-0{split_index + 1}-{repetition + 1:02d}T15:00:00Z"
                            ),
                            "combined_text": f"{tokens[label]} report {repetition}",
                            "chronological_split": split,
                            "label_code_5m": label,
                            "label_available_5m": True,
                            "clean_5m": True,
                        }
                    )
        rows.append(
            {
                "news_cluster_id": "dirty",
                "session_date": "2026-03-07",
                "cluster_start_utc": "2026-03-07T15:00:00Z",
                "combined_text": "later news contamination",
                "chronological_split": "test",
                "label_code_5m": 1,
                "label_available_5m": True,
                "clean_5m": False,
            }
        )
        return pd.DataFrame(rows)

    def test_clean_filter_and_split_order(self) -> None:
        frame, audit = prepare_training_frame(
            self.event_frame(), horizon=5, include_contaminated=False
        )

        self.assertEqual(len(frame), 54)
        self.assertEqual(audit["contaminated_events_removed"], 1)
        self.assertNotIn("dirty", frame.news_cluster_id.tolist())
        self.assertEqual(
            frame.chronological_split.drop_duplicates().tolist(),
            ["train", "validation", "test"],
        )

    def test_regularization_selection_and_probability_alignment(self) -> None:
        frame, _ = prepare_training_frame(
            self.event_frame(), horizon=5, include_contaminated=False
        )
        train = frame.loc[frame.chronological_split.eq("train")]
        validation = frame.loc[frame.chronological_split.eq("validation")]

        selected_c, metrics = tune_regularization(
            train,
            validation,
            c_grid=[0.1, 1.0],
            min_df=1,
            max_df=1.0,
            max_features=100,
            ngram_max=2,
        )

        self.assertIn(selected_c, {0.1, 1.0})
        self.assertEqual(len(metrics), 2)
        self.assertTrue(np.isfinite(metrics.log_loss).all())
        self.assertTrue(metrics.macro_f1.between(0, 1).all())

        # The helper used by the trainer always emits down/neutral/up columns.
        from train_tfidf_arc import make_model

        model = make_model(selected_c, 1, 1.0, 100, 2)
        model.fit(train.combined_text, train.target_label)
        probabilities = aligned_probabilities(model, validation.combined_text)
        self.assertEqual(probabilities.shape, (len(validation), 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()

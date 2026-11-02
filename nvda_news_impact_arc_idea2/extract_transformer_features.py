#!/usr/bin/env python3
"""Extract frozen FinBERT sentiment probabilities and document embeddings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT, atomic_json, atomic_parquet, file_sha256


def resolve_device(requested: str, torch_module: object) -> str:
    torch = torch_module
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_features(
    events_path: Path,
    output_path: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    device_request: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer dependencies are not installed. Install "
            "requirements-transformers.txt in the HPC environment."
        ) from exc

    events = pd.read_parquet(events_path)
    required = {"event_id", "combined_text"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Event dataset is missing columns: {sorted(missing)}")
    if events.event_id.astype(str).duplicated().any():
        raise ValueError("Event identifiers must be unique")

    device = resolve_device(device_request, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()

    raw_mapping = getattr(model.config, "id2label", {})
    label_mapping = {
        int(index): str(label).strip().lower()
        for index, label in raw_mapping.items()
    }
    recognized = {"positive", "negative", "neutral"}
    if not recognized.issubset(set(label_mapping.values())):
        raise ValueError(
            f"Model labels must include positive/negative/neutral; found {label_mapping}"
        )

    texts = events.combined_text.fillna("").astype(str).tolist()
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, return_dict=True)
            batch_probabilities = torch.softmax(output.logits, dim=-1)
            hidden = output.hidden_states[-1]
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        probabilities.append(batch_probabilities.detach().cpu().numpy().astype("float32"))
        embeddings.append(pooled.detach().cpu().numpy().astype("float32"))
        print(
            f"FinBERT {min(start + batch_size, len(texts))}/{len(texts)}",
            flush=True,
        )

    probability_array = np.vstack(probabilities)
    embedding_array = np.vstack(embeddings)
    output = pd.DataFrame({"event_id": events.event_id.astype(str)})
    for index, name in label_mapping.items():
        output[f"finbert_{name}"] = probability_array[:, index]
    for index in range(embedding_array.shape[1]):
        output[f"embedding_{index:04d}"] = embedding_array[:, index]
    atomic_parquet(output, output_path)

    audit: dict[str, object] = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(events_path.resolve()),
            "sha256": file_sha256(events_path),
            "events": len(events),
        },
        "model": {
            "name": model_name,
            "resolved_commit": getattr(model.config, "_commit_hash", None),
            "labels": label_mapping,
            "max_length": max_length,
            "pooling": "attention-mask-weighted mean of final hidden state",
            "fine_tuned_on_project_returns": False,
        },
        "runtime": {
            "device": device,
            "batch_size": batch_size,
            "torch": torch.__version__,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": file_sha256(output_path),
            "rows": len(output),
            "embedding_dimensions": embedding_array.shape[1],
        },
        "look_ahead_policy": (
            "The frozen model receives only each supplied headline/summary. The selected "
            "model should have a training cutoff before the evaluated news period."
        ),
    }
    atomic_json(audit, output_path.with_suffix(".audit.json"))
    return output, audit


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_OUTPUT / "news_events.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT / "transformer_features.parquet",
    )
    parser.add_argument("--model", default="ProsusAI/finbert")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if not args.events.is_file():
        parser.error(f"Event dataset does not exist: {args.events}")
    if args.batch_size < 1 or args.max_length < 8:
        parser.error("Batch size must be positive and max length must be at least 8")
    _, audit = extract_features(
        args.events,
        args.output,
        args.model,
        args.batch_size,
        args.max_length,
        args.device,
    )
    print(json.dumps({"status": "complete", **audit["output"]}, indent=2))


if __name__ == "__main__":
    main()

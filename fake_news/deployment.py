"""Deployment helper to classify new data using a trained model."""

from __future__ import annotations

from typing import Iterable, List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class FakeNewsClassifier:
    """Loads a trained model and performs inference."""

    def __init__(self, model_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def classify(self, texts: Iterable[str]) -> List[dict]:
        """Return classification results for the provided texts."""
        return list(self.pipe(list(texts)))

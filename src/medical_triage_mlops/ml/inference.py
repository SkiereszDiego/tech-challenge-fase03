from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from medical_triage_mlops.ml.train import load_model


@dataclass
class PredictionResult:
    label: str
    confidence: float
    scores: dict[str, float]


class Predictor(Protocol):
    def predict(self, texts: list[str]) -> list[PredictionResult]: ...


class SklearnPredictor:
    def __init__(self, model_path: Path) -> None:
        self._pipeline = load_model(model_path)

    def predict(self, texts: list[str]) -> list[PredictionResult]:
        classes = list(self._pipeline.classes_)
        probabilities = self._pipeline.predict_proba(texts)
        return _build_results(classes, probabilities)


def _build_results(classes: list[str], probabilities: np.ndarray) -> list[PredictionResult]:
    results = []
    for row in probabilities:
        scores = {label: float(score) for label, score in zip(classes, row, strict=True)}
        best_label = max(scores, key=scores.get)
        results.append(
            PredictionResult(label=best_label, confidence=scores[best_label], scores=scores)
        )
    return results


def get_predictor(model_path: Path) -> Predictor:
    return SklearnPredictor(model_path)

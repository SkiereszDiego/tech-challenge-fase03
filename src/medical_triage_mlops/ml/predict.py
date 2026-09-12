"""Load the trained pipeline and run predictions on new report text."""

from functools import lru_cache
from pathlib import Path

import joblib

from medical_triage_mlops.core.config import get_settings
from medical_triage_mlops.ml.dataset import load_label_map


class ModelNotFoundError(RuntimeError):
    """Raised when the model artifact hasn't been trained/saved yet."""


@lru_cache
def _load_pipeline(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Modelo não encontrado em '{path}'. "
            "Rode 'uv run python -m medical_triage_mlops.ml.train' primeiro."
        )
    return joblib.load(path)


@lru_cache
def _load_labels() -> dict[int, str]:
    return load_label_map()


def classify_report(text: str) -> dict:
    """Classify a medical report's text and return the predicted category.

    Returns a dict with the predicted class id, label name and the
    per-class probability distribution.
    """
    settings = get_settings()
    pipeline = _load_pipeline(settings.model_path)
    labels = _load_labels()

    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    class_index = int(proba.argmax())
    predicted_class = int(classes[class_index])

    return {
        "predicted_class": predicted_class,
        "predicted_label": labels[predicted_class],
        "probabilities": {
            labels[int(cls)]: float(prob) for cls, prob in zip(classes, proba, strict=True)
        },
    }

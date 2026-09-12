"""Train the baseline medical report classifier.

Pipeline: TF-IDF vectorizer + Logistic Regression.
Usage:
    uv run python -m medical_triage_mlops.ml.train
"""

import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from medical_triage_mlops.ml.dataset import (
    load_label_map,
    load_test_dataset,
    load_train_dataset,
    save_processed,
)

MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/latency/train_metrics.json")


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=5.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train() -> None:
    """Load data, train the baseline model, evaluate it and persist artifacts."""
    print("Carregando dados...")
    train_ds = load_train_dataset()
    test_ds = load_test_dataset()
    save_processed(train_ds, "train.csv")
    save_processed(test_ds, "test.csv")

    label_map = load_label_map()
    print(f"Treino: {len(train_ds.texts)} amostras | Teste: {len(test_ds.texts)} amostras")
    print(f"Classes: {label_map}")

    pipeline = build_pipeline()

    print("Treinando modelo (TF-IDF + Logistic Regression)...")
    start = time.perf_counter()
    pipeline.fit(train_ds.texts, train_ds.labels)
    train_seconds = time.perf_counter() - start

    predictions = pipeline.predict(test_ds.texts)
    report = classification_report(
        test_ds.labels,
        predictions,
        target_names=[label_map[i] for i in sorted(label_map)],
        output_dict=True,
        zero_division=0,
    )
    print(
        classification_report(
            test_ds.labels,
            predictions,
            target_names=[label_map[i] for i in sorted(label_map)],
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "train_seconds": train_seconds,
                "n_train": len(train_ds.texts),
                "n_test": len(test_ds.texts),
                "accuracy": report["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "per_class": {label_map[i]: report[label_map[i]] for i in sorted(label_map)},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Métricas salvas em {METRICS_PATH}")


if __name__ == "__main__":
    train()

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from medical_triage_mlops.ml.data import LABEL_COLUMN, TEXT_COLUMN


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=20_000, stop_words="english", ngram_range=(1, 1)),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train_model(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[Pipeline, dict[str, Any]]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN],
        df[LABEL_COLUMN],
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL_COLUMN],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return pipeline, metrics


def save_model(pipeline: Pipeline, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(Path(path))


def main() -> None:
    from medical_triage_mlops.core.config import get_settings

    settings = get_settings()
    df = pd.read_csv("data/processed/dataset.csv")

    pipeline, metrics = train_model(df)
    save_model(pipeline, Path(settings.model_path))
    print("Modelo treinado e salvo em", settings.model_path)
    print("Métricas:", metrics)


if __name__ == "__main__":
    main()

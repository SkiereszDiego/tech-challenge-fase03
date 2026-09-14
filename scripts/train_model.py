"""Treina o classificador de urgência e salva o modelo.

Uso:
    uv run python scripts/train_model.py
"""

from pathlib import Path

import pandas as pd

from medical_triage_mlops.core.config import get_settings
from medical_triage_mlops.ml.train import save_model, train_model

PROCESSED_DATASET = Path("data/processed/dataset.csv")


def main() -> None:
    settings = get_settings()

    print(f"Lendo dataset processado de {PROCESSED_DATASET}...")
    df = pd.read_csv(PROCESSED_DATASET)

    print("Treinando o pipeline (TF-IDF + LogisticRegression)...")
    pipeline, metrics = train_model(df)
    print("Relatório de classificação (conjunto de validação):")
    for label, values in metrics.items():
        print(f"  {label}: {values}")

    save_model(pipeline, Path(settings.model_path))
    print(f"Modelo salvo em {settings.model_path}")


if __name__ == "__main__":
    main()

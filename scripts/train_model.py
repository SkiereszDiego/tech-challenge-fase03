"""Treina o classificador de urgência, salva o modelo e exporta uma versão ONNX.

Uso:
    uv run python scripts/train_model.py
"""

from pathlib import Path

import pandas as pd

from medical_triage_mlops.core.config import get_settings
from medical_triage_mlops.ml.onnx_convert import convert_pipeline_to_onnx
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

    convert_pipeline_to_onnx(pipeline, Path(settings.onnx_model_path))
    print(f"Modelo ONNX salvo em {settings.onnx_model_path}")


if __name__ == "__main__":
    main()

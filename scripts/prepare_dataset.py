"""Baixa o dataset bruto e monta o dataset processado (com rótulo de urgência).

Uso:
    uv run python scripts/prepare_dataset.py
"""

from pathlib import Path

from medical_triage_mlops.ml.data import build_processed_dataset, download_raw_dataset

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    print(f"Baixando dataset do Kaggle para {RAW_DIR}...")
    download_raw_dataset(RAW_DIR)

    print("Mapeando categorias para níveis de urgência...")
    df = build_processed_dataset(RAW_DIR, PROCESSED_DIR)

    print(f"Dataset processado salvo em {PROCESSED_DIR / 'dataset.csv'} ({len(df)} amostras).")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()

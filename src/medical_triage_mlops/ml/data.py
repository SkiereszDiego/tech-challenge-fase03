import shutil
from pathlib import Path

import pandas as pd

from medical_triage_mlops.core.labels import CONDITION_TO_URGENCY

KAGGLE_DATASET = "saharalaa/medical-abstracts-tc-corpus"

RAW_TRAIN_FILE = "medical_tc_train.csv"
RAW_TEST_FILE = "medical_tc_test.csv"

TEXT_COLUMN = "text"
CONDITION_COLUMN = "condition_label"
LABEL_COLUMN = "label"


def download_raw_dataset(dest_dir: Path) -> Path:
    """Baixa o Medical Abstracts TC Corpus via kagglehub para dest_dir."""
    import kagglehub

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    for filename in (RAW_TRAIN_FILE, RAW_TEST_FILE):
        # shutil.copyfile (não copy/copy2): esses também copiam permissões/metadados,
        # o que gera PermissionError em alguns bind mounts do Docker (ex.: Docker
        # Desktop no Windows). copyfile copia só o conteúdo do arquivo.
        shutil.copyfile(cache_path / filename, dest_dir / filename)

    return dest_dir


def load_raw_dataset(raw_dir: Path) -> pd.DataFrame:
    """Lê os CSVs de treino e teste de raw_dir e os concatena."""
    raw_dir = Path(raw_dir)
    train_df = pd.read_csv(raw_dir / RAW_TRAIN_FILE)
    test_df = pd.read_csv(raw_dir / RAW_TEST_FILE)

    df = pd.concat([train_df, test_df], ignore_index=True)
    return df.rename(columns={"medical_abstract": TEXT_COLUMN})[[TEXT_COLUMN, CONDITION_COLUMN]]


def map_to_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia os rótulos de condição para os níveis de urgência definidos em CONDITION_TO_URGENCY.
    Raises ValueError se algum rótulo de condição não puder ser mapeado.
    """
    mapped = df.copy()
    mapped[LABEL_COLUMN] = mapped[CONDITION_COLUMN].map(CONDITION_TO_URGENCY)

    unmapped = mapped[LABEL_COLUMN].isna()
    if unmapped.any():
        bad_values = sorted(mapped.loc[unmapped, CONDITION_COLUMN].unique())
        raise ValueError(f"Valores de condition_label sem mapeamento: {bad_values}")

    return mapped[[TEXT_COLUMN, LABEL_COLUMN]]


def build_processed_dataset(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Carrega os dados brutos, mapeia para os rótulos de urgência e salva o dataset processado."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = map_to_urgency(load_raw_dataset(raw_dir))
    df.to_csv(processed_dir / "dataset.csv", index=False)
    return df

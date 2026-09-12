"""Loading and preparation helpers for the Medical Abstracts TC Corpus.

Dataset: sebischair/Medical-Abstracts-TC-Corpus
https://github.com/sebischair/Medical-Abstracts-TC-Corpus

Each row has a medical abstract (`medical_abstract`) and a numeric label
(`condition_label`, 1-5) identifying the clinical condition category the
abstract describes. For this project we use `condition_label` as the
classification target: the API classifies an incoming report into one of
the five clinical categories below. This is an academic stand-in for a
real urgency-triage label (normal / atenção / urgente), which the source
dataset does not provide — see README for the discussion of this choice.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRAIN_FILE = RAW_DIR / "medical_tc_train.csv"
TEST_FILE = RAW_DIR / "medical_tc_test.csv"
LABELS_FILE = RAW_DIR / "medical_tc_labels.csv"

TEXT_COLUMN = "medical_abstract"
LABEL_COLUMN = "condition_label"


@dataclass(frozen=True)
class Dataset:
    """A text classification dataset split."""

    texts: pd.Series
    labels: pd.Series


def load_label_map(labels_file: Path = LABELS_FILE) -> dict[int, str]:
    """Return a mapping from numeric condition_label to its readable name."""
    labels_df = pd.read_csv(labels_file)
    return dict(zip(labels_df["condition_label"], labels_df["condition_name"], strict=True))


def _load_split(path: Path) -> Dataset:
    df = pd.read_csv(path)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].str.strip()
    df = df[df[TEXT_COLUMN].str.len() > 0]
    return Dataset(
        texts=df[TEXT_COLUMN].reset_index(drop=True),
        labels=df[LABEL_COLUMN].astype(int).reset_index(drop=True),
    )


def load_train_dataset(path: Path = TRAIN_FILE) -> Dataset:
    """Load and clean the training split."""
    return _load_split(path)


def load_test_dataset(path: Path = TEST_FILE) -> Dataset:
    """Load and clean the held-out test split."""
    return _load_split(path)


def save_processed(dataset: Dataset, filename: str, directory: Path = PROCESSED_DIR) -> Path:
    """Persist a cleaned split to data/processed/ as CSV."""
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / filename
    pd.DataFrame({TEXT_COLUMN: dataset.texts, LABEL_COLUMN: dataset.labels}).to_csv(
        out_path, index=False
    )
    return out_path

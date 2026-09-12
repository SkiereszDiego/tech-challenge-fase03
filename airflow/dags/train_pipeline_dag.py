"""DAG that simulates the (re)training pipeline for the medical triage model.

Tasks:
    ingest_data   -> validates/loads the raw train and test CSVs.
    train_model   -> trains the TF-IDF + Logistic Regression pipeline and
                      persists the model artifact + evaluation metrics.

This DAG intentionally reuses the exact same code used to train the model
locally (`medical_triage_mlops.ml.train`), so a manual run
(`uv run python -m medical_triage_mlops.ml.train`) and an Airflow-orchestrated
run always produce the same result.
"""

import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

# Make the project's `src/` package importable from within the DAG process,
# since Airflow workers don't install the project as a package by default.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(PROJECT_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@dag(
    dag_id="medical_triage_training_pipeline",
    description="Ingestão de dados e (re)treino do classificador de triagem médica",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "training", "tech-challenge-fase3"],
)
def medical_triage_training_pipeline():
    @task
    def ingest_data() -> dict:
        """Load and validate the raw train/test CSVs, returning row counts."""
        import os

        os.chdir(PROJECT_ROOT)
        from medical_triage_mlops.ml.dataset import load_test_dataset, load_train_dataset

        train_ds = load_train_dataset()
        test_ds = load_test_dataset()

        if len(train_ds.texts) < 2000:
            raise ValueError(
                f"Dataset de treino tem apenas {len(train_ds.texts)} amostras "
                "(mínimo esperado: 2000)."
            )

        return {"n_train": len(train_ds.texts), "n_test": len(test_ds.texts)}

    @task
    def train_model(ingest_result: dict) -> dict:
        """Train the pipeline and persist the model + evaluation metrics."""
        import json
        import os

        os.chdir(PROJECT_ROOT)
        from medical_triage_mlops.ml.train import METRICS_PATH, train

        print(f"Treinando com {ingest_result['n_train']} amostras de treino...")
        train()
        return json.loads(METRICS_PATH.read_text())

    train_model(ingest_data())


medical_triage_training_pipeline()

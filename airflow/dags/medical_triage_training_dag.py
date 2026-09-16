"""DAG simples de treino/retreino do classificador de triagem médica.

Etapas: ingestão dos dados brutos (Kaggle) -> pré-processamento (mapeamento
de urgência) -> treino do modelo -> conversão para ONNX.

Todos os artefatos (dataset processado, modelo .joblib e .onnx) são escritos
em `/opt/airflow/project/{data,models}`, que são montados no host a partir de
`data/` e `models/` na raiz do repositório (ver airflow/docker-compose.yml).
"""

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow/project")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "model.onnx"


def ingest_data() -> None:
    from medical_triage_mlops.ml.data import download_raw_dataset

    download_raw_dataset(RAW_DIR)


def preprocess_data() -> None:
    from medical_triage_mlops.ml.data import build_processed_dataset

    df = build_processed_dataset(RAW_DIR, PROCESSED_DIR)
    print(f"Dataset processado: {len(df)} amostras.")


def train_model_task() -> None:
    import pandas as pd

    from medical_triage_mlops.ml.train import save_model, train_model

    df = pd.read_csv(PROCESSED_DIR / "dataset.csv")
    pipeline, metrics = train_model(df)
    save_model(pipeline, MODEL_PATH)
    print("Métricas:", metrics)


def convert_to_onnx_task() -> None:
    from medical_triage_mlops.ml.onnx_convert import convert_pipeline_to_onnx
    from medical_triage_mlops.ml.train import load_model

    pipeline = load_model(MODEL_PATH)
    convert_pipeline_to_onnx(pipeline, ONNX_MODEL_PATH)


with DAG(
    dag_id="medical_triage_training",
    description="Ingestão, pré-processamento, treino e conversão ONNX do classificador de triagem.",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["medical-triage", "training"],
) as dag:
    ingest = PythonOperator(task_id="ingest_data", python_callable=ingest_data)
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    train = PythonOperator(task_id="train_model", python_callable=train_model_task)
    convert_onnx = PythonOperator(task_id="convert_to_onnx", python_callable=convert_to_onnx_task)

    ingest >> preprocess >> train >> convert_onnx

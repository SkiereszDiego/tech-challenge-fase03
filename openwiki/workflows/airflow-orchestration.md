---
type: workflow
title: Airflow Workflow Orchestration
description: Technical reference for the Apache Airflow DAG orchestration pipeline (medical_triage_training), task dependency sequence, container environment, volume mounts, and data artifact lifecycle.
tags: [airflow, orchestration, dag, mlops, containerization, docker-compose, training-pipeline]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-24T12:24:27.905Z
sources:
  - id: openwiki-source-f0e5a8487b6f56cd83a0e75c
    resource: repo://airflow/dags/medical_triage_training_dag.py
  - id: openwiki-source-caa539f6d0ccde80d3009654
    resource: repo://airflow/docker-compose.yml
  - id: openwiki-source-1b965af0f2701803ffe1c767
    resource: repo://airflow/Dockerfile
  - id: openwiki-source-cd879a63366d809a09833969
    resource: repo://airflow/requirements.txt
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-2299ccd7126379c969cc8df0
    resource: repo://src/medical_triage_mlops/ml/data.py
  - id: openwiki-source-137e7f9335fdb6c59e1aa948
    resource: repo://src/medical_triage_mlops/ml/onnx_convert.py
  - id: openwiki-source-d16b2457c16835197e75ead3
    resource: repo://src/medical_triage_mlops/ml/train.py
generated: { by: "openwiki/0.7.2", at: "2026-10-24T12:24:27.905Z" }
---

# Airflow Workflow Orchestration

The **Airflow Workflow Orchestration** subsystem automates the offline retraining and artifact generation pipeline for the medical triage classification service. Managed by Apache Airflow running in a dedicated Docker container, the orchestration layer guarantees structured, reproducible execution of data ingestion, pre-processing, model fitting, and ONNX graph serialization.

The primary workflow is defined in `airflow/dags/medical_triage_training_dag.py` under the DAG identifier `medical_triage_training`.

---

## Container Environment & Deployment Configuration

The Airflow runtime operates as a standalone service defined in `airflow/docker-compose.yml` and built from `airflow/Dockerfile`.

### Container Image & Base Setup

The container image is derived from the official `apache/airflow:2.10.3-python3.12` image. Custom Python dependencies are installed during build via `airflow/requirements.txt`:

```dockerfile
FROM apache/airflow:2.10.3-python3.12

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
```

### Dependency Synchronization

To ensure that models trained inside the Airflow environment can be loaded and executed seamlessly by the FastAPI inference engine, dependency versions in `airflow/requirements.txt` are tightly aligned with the root `pyproject.toml` manifest:

| Package | Version Requirement | Purpose in Retraining Pipeline |
|---|---|---|
| `pandas` | `>=3.0.5` | Data loading, concatenation, and CSV generation |
| `scikit-learn` | `>=1.9.0` | TF-IDF vectorization and Logistic Regression model training |
| `joblib` | `>=1.5.3` | Binary serialization of fitted `sklearn.pipeline.Pipeline` objects |
| `kagglehub` | `>=1.0.0` | Direct download of Kaggle Medical Abstracts TC Corpus |
| `skl2onnx` | `>=1.20.0` | Conversion of Scikit-Learn pipelines to ONNX format |
| `onnx` | `>=1.22.0` | ONNX graph definitions and binary serialization |
| `onnxruntime` | `>=1.28.0` | Local verification of ONNX model execution |

This version parity prevents `InconsistentVersionWarning` exceptions or unpickling failures when the API layer loads `models/model.joblib`.

### Docker Compose & Environment Variables

The container configuration in `airflow/docker-compose.yml` launches Airflow in `standalone` mode, exposing the Airflow Web UI on host port `8080`:

```yaml
services:
  airflow:
    build: .
    container_name: medical-triage-airflow
    command: standalone
    ports:
      - "8080:8080"
    environment:
      - AIRFLOW__CORE__LOAD_EXAMPLES=false
      - AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
      - PYTHONPATH=/opt/airflow/project/src
    volumes:
      - .:/opt/airflow
      - ../src:/opt/airflow/project/src:ro
      - ../data:/opt/airflow/project/data
      - ../models:/opt/airflow/project/models
    restart: unless-stopped
```

Key environment configuration settings include:
- `AIRFLOW__CORE__LOAD_EXAMPLES=false`: Disables built-in Airflow example DAGs to simplify the Web UI view.
- `AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags`: Specifies the target directory scanned by the Airflow scheduler for DAG definitions.
- `PYTHONPATH=/opt/airflow/project/src`: Adds the mounted core application source code (`medical_triage_mlops` package) to the Python module search path, allowing PythonOperator tasks to import training logic directly.

---

## Volume Mounts & Persistent Paths

The Airflow container uses Docker bind mounts to isolate code execution while persisting generated data and model artifacts to the host filesystem.

```
/opt/airflow/ (Airflow Root)
├── dags/                          <-- Mounted from ./ (airflow/)
└── project/                       <-- Mounted from repository root
    ├── src/  (:ro)                <-- Read-only source code (../src)
    ├── data/                      <-- Persistent data storage (../data)
    │   ├── raw/                   <-- Downloaded Kaggle CSV files
    │   └── processed/             <-- Preprocessed dataset.csv
    └── models/                    <-- Model registry output (../models)
        ├── model.joblib           <-- Scikit-learn binary model
        └── model.onnx             <-- Serialized ONNX runtime graph
```

- **Read-Only Code Boundary (`../src:/opt/airflow/project/src:ro`)**: Prevents Airflow tasks from modifying core application logic while allowing full access to `medical_triage_mlops` functions.
- **Data Persistence (`../data:/opt/airflow/project/data`)**: Hosts both raw downloaded datasets and normalized feature CSVs across workflow runs.
- **Model Storage (`../models:/opt/airflow/project/models`)**: Receives updated Scikit-Learn `.joblib` artifacts and converted `.onnx` files, which are immediately accessible to the API serving container.

---

## DAG Architecture & Task Pipeline

The `medical_triage_training` DAG defines a linear pipeline of four Python execution tasks.

```mermaid
flowchart TD
    subgraph Container_Environment["Airflow Container Environment (/opt/airflow)"]
        subgraph DAG_Tasks["DAG: medical_triage_training"]
            Ingest["ingest_data"]
            Preprocess["preprocess_data"]
            Train["train_model_task"]
            ConvertONNX["convert_to_onnx_task"]
        end

        subgraph Volume_Mounts["Mounted Project Paths (/opt/airflow/project)"]
            RawData["data/raw/"]
            ProcessedData["data/processed/dataset.csv"]
            JoblibModel["models/model.joblib"]
            ONNXModel["models/model.onnx"]
        end
    end

    subgraph Source_Code["Source Code Mount (:ro)"]
        MLModule["medical_triage_mlops.ml"]
    end

    MLModule -.->"Imported by PythonOperators" DAG_Tasks
    Ingest -->|"download_raw_dataset()"| RawData
    Ingest --> Preprocess
    RawData --> Preprocess
    Preprocess -->|"build_processed_dataset()"| ProcessedData
    Preprocess --> Train
    ProcessedData --> Train
    Train -->|"train_model() & save_model()"| JoblibModel
    Train --> ConvertONNX
    JoblibModel --> ConvertONNX
    ConvertONNX -->|"convert_pipeline_to_onnx()"| ONNXModel
```
*Task dependency chain and artifact persistence flow for the medical_triage_training Airflow DAG.*

### DAG Declaration Parameters

The DAG is configured in `airflow/dags/medical_triage_training_dag.py` with explicit control parameters:

```python
with DAG(
    dag_id="medical_triage_training",
    description="Ingestão, pré-processamento, treino e conversão ONNX do classificador de triagem.",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["medical-triage", "training"],
) as dag:
```

- `schedule=None`: Execution is triggered on demand (manually via Web UI / CLI or via external API trigger) rather than running on a fixed cron schedule.
- `catchup=False`: Disables backfilling historical DAG runs prior to `start_date`.

---

## Detailed Task Breakdown

Each step in the DAG pipeline delegates business logic to specialized functions in the `medical_triage_mlops.ml` package.

### 1. Ingest Data (`ingest_data`)

- **Operator**: `PythonOperator(task_id="ingest_data", python_callable=ingest_data)`
- **Function**: Invokes `medical_triage_mlops.ml.data.download_raw_dataset(RAW_DIR)`.
- **Behavior**:
  1. Uses `kagglehub.dataset_download("saharalaa/medical-abstracts-tc-corpus")` to download the raw dataset into the local cache.
  2. Copies `medical_tc_train.csv` and `medical_tc_test.csv` to `/opt/airflow/project/data/raw/`.
  3. Uses `shutil.copyfile` to copy file bytes without copying file permissions or extended attributes, avoiding `PermissionError` on Windows Docker Desktop bind mounts.

### 2. Preprocess Data (`preprocess_data`)

- **Operator**: `PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)`
- **Function**: Invokes `medical_triage_mlops.ml.data.build_processed_dataset(RAW_DIR, PROCESSED_DIR)`.
- **Behavior**:
  1. Reads raw train and test CSVs from `/opt/airflow/project/data/raw/`.
  2. Concatenates both splits and standardizes column naming (`medical_abstract` -> `text`).
  3. Maps numeric condition IDs (1–5) to clinical urgency strings (`normal`, `atenção`, `urgente`) via `map_to_urgency()`.
  4. Writes the transformed DataFrame to `/opt/airflow/project/data/processed/dataset.csv`.

### 3. Train Model (`train_model`)

- **Operator**: `PythonOperator(task_id="train_model", python_callable=train_model_task)`
- **Function**: Executes training sequence via `medical_triage_mlops.ml.train`.
- **Behavior**:
  1. Loads `/opt/airflow/project/data/processed/dataset.csv`.
  2. Splits dataset using stratified sampling (`test_size=0.2`, `random_state=42`).
  3. Fits a `Pipeline` consisting of `TfidfVectorizer(max_features=20000, stop_words="english")` and `LogisticRegression(max_iter=1000)`.
  4. Evaluates performance metrics and saves fitted pipeline via `save_model()` to `/opt/airflow/project/models/model.joblib`.

### 4. Convert to ONNX (`convert_to_onnx`)

- **Operator**: `PythonOperator(task_id="convert_to_onnx", python_callable=convert_to_onnx_task)`
- **Function**: Executes conversion via `medical_triage_mlops.ml.onnx_convert`.
- **Behavior**:
  1. Loads `/opt/airflow/project/models/model.joblib` using `load_model()`.
  2. Invokes `convert_pipeline_to_onnx(pipeline, ONNX_MODEL_PATH)`.
  3. Configures initial input types as `StringTensorType([None, 1])` and disables `zipmap` output wrappers for maximum performance and compatibility with `OnnxPredictor`.
  4. Writes the converted binary model to `/opt/airflow/project/models/model.onnx`.

---

## Operational Safeguards & Invariants

1. **Sequential Task Execution**: The linear dependency chain (`ingest >> preprocess >> train >> convert_onnx`) guarantees that downstream tasks only execute after prerequisites complete successfully.
2. **Path Resolution Consistency**: All file paths are anchored to `PROJECT_ROOT = Path("/opt/airflow/project")`, ensuring uniform file resolution across task boundaries regardless of working directory.
3. **Idempotent Execution**: Each task creates parent directories (`mkdir(parents=True, exist_ok=True)`) and overwrites target artifacts on rerun, making DAG retries safe and deterministic.

---

## Testing & Module Verification

While Airflow DAG execution orchestrates tasks in containerized production, underlying task functions are covered by unit tests in the core repository:

- **Data Ingestion & Urgency Mapping**: Tested in `tests/unit/test_data.py` (verifying dataset concatenation, column mapping, and error handling for unknown labels).
- **Training Pipeline & Serialization**: Tested in `tests/unit/test_train.py` (verifying model fitting, stratification, prediction structure, and `joblib` roundtrip integrity).
- **ONNX Export**: Tested in `tests/unit/test_onnx_convert.py` (verifying parity between Scikit-Learn predictions and ONNX Runtime evaluation).

---

## Related Documents

- [Architecture & System Overview](../architecture/overview.md)
- [Data Pipeline & Urgency Mapping](../concepts/data-and-urgency-mapping.md)

from pathlib import Path

import pandas as pd

from medical_triage_mlops.ml.train import load_model, save_model, train_model


def test_train_model_fits_and_predicts(sample_dataframe: pd.DataFrame) -> None:
    pipeline, metrics = train_model(sample_dataframe, test_size=0.25)

    predictions = pipeline.predict(["Sudden cardiac arrest, patient unresponsive."])

    assert predictions[0] == "urgente"
    assert "accuracy" in metrics
    assert set(pipeline.classes_) == {"normal", "atenção", "urgente"}


def test_save_and_load_model_roundtrip(sample_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    pipeline, _ = train_model(sample_dataframe, test_size=0.25)
    model_path = tmp_path / "model.joblib"

    save_model(pipeline, model_path)
    loaded = load_model(model_path)

    text = "Routine annual checkup, patient feels well."
    assert loaded.predict([text])[0] == pipeline.predict([text])[0]

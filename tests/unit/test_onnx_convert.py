from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from medical_triage_mlops.ml.onnx_convert import convert_pipeline_to_onnx
from medical_triage_mlops.ml.train import train_model


def test_onnx_predictions_match_sklearn(sample_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    pipeline, _ = train_model(sample_dataframe, test_size=0.25)
    onnx_path = convert_pipeline_to_onnx(pipeline, tmp_path / "model.onnx")

    texts = [
        "Sudden cardiac arrest, patient unresponsive.",
        "Routine annual checkup, patient feels well.",
        "Suspicious mass detected, biopsy recommended.",
    ]

    sklearn_predictions = pipeline.predict(texts)

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    input_array = np.array([[text] for text in texts], dtype=object)
    onnx_labels, _onnx_probabilities = session.run(None, {input_name: input_array})

    assert list(onnx_labels) == list(sklearn_predictions)

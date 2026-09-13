import pytest
from fastapi.testclient import TestClient

from medical_triage_mlops.api.main import app

client = TestClient(app)

SAMPLE_REPORT = (
    "Patient presents with acute chest pain radiating to the left arm, "
    "shortness of breath and elevated troponin levels suggestive of "
    "myocardial infarction."
)


def test_classify_returns_prediction() -> None:
    response = client.post("/classify", json={"text": SAMPLE_REPORT})

    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert "predicted_label" in body
    assert "probabilities" in body
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_classify_rejects_short_text() -> None:
    response = client.post("/classify", json={"text": "short"})

    assert response.status_code == 422


def test_classify_does_not_depend_on_raw_dataset(monkeypatch, tmp_path) -> None:
    """Regression test: inference must not read data/raw/*.

    The raw training data is excluded from the Docker image (see
    .dockerignore), so classify_report() must work using only the
    artifacts under models/ (model.joblib + labels.json).
    """
    import medical_triage_mlops.ml.predict as predict_module

    monkeypatch.setattr(predict_module, "LABELS_ARTIFACT_PATH", tmp_path / "does-not-exist.json")
    predict_module._load_labels.cache_clear()

    try:
        with pytest.raises(predict_module.ModelNotFoundError):
            predict_module.classify_report(SAMPLE_REPORT)
    finally:
        predict_module._load_labels.cache_clear()

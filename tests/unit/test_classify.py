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

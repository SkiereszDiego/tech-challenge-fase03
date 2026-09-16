from fastapi.testclient import TestClient

from medical_triage_mlops.api.main import app, get_current_predictor
from medical_triage_mlops.ml.inference import PredictionResult


class FakePredictor:
    def predict(self, texts: list[str]) -> list[PredictionResult]:
        return [
            PredictionResult(
                label="urgente",
                confidence=0.9,
                scores={"normal": 0.05, "atenção": 0.05, "urgente": 0.9},
            )
            for _ in texts
        ]


app.dependency_overrides[get_current_predictor] = lambda: FakePredictor()
client = TestClient(app)


def test_classify_returns_prediction() -> None:
    response = client.post("/classify", json={"text": "Acute chest pain and shortness of breath."})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "urgente"
    assert body["confidence"] == 0.9
    assert body["scores"]["urgente"] == 0.9


def test_classify_rejects_empty_text() -> None:
    response = client.post("/classify", json={"text": ""})

    assert response.status_code == 422


def test_metrics_endpoint_exposes_request_count() -> None:
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text

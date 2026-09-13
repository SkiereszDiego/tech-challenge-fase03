import logging
import time

from fastapi import FastAPI, HTTPException

from medical_triage_mlops.api.schemas import ReportClassificationResponse, ReportRequest
from medical_triage_mlops.ml.predict import ModelNotFoundError, classify_report

logger = logging.getLogger("medical_triage_mlops")

app = FastAPI(
    title="Medical Triage API",
    description="API para classificação de urgência em laudos médicos.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Return the current health status of the API."""
    return {"status": "ok"}


@app.post("/classify", response_model=ReportClassificationResponse, tags=["Classification"])
def classify(payload: ReportRequest) -> ReportClassificationResponse:
    """Classify a medical report's text and return the predicted category."""
    start = time.perf_counter()
    try:
        result = classify_report(payload.text)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("classify request handled in %.2f ms", elapsed_ms)
    return ReportClassificationResponse(**result)

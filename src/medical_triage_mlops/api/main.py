import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from medical_triage_mlops.api.schemas import TriageRequest, TriageResponse
from medical_triage_mlops.core.config import get_settings
from medical_triage_mlops.ml.inference import Predictor, get_predictor
from medical_triage_mlops.monitoring.metrics import PrometheusMiddleware, metrics_endpoint

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    try:
        app.state.predictor = get_predictor(Path(settings.model_path))
    except Exception:
        logger.warning(
            "Nenhum modelo encontrado em %s — treine o modelo antes de usar /classify.",
            settings.model_path,
        )
        app.state.predictor = None
    yield


app = FastAPI(
    title="Medical Triage API",
    description="API para classificação de urgência em laudos médicos.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(PrometheusMiddleware)
app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], tags=["Monitoring"])


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Retorna o estado atual da API"""
    return {"status": "ok"}


def get_current_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Modelo não carregado. Treine o modelo primeiro."
        )
    return predictor


@app.post("/classify", response_model=TriageResponse, tags=["Triage"])
def classify(
    payload: TriageRequest,
    predictor: Predictor = Depends(get_current_predictor),  # noqa: B008
) -> TriageResponse:
    result = predictor.predict([payload.text])[0]
    return TriageResponse(label=result.label, confidence=result.confidence, scores=result.scores)

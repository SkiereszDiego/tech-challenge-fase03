# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install dependencies first, separately from app code, to maximize layer caching.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    MODEL_PATH=models/model.joblib \
    ONNX_MODEL_PATH=models/model.onnx

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY models/ models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "medical_triage_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

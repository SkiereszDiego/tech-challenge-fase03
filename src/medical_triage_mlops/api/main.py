from fastapi import FastAPI

app = FastAPI(
    title="Medical Triage API",
    description="API para classificação de urgência em laudos médicos.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Return the current health status of the API."""
    return {"status": "ok"}

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, description="Texto do laudo médico a ser classificado.")


class TriageResponse(BaseModel):
    label: str = Field(description="Nível de urgência previsto: normal, atenção ou urgente.")
    confidence: float = Field(description="Confiança (probabilidade) da classe prevista.")
    scores: dict[str, float] = Field(description="Probabilidade prevista para cada classe.")

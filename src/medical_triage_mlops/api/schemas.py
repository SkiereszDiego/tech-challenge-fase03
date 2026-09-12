from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    """Payload with the medical report text to classify."""

    text: str = Field(
        ...,
        min_length=10,
        description="Texto do laudo médico a ser classificado.",
        examples=[
            "Patient presents with acute chest pain radiating to the left arm, "
            "shortness of breath and elevated troponin levels."
        ],
    )


class ReportClassificationResponse(BaseModel):
    """Prediction result for a classified medical report."""

    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]

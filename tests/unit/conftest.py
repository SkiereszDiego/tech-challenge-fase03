import pandas as pd
import pytest

from medical_triage_mlops.core.labels import URGENCY_ATENCAO, URGENCY_NORMAL, URGENCY_URGENTE

NORMAL_TEXTS = [
    "Routine annual checkup, patient feels well with no complaints.",
    "Standard physical exam results within normal range.",
    "Mild seasonal allergy symptoms, otherwise healthy.",
    "Regular dental cleaning appointment, no issues found.",
    "Patient reports no symptoms of concern during visit.",
    "Minor cold symptoms resolving on their own.",
    "Wellness visit, blood tests within normal range.",
    "Follow-up visit, patient recovering well as expected.",
]

ATENCAO_TEXTS = [
    "Tumor markers slightly elevated, requires close monitoring.",
    "Abnormal nerve conduction study observed in the extremities.",
    "Suspicious mass detected, biopsy recommended for evaluation.",
    "Gradual memory loss and confusion reported by patient.",
    "Chronic headaches with occasional dizziness under investigation.",
    "Elevated tumor marker levels detected on recent bloodwork.",
    "Early signs of neurological decline noted during exam.",
    "Persistent numbness in limbs warrants specialist referral.",
]

URGENTE_TEXTS = [
    "Acute chest pain with shortness of breath and diaphoresis.",
    "Sudden cardiac arrest reported, emergency resuscitation underway.",
    "Severe hypertensive crisis requiring immediate intervention.",
    "Myocardial infarction confirmed by ECG, patient unstable.",
    "Patient in cardiac arrest, unresponsive on arrival.",
    "Critical drop in blood pressure, emergency response needed.",
    "Acute stroke symptoms with sudden onset weakness.",
    "Severe arrhythmia detected, immediate cardiac intervention required.",
]


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    rows = (
        [(text, URGENCY_NORMAL) for text in NORMAL_TEXTS]
        + [(text, URGENCY_ATENCAO) for text in ATENCAO_TEXTS]
        + [(text, URGENCY_URGENTE) for text in URGENTE_TEXTS]
    )
    return pd.DataFrame(rows, columns=["text", "label"])

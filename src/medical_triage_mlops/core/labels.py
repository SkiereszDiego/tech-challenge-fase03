"""Níveis de urgência usados na classificação de triagem.

O dataset de origem (Medical Abstracts TC Corpus) só traz um rótulo de
categoria de doença (`condition_label`), não uma classificação clínica de
urgência. `CONDITION_TO_URGENCY` mapeia essas categorias para os 3 níveis de
urgência exigidos pelo desafio; é uma simplificação pedagógica documentada no
README, não uma afirmação clínica.
"""

URGENCY_NORMAL = "normal"
URGENCY_ATENCAO = "atenção"
URGENCY_URGENTE = "urgente"

URGENCY_LABELS: list[str] = [URGENCY_NORMAL, URGENCY_ATENCAO, URGENCY_URGENTE]

CONDITION_NAMES: dict[int, str] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

CONDITION_TO_URGENCY: dict[int, str] = {
    4: URGENCY_URGENTE,
    1: URGENCY_ATENCAO,
    3: URGENCY_ATENCAO,
    2: URGENCY_NORMAL,
    5: URGENCY_NORMAL,
}

import pandas as pd
import pytest

from medical_triage_mlops.ml.data import map_to_urgency


def test_map_to_urgency_maps_all_known_condition_labels() -> None:
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d", "e"],
            "condition_label": [1, 2, 3, 4, 5],
        }
    )

    mapped = map_to_urgency(df)

    assert list(mapped.columns) == ["text", "label"]
    assert set(mapped["label"]) == {"normal", "atenção", "urgente"}
    assert mapped.loc[mapped["text"] == "d", "label"].item() == "urgente"


def test_map_to_urgency_raises_for_unknown_condition_label() -> None:
    df = pd.DataFrame({"text": ["a"], "condition_label": [99]})

    with pytest.raises(ValueError, match="sem mapeamento"):
        map_to_urgency(df)

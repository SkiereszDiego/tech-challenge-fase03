"""Compara a latência de inferência entre o pipeline sklearn e o modelo ONNX.

Uso:
    uv run python scripts/measure_latency.py
"""

import json
import statistics
import time
from pathlib import Path

from medical_triage_mlops.core.config import get_settings
from medical_triage_mlops.ml.inference import OnnxPredictor, Predictor, SklearnPredictor

REPORT_DIR = Path("reports/latency")
WARMUP_RUNS = 10
MEASURED_RUNS = 200

SAMPLE_TEXTS = [
    "Patient presents with acute chest pain radiating to the left arm, "
    "diaphoresis and shortness of breath.",
    "Routine follow-up visit, patient reports feeling well with no new complaints.",
    "Persistent abdominal discomfort for the past three days, mild and non-radiating.",
]


def _measure(predictor: Predictor) -> dict[str, float]:
    """
    Mensura a latência de inferência do preditor fornecido,
    realizando um número especificado de execuções de aquecimento e medição.
    """

    for _ in range(WARMUP_RUNS):
        predictor.predict(SAMPLE_TEXTS)

    durations_ms = []
    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        predictor.predict(SAMPLE_TEXTS)
        durations_ms.append((time.perf_counter() - start) * 1000)

    durations_ms.sort()
    p95_index = int(len(durations_ms) * 0.95)
    return {
        "mean_ms": statistics.mean(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "p95_ms": durations_ms[p95_index],
        "runs": MEASURED_RUNS,
        "batch_size": len(SAMPLE_TEXTS),
    }


def main() -> None:
    settings = get_settings()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sklearn_predictor = SklearnPredictor(Path(settings.model_path))
    onnx_predictor = OnnxPredictor(Path(settings.onnx_model_path))

    results = {
        "sklearn": _measure(sklearn_predictor),
        "onnx": _measure(onnx_predictor),
    }
    speedup = results["sklearn"]["mean_ms"] / results["onnx"]["mean_ms"]
    results["speedup_x"] = speedup

    (REPORT_DIR / "latency_comparison.json").write_text(json.dumps(results, indent=2))

    sk = results["sklearn"]
    onnx = results["onnx"]
    sk_row = f"| scikit-learn | {sk['mean_ms']:.3f} | {sk['median_ms']:.3f} | {sk['p95_ms']:.3f} |"
    onnx_row = (
        f"| ONNX Runtime | {onnx['mean_ms']:.3f} | {onnx['median_ms']:.3f} | {onnx['p95_ms']:.3f} |"
    )
    report_md = f"""# Comparação de latência — sklearn vs ONNX Runtime

Batch de {len(SAMPLE_TEXTS)} textos, {MEASURED_RUNS} execuções (após {WARMUP_RUNS} de aquecimento).

| Backend | Média (ms) | Mediana (ms) | P95 (ms) |
|---|---|---|---|
{sk_row}
{onnx_row}

**Speedup (ONNX vs sklearn):** {speedup:.2f}x
"""
    (REPORT_DIR / "latency_comparison.md").write_text(report_md, encoding="utf-8")

    print(report_md)


if __name__ == "__main__":
    main()

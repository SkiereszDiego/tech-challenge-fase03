"""Measure baseline inference latency of the classification pipeline.

Runs the loaded model directly (in-process, no HTTP overhead) against a
sample of test reports and records p50/p95/p99 latency. This is the
baseline used in Etapa 4 to compare against the ONNX-optimized model.

Usage:
    uv run python -m medical_triage_mlops.ml.benchmark_latency
"""

import json
import statistics
import time
from pathlib import Path

from medical_triage_mlops.ml.dataset import load_test_dataset
from medical_triage_mlops.ml.predict import classify_report

OUTPUT_PATH = Path("reports/latency/baseline_latency.json")
N_WARMUP = 10
N_SAMPLES = 200


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    index = min(int(len(values) * pct), len(values) - 1)
    return values[index]


def run_benchmark() -> None:
    test_ds = load_test_dataset()
    texts = test_ds.texts.tolist()[: N_WARMUP + N_SAMPLES]

    print(f"Aquecendo com {N_WARMUP} chamadas...")
    for text in texts[:N_WARMUP]:
        classify_report(text)

    print(f"Medindo latência em {N_SAMPLES} chamadas...")
    durations_ms: list[float] = []
    for text in texts[N_WARMUP : N_WARMUP + N_SAMPLES]:
        start = time.perf_counter()
        classify_report(text)
        durations_ms.append((time.perf_counter() - start) * 1000)

    result = {
        "model": "tfidf_logreg_baseline",
        "n_samples": len(durations_ms),
        "mean_ms": statistics.mean(durations_ms),
        "p50_ms": percentile(durations_ms, 0.50),
        "p95_ms": percentile(durations_ms, 0.95),
        "p99_ms": percentile(durations_ms, 0.99),
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
    }

    print(json.dumps(result, indent=2))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Resultado salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    run_benchmark()

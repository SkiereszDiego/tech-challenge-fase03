# Comparação de latência — sklearn vs ONNX Runtime

Batch de 3 textos, 200 execuções (após 10 de aquecimento).

| Backend | Média (ms) | Mediana (ms) | P95 (ms) |
|---|---|---|---|
| scikit-learn | 0.694 | 0.585 | 0.999 |
| ONNX Runtime | 0.297 | 0.277 | 0.402 |

**Speedup (ONNX vs sklearn):** 2.34x

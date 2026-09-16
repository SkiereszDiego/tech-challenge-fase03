from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline


def convert_pipeline_to_onnx(pipeline: Pipeline, output_path: Path) -> Path:
    """Converte um pipeline sklearn ajustado (TF-IDF + classificador) para ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier = pipeline.steps[-1][1]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={id(classifier): {"zipmap": False}},
    )

    output_path.write_bytes(onnx_model.SerializeToString())
    return output_path

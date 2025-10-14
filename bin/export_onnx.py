#!/usr/bin/env python3
"""Utility to export IsoNet TensorFlow models to ONNX."""
import argparse
import pathlib
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a trained IsoNet TensorFlow (.h5) model into ONNX."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained TensorFlow model (.h5 or SavedModel directory).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Target ONNX file path. Parent directory will be created if missing.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to target (default: 13).",
    )
    parser.add_argument(
        "--fold-constants",
        action="store_true",
        help="Enable constant folding during export (can reduce graph size).",
    )
    return parser.parse_args()


def _ensure_dependencies():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to load the trained IsoNet model. "
            "Install e.g. `pip install tensorflow`."
        ) from exc
    try:
        import tf2onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "tf2onnx is required for exporting to ONNX. "
            "Install it via `pip install tf2onnx`."
        ) from exc


def export(model_path: pathlib.Path, output_path: pathlib.Path, opset: int, fold_constants: bool):
    import tensorflow as tf
    import tf2onnx

    if model_path.is_dir():
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        model = tf.keras.models.load_model(str(model_path), compile=False)

    input_signature = []
    for node in model.inputs:
        tensor_shape = []
        for dim in node.shape:
            tensor_shape.append(dim if dim is not None else None)
        input_signature.append(
            tf.TensorSpec(shape=tensor_shape, dtype=node.dtype, name=node.name.split(":")[0])
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature or None,
        opset=opset,
        output_path=str(output_path),
        # fold_const=fold_constants,
    )


def main():
    args = parse_args()
    _ensure_dependencies()
    model_path = pathlib.Path(args.model).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    try:
        export(model_path, output_path, args.opset, args.fold_constants)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] Failed to export model: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Wrote ONNX model to {output_path}")


if __name__ == "__main__":
    main()

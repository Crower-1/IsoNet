"""ONNXRuntime-based prediction helpers mirroring the TensorFlow workflow."""
from __future__ import annotations

import logging
import os
import sys
from typing import Iterable, List, Sequence

import mrcfile
import numpy as np
from IsoNet.preprocessing.img_processing import normalize
from IsoNet.util.toTile import reform3D
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - import guard triggers before runtime errors
    raise ImportError(
        "onnxruntime is required for ONNX inference. Install it with `pip install onnxruntime` "
        "or `onnxruntime-gpu` depending on your environment."
    ) from exc


def _parse_providers(provider_spec: str | Sequence[str] | None) -> List[str] | None:
    if provider_spec is None:
        return None
    if isinstance(provider_spec, str):
        candidates: Iterable[str] = (item.strip() for item in provider_spec.split(","))
    else:
        candidates = (str(item).strip() for item in provider_spec)
    providers = [item for item in candidates if item]
    return providers or None


def _load_session(model_path: str, providers: Sequence[str] | None = None) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if providers:
        missing = [p for p in providers if p not in available]
        if missing:
            raise RuntimeError(
                f"Requested ONNXRuntime providers {missing} are not available. "
                f"Detected providers: {available}"
            )
        provider_chain = list(providers)
    else:
        provider_chain = available
    session_opts = ort.SessionOptions()
    return ort.InferenceSession(model_path, sess_options=session_opts, providers=provider_chain)


def predict_one(args, one_tomo: str, output_file: str | None = None):
    """Predict one tomogram in MRC format using an ONNX model."""
    logging.info("Loading ONNX model from %s", args.model)
    providers = _parse_providers(getattr(args, "onnx_providers", None))
    session = _load_session(args.model, providers=providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    root_name = os.path.splitext(os.path.basename(one_tomo))[0]
    if output_file is None:
        if os.path.isdir(args.output_dir):
            output_file = os.path.join(args.output_dir, root_name + "_corrected.mrc")
        else:
            output_file = root_name + "_corrected.mrc"

    logging.info("Predicting %s with ONNXRuntime (providers=%s)", root_name, session.get_providers())

    with mrcfile.open(one_tomo, permissive=True) as mrc_data:
        real_data = mrc_data.data.astype(np.float32) * -1
        voxelsize = mrc_data.voxel_size

    data = normalize(real_data, percentile=args.normalize_percentile)
    reform_ins = reform3D(data, args.cube_size, args.crop_size, 9)
    cubes = reform_ins.pad_and_crop().astype(np.float32)
    cubes = cubes[..., np.newaxis]  # add channel dimension

    batch_size = args.batch_size
    if batch_size is None or batch_size <= 0:
        batch_size = 1
    num_patches = cubes.shape[0]
    remainder = num_patches % batch_size
    if remainder:
        pad_len = batch_size - remainder
        cubes = np.concatenate([cubes, cubes[:pad_len]], axis=0)
    else:
        pad_len = 0

    predictions = np.zeros(cubes.shape[:-1], dtype=np.float32)
    num_batches = cubes.shape[0] // batch_size
    logging.info("Total batches: %s", num_batches)

    for idx in tqdm(range(num_batches), file=sys.stdout):
        batch = cubes[idx * batch_size : (idx + 1) * batch_size]
        ort_inputs = {input_name: batch}
        ort_outputs = session.run(None, ort_inputs)
        batch_out = ort_outputs[0]
        if batch_out.ndim == 5 and batch_out.shape[-1] == 1:
            batch_out = batch_out[..., 0]
        elif batch_out.ndim != 4:
            raise RuntimeError(f"Unexpected ONNX output shape: {batch_out.shape}")
        predictions[idx * batch_size : (idx + 1) * batch_size] = batch_out

    if pad_len:
        predictions = predictions[:num_patches]

    restored = reform_ins.restore(predictions)
    restored = normalize(restored, percentile=args.normalize_percentile)

    with mrcfile.new(output_file, overwrite=True) as output_mrc:
        output_mrc.set_data(-restored)
        output_mrc.voxel_size = voxelsize

    logging.info("Saved corrected tomogram to %s", output_file)

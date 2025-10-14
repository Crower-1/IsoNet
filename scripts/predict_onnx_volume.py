#!/usr/bin/env python3
"""Standalone ONNX prediction script for IsoNet models.

This replicates the TensorFlow-based `predict_one` workflow using only NumPy and
onnxruntime so it can be embedded in external software. All required helpers
(`normalize`, `reform3D`) are copied locally to avoid importing IsoNet modules.
By default the script multiplies input and output volumes by -1, mirroring the
original MRC handling; disable with `--no-flip-sign` if unnecessary.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, List, Sequence
import mrcfile
import os

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "onnxruntime is required to run this script. Install it via "
        "`pip install onnxruntime` or `pip install onnxruntime-gpu`."
    ) from exc


def normalize(
    x: np.ndarray,
    percentile: bool = True,
    pmin: float = 4.0,
    pmax: float = 96.0,
    axis=None,
    clip: bool = False,
    eps: float = 1e-20,
) -> np.ndarray:
    """Percentile or z-score normalization (replicated from preprocessing/img_processing.py)."""
    if percentile:
        mi = np.percentile(x, pmin, axis=axis, keepdims=True)
        ma = np.percentile(x, pmax, axis=axis, keepdims=True)
        out = (x - mi) / (ma - mi + eps)
        out = out.astype(np.float32)
        if clip:
            return np.clip(out, 0, 1)
        return out
    out = (x - np.mean(x)) / np.std(x)
    out = out.astype(np.float32)
    return out


class reform3D:
    """Utility to split and stitch 3D volumes (copied from util/toTile.py)."""

    def __init__(self, data3D: np.ndarray, cubesize: int, cropsize: int, edge_depth: int):
        self._sp = np.array(data3D.shape)
        self._orig_data = data3D
        self.cubesize = cubesize
        self.cropsize = cropsize
        self.edge_depth = edge_depth
        self._sidelen = np.ceil((self._sp + edge_depth * 2) / self.cubesize).astype(int)

    def pad_and_crop(self) -> np.ndarray:
        pad_left = int((self.cropsize - self.cubesize) / 2 + self.edge_depth)
        pad_right = (
            self._sidelen * self.cubesize + (self.cropsize - self.cubesize) - pad_left - self._sp
        ).astype(int)
        data = np.pad(
            self._orig_data,
            ((pad_left, pad_right[0]), (pad_left, pad_right[1]), (pad_left, pad_right[2])),
            "symmetric",
        )
        outdata = []
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    cube = data[
                        i * self.cubesize : i * self.cubesize + self.cropsize,
                        j * self.cubesize : j * self.cubesize + self.cropsize,
                        k * self.cubesize : k * self.cubesize + self.cropsize,
                    ]
                    outdata.append(cube)
        return np.array(outdata)

    def mask(self, x_len: int, y_len: int, z_len: int) -> np.ndarray:
        p = 2 * self.edge_depth
        assert x_len > 2 * p
        assert y_len > 2 * p
        assert z_len > 2 * p

        array_x = np.minimum(np.arange(x_len + 1), p) / p
        array_x = array_x * np.flip(array_x)
        array_x = array_x[np.newaxis, np.newaxis, :]

        array_y = np.minimum(np.arange(y_len + 1), p) / p
        array_y = array_y * np.flip(array_y)
        array_y = array_y[np.newaxis, :, np.newaxis]

        array_z = np.minimum(np.arange(z_len + 1), p) / p
        array_z = array_z * np.flip(array_z)
        array_z = array_z[:, np.newaxis, np.newaxis]

        out = array_x * array_y * array_z
        return out[:x_len, :y_len, :z_len]

    def restore(self, cubes: np.ndarray) -> np.ndarray:
        start = (self.cropsize - self.cubesize) // 2 - self.edge_depth
        end = (self.cropsize - self.cubesize) // 2 + self.cubesize + self.edge_depth
        cubes = cubes[:, start:end, start:end, start:end]

        restored = np.zeros(
            (
                self._sidelen[0] * self.cubesize + self.edge_depth * 2,
                self._sidelen[1] * self.cubesize + self.edge_depth * 2,
                self._sidelen[2] * self.cubesize + self.edge_depth * 2,
            )
        )
        mask_cube = self.mask(
            self.cubesize + self.edge_depth * 2,
            self.cubesize + self.edge_depth * 2,
            self.cubesize + self.edge_depth * 2,
        )
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    restored[
                        i * self.cubesize : (i + 1) * self.cubesize + self.edge_depth * 2,
                        j * self.cubesize : (j + 1) * self.cubesize + self.edge_depth * 2,
                        k * self.cubesize : (k + 1) * self.cubesize + self.edge_depth * 2,
                    ] += (
                        cubes[i * self._sidelen[1] * self._sidelen[2] + j * self._sidelen[2] + k]
                        * mask_cube
                    )
        p = self.edge_depth * 2
        restored = restored[p : p + self._sp[0], p : p + self._sp[1], p : p + self._sp[2]]
        return restored


def _parse_providers(provider_spec: str | Sequence[str] | None) -> List[str] | None:
    if provider_spec is None:
        return None
    if isinstance(provider_spec, str):
        candidates: Iterable[str] = (item.strip() for item in provider_spec.split(","))
    else:
        candidates = (str(item).strip() for item in provider_spec)
    providers = [item for item in candidates if item]
    return providers or None


def _load_session(model_path: pathlib.Path, providers: Sequence[str] | None) -> ort.InferenceSession:
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
    return ort.InferenceSession(str(model_path), sess_options=session_opts, providers=provider_chain)


def predict_volume(
    volume: np.ndarray,
    model_path: pathlib.Path,
    cube_size: int = 64,
    crop_size: int = 96,
    batch_size: int = 1,
    normalize_percentile: bool = True,
    edge_depth: int = 9,
    flip_sign: bool = True,
    providers: Sequence[str] | None = None,
) -> np.ndarray:
    """Run ONNX inference on a 3D volume and return the corrected volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {volume.shape}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    volume = np.asarray(volume, dtype=np.float32)
    if flip_sign:
        volume = -volume
    data = normalize(volume, percentile=normalize_percentile)

    reform = reform3D(data, cube_size, crop_size, edge_depth)
    cubes = reform.pad_and_crop().astype(np.float32)
    cubes = cubes[..., np.newaxis]

    session = _load_session(model_path, providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    num_patches = cubes.shape[0]
    remainder = num_patches % batch_size
    if remainder:
        pad_len = batch_size - remainder
        cubes = np.concatenate([cubes, cubes[:pad_len]], axis=0)
    else:
        pad_len = 0

    predictions = np.zeros(cubes.shape[:-1], dtype=np.float32)
    num_batches = cubes.shape[0] // batch_size
    for idx in range(num_batches):
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

    restored = reform.restore(predictions)
    restored = normalize(restored, percentile=normalize_percentile)
    if flip_sign:
        restored = -restored
    return restored


def _load_array(path: pathlib.Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "arr_0" in data:
            return data["arr_0"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError("Multiple arrays in npz file; please provide a single-array file.")
    raise ValueError(f"Unsupported array format for {path}. Use .npy or .npz.")


def _save_array(path: pathlib.Path, array: np.ndarray) -> None:
    if path.suffix == ".npy":
        np.save(path, array)
    elif path.suffix == ".npz":
        np.savez_compressed(path, array)
    else:
        raise ValueError(f"Unsupported output format for {path}. Use .npy or .npz.")


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Run IsoNet ONNX prediction on a 3D NumPy array.")
#     parser.add_argument("--model", required=True, help="Path to the exported ONNX model.")
#     parser.add_argument("--input", required=True, help="Input volume (.npy or .npz) containing a 3D array.")
#     parser.add_argument("--output", required=True, help="Output path (.npy or .npz) to store the corrected volume.")
#     parser.add_argument("--cube-size", type=int, default=64, help="Cube size used during training (default: 64).")
#     parser.add_argument("--crop-size", type=int, default=96, help="Crop size used during training (default: 96).")
#     parser.add_argument("--batch-size", type=int, default=1, help="Number of patches per ONNX inference batch.")
#     parser.add_argument(
#         "--edge-depth",
#         type=int,
#         default=9,
#         help="Edge merging depth when tiling cubes (default: 9, matches IsoNet).",
#     )
#     parser.add_argument(
#         "--no-percentile",
#         action="store_true",
#         help="Disable percentile normalization (use z-score instead).",
#     )
#     parser.add_argument(
#         "--providers",
#         default=None,
#         help="Comma-separated ONNXRuntime providers, e.g. 'CUDAExecutionProvider,CPUExecutionProvider'.",
#     )
#     parser.add_argument(
#         "--no-flip-sign",
#         action="store_true",
#         help="Do not multiply the input/output by -1 (TensorFlow pipeline defaults to flipping).",
#     )
#     return parser.parse_args()

def get_tomo(path ,return_spacing=False):
    """
    Load a 3D MRC file as a numpy array.

    Parameters:
    - path: str
        Path to the MRC file.

    Returns:
    - data: ndarray
        The 3D data loaded from the MRC file.
    """
    with mrcfile.open(path) as mrc:
        data = mrc.data
        voxel_size = mrc.voxel_size
        
    if return_spacing:
        return data, voxel_size
    else:
        return data

def save_tomo(data, path, voxel_size=17.14, data_type=np.int8):
    """
    Save a 3D numpy array as an MRC file.

    Parameters:
    - data: ndarray
        The 3D data to save.
    - voxel_size: float
        The voxel size of the data.
    """
        # 获取文件所在的目录
    directory = os.path.dirname(path)
    
    # 如果目录不存在，则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with mrcfile.new(path, overwrite=True) as mrc:
        data = data.astype(data_type)
        mrc.set_data(data)
        mrc.voxel_size = voxel_size

def main():
    # args = parse_args()
    # model_path = pathlib.Path(args.model).expanduser().resolve()
    # if not model_path.exists():
    #     raise FileNotFoundError(model_path)
    # input_path = pathlib.Path(args.input).expanduser().resolve()
    # output_path = pathlib.Path(args.output).expanduser().resolve()
    

    volume = get_tomo('/media/liushuo/data3/lwd/pp052/synapse_seg/isonet/tomo_deconv/pp052_wbp_resample.mrc')
    providers = ['CUDAExecutionProvider',]
    corrected = predict_volume(
        volume,
        model_path='/home/liushuo/Documents/code/SynapseSeg/pretrained/vesicle_corrected_model.onnx',
        cube_size=64,
        crop_size=96,
        batch_size=1,
        # normalize_percentile=not args.no_percentile,
        edge_depth=9,
        # flip_sign=not args.no_flip_sign,
        providers=providers,
    )
    save_tomo(corrected, '/media/liushuo/data3/lwd/pp052/synapse_seg/isonet/tomo_deconv/pp052_wbp_resample_corrected.onnx.mrc', voxel_size=17.14, data_type=np.float32)


if __name__ == "__main__":
    main()

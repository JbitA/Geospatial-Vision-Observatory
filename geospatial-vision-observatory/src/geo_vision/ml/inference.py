from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .integrity import sha256_file, verify_bundle
from .schema import CLASS_NAMES, CLASS_RGB, INPUT_BANDS, WORLDCOVER_CODES

MAX_INFERENCE_PIXELS = 100_000_000


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"model bundle {label} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"model bundle {label} must be a JSON object")
    return value


def _runtime_contract(
    bundle_dir: Path, bundle: dict[str, Any]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], int]:
    """Validate support-file semantics after byte-level bundle verification."""

    if bundle.get("task") != "worldcover_landcover_segmentation":
        raise ValueError("model bundle task is incompatible with the runtime")
    bundle_bands = bundle.get("input_bands")
    bundle_classes = bundle.get("class_names")
    if not isinstance(bundle_bands, list) or tuple(bundle_bands) != INPUT_BANDS:
        raise ValueError("model bundle input bands are incompatible with the runtime")
    if not isinstance(bundle_classes, list) or tuple(bundle_classes) != CLASS_NAMES:
        raise ValueError("model bundle class names are incompatible with the runtime")

    normalization = _json_object(bundle_dir / "normalization.json", "normalization")
    classes = _json_object(bundle_dir / "classes.json", "class metadata")
    training_config = _json_object(bundle_dir / "training-config.json", "training configuration")
    normalization_bands = normalization.get("bands")
    class_names = classes.get("class_names")
    worldcover_codes = classes.get("worldcover_codes")
    if not isinstance(normalization_bands, list) or tuple(normalization_bands) != INPUT_BANDS:
        raise ValueError("model bundle band order is incompatible with the runtime")
    if (
        not isinstance(class_names, list)
        or tuple(class_names) != CLASS_NAMES
        or not isinstance(worldcover_codes, list)
        or tuple(worldcover_codes) != WORLDCOVER_CODES
    ):
        raise ValueError("model bundle class metadata is incompatible with the runtime")

    try:
        mean = np.asarray(normalization.get("mean"), dtype=np.float32)[:, None, None]
        std = np.asarray(normalization.get("std"), dtype=np.float32)[:, None, None]
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("model bundle normalization shape is invalid") from error
    if mean.shape != (len(INPUT_BANDS), 1, 1) or std.shape != mean.shape:
        raise ValueError("model bundle normalization shape is invalid")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("model bundle normalization values are invalid")

    patch_size = training_config.get("patch_size")
    if not isinstance(patch_size, int) or isinstance(patch_size, bool) or not 32 <= patch_size <= 2048:
        raise ValueError("model bundle patch size is invalid")
    return mean, std, patch_size


def _block_size(length: int) -> int:
    if length < 16:
        return 0
    return min(256, max(16, (min(length, 256) // 16) * 16))


def predict_scene(
    stack_path: Path,
    bundle_dir: Path,
    output_path: Path,
    *,
    device: str = "cpu",
    tile_size: int | None = None,
) -> dict[str, Any]:
    """Run bounded-memory, hash-verified inference over a six-band GeoTIFF.

    The raster is read and written tile-by-tile so inference memory is O(tile_size²), not O(scene).
    The output is published atomically only after the complete prediction is written.
    """

    import rasterio  # type: ignore[import-untyped]
    import torch
    from rasterio.windows import Window  # type: ignore[import-untyped]

    bundle = verify_bundle(bundle_dir)
    mean, std, exported_tile_size = _runtime_contract(bundle_dir, bundle)
    if tile_size is None:
        tile_size = exported_tile_size
    if tile_size != exported_tile_size or tile_size < 32 or tile_size > 2048:
        raise ValueError(f"exported model requires tile_size={exported_tile_size}")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("unsupported inference device")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS requested but unavailable")

    model_file = str(bundle["model_file"])
    # PyTorch currently emits a non-actionable read-only-buffer warning while loading a verified
    # PT2 archive. Suppress only that exact implementation warning; all other load warnings surface.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The given buffer is not writable.*",
            category=UserWarning,
            module=r"torch\.export\.pt2_archive\._package",
        )
        exported = torch.export.load(str(bundle_dir / model_file))
    model = exported.module().to(device)

    if stack_path.resolve() == output_path.resolve():
        raise ValueError("prediction output must not overwrite the source stack")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()

    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    valid_pixels = 0
    confidence_sum = 0.0
    try:
        with rasterio.open(stack_path) as src:
            if src.count != len(INPUT_BANDS):
                raise ValueError(
                    "input raster must contain six Sentinel-2 bands in the documented order"
                )
            descriptions = tuple(src.descriptions)
            if any(descriptions) and descriptions != INPUT_BANDS:
                raise ValueError("input raster band descriptions do not match the required order")
            if src.crs is None:
                raise ValueError("input raster must include a coordinate reference system")
            height, width = int(src.height), int(src.width)
            if height <= 0 or width <= 0:
                raise ValueError("input raster dimensions are invalid")
            if height * width > MAX_INFERENCE_PIXELS:
                raise ValueError("input raster exceeds the configured inference pixel ceiling")
            profile = src.profile.copy()
            nodata = src.nodata
            profile.update(count=2, dtype="float32", nodata=-9999.0, compress="deflate")
            block_x, block_y = _block_size(width), _block_size(height)
            if block_x and block_y:
                profile.update(tiled=True, blockxsize=block_x, blockysize=block_y)
            else:
                profile.pop("tiled", None)
                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)

            with rasterio.open(temporary_path, "w", **profile) as dst, torch.inference_mode():
                dst.set_band_description(1, "landcover_class_index")
                dst.set_band_description(2, "prediction_confidence")
                for y in range(0, height, tile_size):
                    for x in range(0, width, tile_size):
                        tile_h = min(tile_size, height - y)
                        tile_w = min(tile_size, width - x)
                        window = Window(x, y, tile_w, tile_h)
                        tile: npt.NDArray[np.float32] = np.asarray(
                            src.read(window=window), dtype=np.float32
                        )
                        invalid = ~np.all(np.isfinite(tile), axis=0)
                        if nodata is not None:
                            invalid |= np.any(tile == nodata, axis=0)
                        invalid |= ~np.all((tile > -0.25) & (tile < 2.0), axis=0)
                        normalized: npt.NDArray[np.float32] = np.nan_to_num(
                            (tile - mean) / std, nan=0.0, posinf=0.0, neginf=0.0
                        ).astype(np.float32, copy=False)
                        pad_y, pad_x = tile_size - tile_h, tile_size - tile_w
                        if pad_y or pad_x:
                            mode: Literal["reflect", "edge"] = (
                                "reflect" if tile_h > 1 and tile_w > 1 else "edge"
                            )
                            normalized = np.pad(
                                normalized, ((0, 0), (0, pad_y), (0, pad_x)), mode=mode
                            )
                        tensor = torch.from_numpy(np.ascontiguousarray(normalized[None])).to(device)
                        logits = model(tensor)
                        expected_shape = (1, len(CLASS_NAMES), tile_size, tile_size)
                        if tuple(logits.shape) != expected_shape:
                            raise RuntimeError(
                                f"exported model returned {tuple(logits.shape)}, expected {expected_shape}"
                            )
                        probability = torch.softmax(logits, dim=1)[0]
                        conf, pred = probability.max(dim=0)
                        pred_np = pred[:tile_h, :tile_w].cpu().numpy().astype(np.uint8)
                        conf_np = conf[:tile_h, :tile_w].cpu().numpy().astype(np.float32)
                        pred_np[invalid] = 255
                        conf_np[invalid] = np.nan

                        valid = ~invalid
                        if valid.any():
                            values = pred_np[valid]
                            class_counts += np.bincount(
                                values, minlength=len(CLASS_NAMES)
                            )[: len(CLASS_NAMES)]
                            valid_pixels += int(valid.sum())
                            confidence_sum += float(np.nansum(conf_np[valid], dtype=np.float64))
                        output = np.stack(
                            (
                                np.where(pred_np == 255, -9999.0, pred_np).astype(np.float32),
                                np.where(np.isfinite(conf_np), conf_np, -9999.0).astype(np.float32),
                            )
                        )
                        dst.write(output, window=window)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "class_fractions": {
            name: (float(class_counts[index] / valid_pixels) if valid_pixels else None)
            for index, name in enumerate(CLASS_NAMES)
        },
        "mean_confidence": confidence_sum / valid_pixels if valid_pixels else None,
        "valid_pixels": valid_pixels,
        "prediction_sha256": sha256_file(output_path),
    }


def colorize_prediction(
    prediction: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint8]:
    rgb = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for index, color in enumerate(CLASS_RGB):
        rgb[prediction == index] = color
    return rgb


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run verified six-band land-cover inference")
    parser.add_argument("stack", type=Path)
    parser.add_argument("--bundle", type=Path, default=Path("models/landcover"))
    parser.add_argument("--output", type=Path, default=Path("reports/prediction.tif"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    result = predict_scene(args.stack, args.bundle, args.output, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

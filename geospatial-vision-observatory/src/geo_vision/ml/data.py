from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .schema import CODE_TO_INDEX, IGNORE_INDEX, INPUT_BANDS

SCL_EXCLUDED = frozenset({0, 1, 3, 7, 8, 9, 10})


@dataclass(frozen=True)
class Scene:
    key: str
    image: npt.NDArray[np.float32]
    target: npt.NDArray[np.uint8]
    valid: npt.NDArray[np.bool_]
    source_manifest: dict[str, Any]
    transform: Any
    crs: Any


@dataclass(frozen=True)
class PatchRef:
    scene_index: int
    y: int
    x: int
    valid_fraction: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_worldcover(labels: npt.NDArray[np.integer[Any]]) -> npt.NDArray[np.uint8]:
    mapped = np.full(labels.shape, IGNORE_INDEX, dtype=np.uint8)
    for code, index in CODE_TO_INDEX.items():
        mapped[labels == code] = index
    return mapped


def _verify_manifest_file(directory: Path, manifest: dict[str, Any], name: str) -> None:
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError(f"{directory.name}: curated manifest has no integrity section")
    outputs = integrity.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{directory.name}: curated manifest has no output hashes")
    metadata = outputs.get(name)
    if not isinstance(metadata, dict):
        raise ValueError(f"{directory.name}: manifest does not hash required output {name}")
    expected = metadata.get("sha256")
    expected_bytes = metadata.get("bytes")
    path = directory / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{directory.name}: required curated output is missing: {name}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{directory.name}: invalid SHA-256 metadata for {name}")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        raise ValueError(f"{directory.name}: invalid byte-count metadata for {name}")
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected:
        raise ValueError(f"{directory.name}: curated output integrity mismatch: {name}")


def load_scene(directory: Path) -> Scene:
    try:
        import rasterio  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the 'ml' extra to load GeoTIFF training scenes") from error

    stack_path = directory / "sentinel2_multispectral.tif"
    labels_path = directory / "worldcover_2021_on_sentinel.tif"
    manifest_path = directory / "manifest.json"
    if not (stack_path.is_file() and labels_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError(f"incomplete curated scene: {directory}")

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 3:
        raise ValueError(f"{directory.name}: unsupported curated manifest schema")
    aoi = manifest.get("aoi")
    if not isinstance(aoi, dict) or aoi.get("key") != directory.name:
        raise ValueError(f"{directory.name}: manifest AOI identity does not match its directory")
    _verify_manifest_file(directory, manifest, stack_path.name)
    _verify_manifest_file(directory, manifest, labels_path.name)
    with rasterio.open(stack_path) as src:
        image = src.read().astype(np.float32)
        transform, crs = src.transform, src.crs
        nodata = src.nodata
        descriptions = src.descriptions
    if image.shape[0] != len(INPUT_BANDS):
        raise ValueError(f"{directory.name}: expected {len(INPUT_BANDS)} bands, got {image.shape[0]}")
    if any(descriptions) and tuple(descriptions) != INPUT_BANDS:
        raise ValueError(f"{directory.name}: multispectral band descriptions/order are invalid")
    if crs is None:
        raise ValueError(f"{directory.name}: multispectral raster is missing a CRS")
    if nodata is not None:
        image[image == nodata] = np.nan
    with rasterio.open(labels_path) as src:
        raw_labels = src.read(1)
        label_transform, label_crs = src.transform, src.crs
    if raw_labels.shape != image.shape[1:] or label_crs != crs or label_transform != transform:
        raise ValueError(f"{directory.name}: WorldCover labels are not aligned to the Sentinel grid")
    target = map_worldcover(raw_labels)

    valid = np.all(np.isfinite(image), axis=0) & (target != IGNORE_INDEX)
    scl_path = directory / "sentinel2_scl.tif"
    if scl_path.is_file():
        _verify_manifest_file(directory, manifest, scl_path.name)
        with rasterio.open(scl_path) as src:
            scl = src.read(1)
            scl_transform, scl_crs = src.transform, src.crs
        if scl.shape != image.shape[1:] or scl_crs != crs or scl_transform != transform:
            raise ValueError(f"{directory.name}: SCL raster is not aligned to the Sentinel grid")
        valid &= ~np.isin(scl, tuple(SCL_EXCLUDED))
    # Guard against corrupt/calibration-outlier reflectance while preserving small negatives.
    valid &= np.all((image > -0.25) & (image < 2.0), axis=0)
    target = target.copy()
    target[~valid] = IGNORE_INDEX
    return Scene(directory.name, image, target, valid, manifest, transform, crs)


def scene_fingerprint(directory: Path) -> dict[str, str]:
    names = (
        "sentinel2_multispectral.tif",
        "worldcover_2021_on_sentinel.tif",
        "sentinel2_scl.tif",
        "manifest.json",
    )
    return {name: file_sha256(directory / name) for name in names if (directory / name).is_file()}


def patch_index(
    scenes: list[Scene],
    *,
    patch_size: int,
    stride: int,
    min_valid_fraction: float = 0.75,
) -> list[PatchRef]:
    if patch_size < 32 or stride < 16:
        raise ValueError("patch_size >= 32 and stride >= 16 are required")
    refs: list[PatchRef] = []
    for scene_index, scene in enumerate(scenes):
        height, width = scene.target.shape
        if height < patch_size or width < patch_size:
            continue
        ys = list(range(0, height - patch_size + 1, stride))
        xs = list(range(0, width - patch_size + 1, stride))
        if ys[-1] != height - patch_size:
            ys.append(height - patch_size)
        if xs[-1] != width - patch_size:
            xs.append(width - patch_size)
        for y in ys:
            for x in xs:
                valid_fraction = float(
                    scene.valid[y : y + patch_size, x : x + patch_size].mean()
                )
                if valid_fraction >= min_valid_fraction:
                    refs.append(PatchRef(scene_index, y, x, valid_fraction))
    if not refs:
        raise ValueError("no training patches passed the valid-pixel threshold")
    return refs


def compute_normalization(
    scenes: list[Scene], *, sample_pixels_per_scene: int = 200_000, seed: int = 20260823
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    rng = np.random.default_rng(seed)
    samples: list[npt.NDArray[np.float32]] = []
    for scene in scenes:
        locations = np.flatnonzero(scene.valid.ravel())
        if locations.size == 0:
            continue
        if locations.size > sample_pixels_per_scene:
            locations = rng.choice(locations, sample_pixels_per_scene, replace=False)
        flattened = scene.image.reshape(scene.image.shape[0], -1)[:, locations]
        samples.append(flattened)
    if not samples:
        raise ValueError("no valid training pixels available for normalization")
    merged = np.concatenate(samples, axis=1)
    mean = np.nanmean(merged, axis=1).astype(np.float32)
    std = np.nanstd(merged, axis=1).astype(np.float32)
    std = np.maximum(std, 1e-4).astype(np.float32)
    return mean, std


def class_weights(scenes: list[Scene], max_weight: float = 5.0) -> npt.NDArray[np.float32]:
    counts = np.zeros(len(CODE_TO_INDEX), dtype=np.float64)
    for scene in scenes:
        values = scene.target[scene.target != IGNORE_INDEX]
        counts += np.bincount(values, minlength=len(counts))[: len(counts)]
    present = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    if present.any():
        frequencies = counts[present] / counts[present].sum()
        median = float(np.median(frequencies))
        weights[present] = np.clip(median / np.maximum(frequencies, 1e-12), 0.25, max_weight)
    return weights.astype(np.float32)


def dataset_signature(scene_dirs: list[Path]) -> str:
    payload = {"scenes": {path.name: scene_fingerprint(path) for path in sorted(scene_dirs)}}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def experiment_signature(dataset_sha256: str, training_config: dict[str, Any]) -> str:
    payload = {"dataset_signature": dataset_sha256, "training_config": training_config}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

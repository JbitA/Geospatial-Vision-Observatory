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
class LazyScene:
    """Validated raster-backed scene whose pixel arrays are read only by requested window."""

    key: str
    directory: Path
    width: int
    height: int
    source_manifest: dict[str, Any]
    transform: Any
    crs: Any
    nodata: float | int | None
    has_scl: bool

    def read_patch(
        self, y: int, x: int, height: int, width: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        return _read_lazy_window(self, y=y, x=x, height=height, width=width)


SceneLike = Scene | LazyScene


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
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {3, 4}:
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


def load_lazy_scene(directory: Path) -> LazyScene:
    """Validate a curated scene without materializing its rasters in memory."""

    try:
        import rasterio  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the 'ml' extra to load GeoTIFF training scenes") from error

    stack_path = directory / "sentinel2_multispectral.tif"
    labels_path = directory / "worldcover_2021_on_sentinel.tif"
    manifest_path = directory / "manifest.json"
    if not (stack_path.is_file() and labels_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError(f"incomplete curated scene: {directory}")
    manifest_raw = json.loads(manifest_path.read_text())
    if not isinstance(manifest_raw, dict) or manifest_raw.get("schema_version") not in {3, 4}:
        raise ValueError(f"{directory.name}: unsupported curated manifest schema")
    manifest: dict[str, Any] = manifest_raw
    aoi = manifest.get("aoi")
    if not isinstance(aoi, dict) or aoi.get("key") != directory.name:
        raise ValueError(f"{directory.name}: manifest AOI identity does not match its directory")
    _verify_manifest_file(directory, manifest, stack_path.name)
    _verify_manifest_file(directory, manifest, labels_path.name)

    with rasterio.open(stack_path) as src:
        if src.count != len(INPUT_BANDS):
            raise ValueError(f"{directory.name}: expected {len(INPUT_BANDS)} bands, got {src.count}")
        descriptions = tuple(src.descriptions)
        if any(descriptions) and descriptions != INPUT_BANDS:
            raise ValueError(f"{directory.name}: multispectral band descriptions/order are invalid")
        if src.crs is None:
            raise ValueError(f"{directory.name}: multispectral raster is missing a CRS")
        width, height = int(src.width), int(src.height)
        transform, crs, nodata = src.transform, src.crs, src.nodata
    with rasterio.open(labels_path) as src:
        if (int(src.width), int(src.height)) != (width, height) or src.crs != crs or src.transform != transform:
            raise ValueError(f"{directory.name}: WorldCover labels are not aligned to the Sentinel grid")

    scl_path = directory / "sentinel2_scl.tif"
    has_scl = scl_path.is_file()
    if has_scl:
        _verify_manifest_file(directory, manifest, scl_path.name)
        with rasterio.open(scl_path) as src:
            if (int(src.width), int(src.height)) != (width, height) or src.crs != crs or src.transform != transform:
                raise ValueError(f"{directory.name}: SCL raster is not aligned to the Sentinel grid")

    return LazyScene(
        key=directory.name,
        directory=directory,
        width=width,
        height=height,
        source_manifest=manifest,
        transform=transform,
        crs=crs,
        nodata=nodata,
        has_scl=has_scl,
    )


def _read_lazy_window(
    scene: LazyScene, *, y: int, x: int, height: int, width: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
    try:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.windows import Window  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the 'ml' extra to load GeoTIFF training scenes") from error
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (y, x, height, width)):
        raise TypeError("window coordinates and dimensions must be integers")
    if y < 0 or x < 0 or height <= 0 or width <= 0 or y + height > scene.height or x + width > scene.width:
        raise ValueError("requested scene window is outside raster bounds")
    window = Window(x, y, width, height)
    with rasterio.open(scene.directory / "sentinel2_multispectral.tif") as src:
        image = np.asarray(src.read(window=window), dtype=np.float32)
    if scene.nodata is not None:
        image[image == scene.nodata] = np.nan
    with rasterio.open(scene.directory / "worldcover_2021_on_sentinel.tif") as src:
        raw_labels = src.read(1, window=window)
    target = map_worldcover(raw_labels)
    valid = np.all(np.isfinite(image), axis=0) & (target != IGNORE_INDEX)
    if scene.has_scl:
        with rasterio.open(scene.directory / "sentinel2_scl.tif") as src:
            scl = src.read(1, window=window)
        valid &= ~np.isin(scl, tuple(SCL_EXCLUDED))
    valid &= np.all((image > -0.25) & (image < 2.0), axis=0)
    target = target.copy()
    target[~valid] = IGNORE_INDEX
    return image, target, valid


def _iter_lazy_blocks(
    scene: LazyScene, *, block_size: int = 512
):
    for y in range(0, scene.height, block_size):
        height = min(block_size, scene.height - y)
        for x in range(0, scene.width, block_size):
            width = min(block_size, scene.width - x)
            yield _read_lazy_window(scene, y=y, x=x, height=height, width=width)


def scene_fingerprint(directory: Path) -> dict[str, str]:
    names = (
        "sentinel2_multispectral.tif",
        "worldcover_2021_on_sentinel.tif",
        "sentinel2_scl.tif",
        "manifest.json",
    )
    return {name: file_sha256(directory / name) for name in names if (directory / name).is_file()}


def patch_index(
    scenes: list[SceneLike],
    *,
    patch_size: int,
    stride: int,
    min_valid_fraction: float = 0.75,
) -> list[PatchRef]:
    if patch_size < 32 or stride < 16:
        raise ValueError("patch_size >= 32 and stride >= 16 are required")
    refs: list[PatchRef] = []
    for scene_index, scene in enumerate(scenes):
        if isinstance(scene, Scene):
            height, width = scene.target.shape
        else:
            height, width = scene.height, scene.width
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
                if isinstance(scene, Scene):
                    valid = scene.valid[y : y + patch_size, x : x + patch_size]
                else:
                    _, _, valid = scene.read_patch(y, x, patch_size, patch_size)
                valid_fraction = float(valid.mean())
                if valid_fraction >= min_valid_fraction:
                    refs.append(PatchRef(scene_index, y, x, valid_fraction))
    if not refs:
        raise ValueError("no training patches passed the valid-pixel threshold")
    return refs


def compute_normalization(
    scenes: list[SceneLike], *, sample_pixels_per_scene: int = 200_000, seed: int = 20260823
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if sample_pixels_per_scene <= 0:
        raise ValueError("sample_pixels_per_scene must be positive")
    rng = np.random.default_rng(seed)
    samples: list[npt.NDArray[np.float32]] = []
    for scene in scenes:
        if isinstance(scene, Scene):
            locations = np.flatnonzero(scene.valid.ravel())
            if locations.size == 0:
                continue
            if locations.size > sample_pixels_per_scene:
                locations = rng.choice(locations, sample_pixels_per_scene, replace=False)
            samples.append(scene.image.reshape(scene.image.shape[0], -1)[:, locations])
            continue

        # Preserve the eager sampler exactly without materializing the AOI-sized valid mask.
        # Pass 1 counts valid pixels. If sampling is needed, choose the same ranks that
        # ``rng.choice(flat_valid_locations, ...)`` would choose; pass 2 collects only those
        # ranks from bounded raster windows. This keeps memory O(sample_pixels_per_scene) and
        # keeps normalization reproducible across eager and lazy access modes.
        valid_count = 0
        for _image, _target, valid in _iter_lazy_blocks(scene):
            valid_count += int(np.count_nonzero(valid))
        if valid_count == 0:
            continue

        sample_count = min(valid_count, sample_pixels_per_scene)
        if valid_count > sample_pixels_per_scene:
            selected_ranks = rng.choice(valid_count, sample_count, replace=False)
            sort_order = np.argsort(selected_ranks)
            sorted_ranks = selected_ranks[sort_order]
            retained_values = np.empty((len(INPUT_BANDS), sample_count), dtype=np.float32)
            valid_offset = 0
            for image, _target, valid in _iter_lazy_blocks(scene):
                values = image.reshape(image.shape[0], -1)[:, valid.ravel()]
                block_count = values.shape[1]
                if block_count == 0:
                    continue
                lo = int(np.searchsorted(sorted_ranks, valid_offset, side="left"))
                hi = int(np.searchsorted(sorted_ranks, valid_offset + block_count, side="left"))
                if lo < hi:
                    local_ranks = sorted_ranks[lo:hi] - valid_offset
                    retained_values[:, sort_order[lo:hi]] = values[:, local_ranks]
                valid_offset += block_count
            samples.append(retained_values)
            continue

        retained_blocks: list[npt.NDArray[np.float32]] = []
        for image, _target, valid in _iter_lazy_blocks(scene):
            values = image.reshape(image.shape[0], -1)[:, valid.ravel()]
            if values.shape[1]:
                retained_blocks.append(values)
        samples.append(np.concatenate(retained_blocks, axis=1))
    if not samples:
        raise ValueError("no valid training pixels available for normalization")
    merged = np.concatenate(samples, axis=1)
    # Accumulate normalization statistics in float64 so identical pixel sets produce
    # stable results regardless of raster-window/array memory layout. The persisted
    # model contract remains float32.
    mean = np.nanmean(merged, axis=1, dtype=np.float64).astype(np.float32)
    std = np.nanstd(merged, axis=1, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4).astype(np.float32)
    return mean, std


def class_weights(scenes: list[SceneLike], max_weight: float = 5.0) -> npt.NDArray[np.float32]:
    counts = np.zeros(len(CODE_TO_INDEX), dtype=np.float64)
    for scene in scenes:
        if isinstance(scene, Scene):
            values = scene.target[scene.target != IGNORE_INDEX]
            counts += np.bincount(values, minlength=len(counts))[: len(counts)]
            continue
        for _image, target, valid in _iter_lazy_blocks(scene):
            values = target[valid]
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

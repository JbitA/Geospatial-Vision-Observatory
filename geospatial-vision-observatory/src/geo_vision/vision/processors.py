from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image


class Processor(Protocol):
    name: str
    version: str

    def process(self, image_path: Path, previous_path: Path | None = None) -> dict[str, object]: ...


def load_rgb(path: Path, max_side: int = 1024) -> npt.NDArray[np.uint8]:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        converted.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(converted, dtype=np.uint8)


class QualityControl:
    """Deterministic guardrail; not presented as deep learning."""

    name = "quality_control"
    version = "1.0.0"

    def process(self, image_path: Path, previous_path: Path | None = None) -> dict[str, object]:
        rgb = load_rgb(image_path)
        gray = rgb.astype(np.float32).mean(axis=2)
        clipped = float(np.mean((gray <= 2) | (gray >= 253)))
        # Mean squared gradient is a transparent sharpness proxy.
        gradient = float(np.mean(np.diff(gray, axis=0) ** 2) + np.mean(np.diff(gray, axis=1) ** 2))
        return {
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "mean_luminance": round(float(gray.mean()), 4),
            "std_luminance": round(float(gray.std()), 4),
            "clipped_fraction": round(clipped, 6),
            "gradient_energy": round(gradient, 4),
            "passed": bool(gradient > 1.0 and clipped < 0.95),
            "method_class": "deterministic_qc",
        }


class GeospatialRgbBaseline:
    """Transparent RGB land-surface proxies for operational previews.

    These measurements are deliberately named proxies because true vegetation indices
    such as NDVI require NIR. Analysis-grade land-cover metrics come from aligned
    Sentinel-2 COG + WorldCover/Hansen datasets prepared by the geospatial tooling.
    """

    name = "geospatial_rgb_baseline"
    version = "1.0.0"

    def process(self, image_path: Path, previous_path: Path | None = None) -> dict[str, object]:
        rgb = load_rgb(image_path)
        scaled = rgb.astype(np.float32) / 255.0
        red, green, blue = scaled[:, :, 0], scaled[:, :, 1], scaled[:, :, 2]
        maximum, minimum = scaled.max(axis=2), scaled.min(axis=2)
        saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
        luminance = scaled.mean(axis=2)

        excess_green = 2.0 * green - red - blue
        vegetation_proxy = (excess_green > 0.08) & (green > 0.20)
        water_proxy = (blue > red * 1.12) & (blue > green * 1.04) & (luminance < 0.68)

        gray = luminance
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        edge = np.sqrt(gx * gx + gy * gy)
        built_proxy = (saturation < 0.22) & (luminance > 0.28) & (edge > 0.035)

        return {
            "vegetation_rgb_proxy_fraction": round(float(vegetation_proxy.mean()), 6),
            "water_rgb_proxy_fraction": round(float(water_proxy.mean()), 6),
            "built_surface_rgb_proxy_fraction": round(float(built_proxy.mean()), 6),
            "mean_excess_green": round(float(excess_green.mean()), 6),
            "mean_saturation": round(float(saturation.mean()), 6),
            "edge_density_at_0_035": round(float((edge > 0.035).mean()), 6),
            "method_class": "descriptive_rgb_land_surface_proxy",
            "warning": (
                "RGB proxies are operational descriptors, not land-cover labels or NDVI; "
                "use the WorldCover/Hansen aligned benchmark for scientific evaluation"
            ),
        }


class TemporalChangeBaseline:
    name = "temporal_change_baseline"
    version = "1.0.0"

    def process(self, image_path: Path, previous_path: Path | None = None) -> dict[str, object]:
        if previous_path is None:
            return {
                "available": False,
                "reason": "no_previous_frame",
                "method_class": "pixel_baseline",
            }
        current = load_rgb(image_path, 512).astype(np.float32) / 255.0
        previous = load_rgb(previous_path, 512).astype(np.float32) / 255.0
        if current.shape != previous.shape:
            previous = (
                np.asarray(
                    Image.open(previous_path)
                    .convert("RGB")
                    .resize((current.shape[1], current.shape[0])),
                    dtype=np.float32,
                )
                / 255.0
            )
        delta = np.abs(current - previous).mean(axis=2)
        return {
            "available": True,
            "mean_absolute_change": round(float(delta.mean()), 6),
            "p95_absolute_change": round(float(np.quantile(delta, 0.95)), 6),
            "changed_fraction_at_0_15": round(float((delta > 0.15).mean()), 6),
            "method_class": "pixel_baseline",
            "warning": "unregistered preview change; do not interpret as physical surface motion",
        }



def timed_process(
    processor: Processor, path: Path, previous: Path | None
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    result = processor.process(path, previous)
    return result, (time.perf_counter() - started) * 1_000

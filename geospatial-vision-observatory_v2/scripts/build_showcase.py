#!/usr/bin/env python3
"""Generate measured GitHub showcase artifacts from a verified trained model bundle."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geo_vision.ml.data import map_worldcover
from geo_vision.ml.inference import colorize_prediction, predict_scene, verify_bundle
from geo_vision.ml.schema import CLASS_NAMES, SHOWCASE_SPLIT


def _safe(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.3f}"


def _svg_bar_chart(values: list[float | None], names: tuple[str, ...], path: Path) -> None:
    width, height = 960, 420
    left, top, chart_h = 220, 40, 330
    bar_h = chart_h / len(names)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="20" font-weight="700">External-test IoU by class</text>',
    ]
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        y = top + index * bar_h
        pieces.append(
            f'<text x="10" y="{y + bar_h * 0.68:.1f}" font-family="sans-serif" font-size="13">{html.escape(name.replace("_", " "))}</text>'
        )
        if value is not None and math.isfinite(value):
            bar_w = max(0.0, min(1.0, value)) * (width - left - 70)
            pieces.append(
                f'<rect x="{left}" y="{y + 4:.1f}" width="{bar_w:.1f}" height="{bar_h - 8:.1f}" rx="3" fill="#2563eb"/>'
            )
            pieces.append(
                f'<text x="{left + bar_w + 8:.1f}" y="{y + bar_h * 0.68:.1f}" font-family="monospace" font-size="12">{value:.3f}</text>'
            )
        else:
            pieces.append(
                f'<text x="{left + 8}" y="{y + bar_h * 0.68:.1f}" font-family="monospace" font-size="12">not present</text>'
            )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n")


def _svg_history(history: list[dict[str, float]], path: Path) -> None:
    width, height = 900, 360
    left, right, top, bottom = 60, 30, 45, 55
    chart_w, chart_h = width - left - right, height - top - bottom
    points = [float(row["validation_macro_iou"]) for row in history]
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="20" font-weight="700">Validation macro IoU during training</text>',
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(f'<text x="15" y="{y+4:.1f}" font-family="monospace" font-size="11">{value:.1f}</text>')
    if points:
        coords = []
        for index, value in enumerate(points):
            x = left + (chart_w * index / max(len(points) - 1, 1))
            y = top + chart_h * (1 - max(0.0, min(1.0, value)))
            coords.append(f"{x:.1f},{y:.1f}")
        pieces.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="#2563eb" stroke-width="3"/>')
    pieces.append(f'<text x="{width/2-45:.1f}" y="{height-15}" font-family="sans-serif" font-size="12">training epoch</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n")


def _triptych(scene_dir: Path, prediction_tif: Path, destination: Path) -> None:
    import rasterio

    preview = Image.open(scene_dir / "sentinel2_preview.png").convert("RGB")
    with rasterio.open(scene_dir / "worldcover_2021_on_sentinel.tif") as src:
        reference = map_worldcover(src.read(1))
    with rasterio.open(prediction_tif) as src:
        predicted = src.read(1)
        predicted = np.where(predicted < 0, 255, predicted).astype(np.uint8)
    ref_rgb = Image.fromarray(colorize_prediction(reference)).convert("RGB")
    pred_rgb = Image.fromarray(colorize_prediction(predicted)).convert("RGB")
    target_h = 420
    panels = []
    for image in (preview, ref_rgb, pred_rgb):
        ratio = target_h / image.height
        panels.append(image.resize((int(image.width * ratio), target_h), Image.Resampling.LANCZOS))
    margin, title_h = 24, 56
    width = sum(panel.width for panel in panels) + margin * 4
    canvas = Image.new("RGB", (width, target_h + title_h + 44), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = margin
    titles = ("Sentinel-2 RGB", "WorldCover reference", "Model prediction")
    for title, panel in zip(titles, panels, strict=True):
        draw.text((x, 18), title, fill="black", font=font)
        canvas.paste(panel, (x, title_h))
        x += panel.width + margin
    draw.text(
        (margin, target_h + title_h + 16),
        "External holdout • visualization generated from measured model output",
        fill="black",
        font=font,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)


def _results_markdown(evaluation: dict, bundle: dict) -> str:
    ext = evaluation["external_test"]
    iou = ext["iou"]
    by_name = dict(zip(CLASS_NAMES, iou, strict=True))
    return f"""## Measured model result

The flagship model is a compact six-band U-Net trained from scratch on spatially isolated Sentinel-2
scenes with ESA WorldCover reference labels. Validation and external testing use whole AOIs that are
never cropped into the training split.

| Metric | External holdout |
| --- | ---: |
| Macro IoU | **{_safe(ext['macro_iou'])}** |
| 95% patch-bootstrap CI | {_safe(ext['macro_iou_ci95'][0])}–{_safe(ext['macro_iou_ci95'][1])} |
| Weighted IoU | {_safe(ext['weighted_iou'])} |
| Macro Dice | {_safe(ext['macro_dice'])} |
| Pixel accuracy | {_safe(ext['pixel_accuracy'])} |
| ECE | {_safe(ext['ece'])} |
| Tree-cover IoU | {_safe(by_name.get('tree_cover'))} |
| Built-up IoU | {_safe(by_name.get('built_up'))} |

External AOIs: {', '.join(SHOWCASE_SPLIT.external_test)}. The repository reports absent classes as
unavailable rather than converting them into misleading zero-IoU values. The deployment artifact is
loaded only after SHA-256 verification against `models/landcover/bundle.json`.

![Sentinel-2, reference and prediction comparison](docs/assets/prediction-triptych.png)

![External class IoU](docs/assets/class-iou.svg)

![Training history](docs/assets/training-history.svg)
"""


def update_readme(readme: Path, content: str) -> None:
    start = "<!-- SHOWCASE_RESULTS_START -->"
    end = "<!-- SHOWCASE_RESULTS_END -->"
    text = readme.read_text()
    if start not in text or end not in text:
        raise ValueError("README is missing showcase result markers")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(before + start + "\n\n" + content.rstrip() + "\n\n" + end + after)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/curated"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("models/landcover"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/landcover"))
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    bundle = verify_bundle(args.bundle_dir)
    evaluation = json.loads((args.reports_dir / "evaluation.json").read_text())
    if (
        evaluation["dataset_signature"] != bundle["dataset_signature"]
        or evaluation["experiment_signature"] != bundle["experiment_signature"]
    ):
        raise ValueError("evaluation and model bundle were built from different dataset signatures")
    scene_key = SHOWCASE_SPLIT.external_test[0]
    scene_dir = args.data_root / scene_key
    prediction_tif = args.reports_dir / f"{scene_key}-prediction.tif"
    inference = predict_scene(
        scene_dir / "sentinel2_multispectral.tif",
        args.bundle_dir,
        prediction_tif,
        device=args.device,
    )
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    _triptych(scene_dir, prediction_tif, args.assets_dir / "prediction-triptych.png")
    _svg_bar_chart(
        evaluation["external_test"]["iou"], CLASS_NAMES, args.assets_dir / "class-iou.svg"
    )
    _svg_history(evaluation["history"], args.assets_dir / "training-history.svg")
    summary = {
        "schema_version": "1.0",
        "dataset_signature": evaluation["dataset_signature"],
        "experiment_signature": evaluation["experiment_signature"],
        "model_bundle_sha256": bundle["files"][bundle["model_file"]]["sha256"],
        "external_test": evaluation["external_test"],
        "showcase_scene": scene_key,
        "inference": inference,
        "integrity_note": "results are generated from the same verified model/data signature",
    }
    (args.reports_dir / "showcase-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    content = _results_markdown(evaluation, bundle)
    Path("docs/RESULTS.md").write_text("# Results\n\n" + content)
    update_readme(args.readme, content)
    print(args.reports_dir / "showcase-summary.json")


if __name__ == "__main__":
    main()

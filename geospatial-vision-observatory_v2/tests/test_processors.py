from pathlib import Path

from PIL import Image, ImageDraw

from geo_vision.vision.processors import (
    GeospatialRgbBaseline,
    QualityControl,
    TemporalChangeBaseline,
)


def image(path: Path, offset: int = 0) -> None:
    canvas = Image.new("RGB", (512, 512), (105, 105, 105))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 270, 512, 512), fill=(35, 125, 55))
    draw.rectangle((80 + offset, 120, 260 + offset, 260), fill=(145, 145, 145))
    draw.rectangle((330, 80, 500, 200), fill=(35, 70, 140))
    canvas.save(path)


def test_baseline_processors(tmp_path: Path) -> None:
    current, previous = tmp_path / "current.png", tmp_path / "previous.png"
    image(current, 3)
    image(previous)
    quality = QualityControl().process(current)
    scene = GeospatialRgbBaseline().process(current)
    change = TemporalChangeBaseline().process(current, previous)
    assert quality["passed"] is True
    assert scene["method_class"] == "descriptive_rgb_land_surface_proxy"
    assert float(scene["vegetation_rgb_proxy_fraction"]) > 0
    assert change["available"] is True
    assert TemporalChangeBaseline().process(current)["available"] is False


def test_blank_scene_has_no_vegetation_proxy(tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    Image.new("RGB", (512, 512), "black").save(path)
    assert GeospatialRgbBaseline().process(path)["vegetation_rgb_proxy_fraction"] == 0

from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = 255
INPUT_BANDS = ("red", "green", "blue", "nir", "swir16", "swir22")
WORLDCOVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
CLASS_NAMES = (
    "tree_cover",
    "shrubland",
    "grassland",
    "cropland",
    "built_up",
    "bare_sparse_vegetation",
    "snow_ice",
    "permanent_water",
    "herbaceous_wetland",
    "mangroves",
    "moss_lichen",
)
CODE_TO_INDEX = {code: index for index, code in enumerate(WORLDCOVER_CODES)}
INDEX_TO_CODE = dict(enumerate(WORLDCOVER_CODES))

# Stable cartographic palette used only for generated visual artifacts.
CLASS_RGB = (
    (0, 100, 0),
    (255, 187, 34),
    (255, 255, 76),
    (240, 150, 255),
    (250, 0, 0),
    (180, 180, 180),
    (240, 240, 240),
    (0, 100, 200),
    (0, 150, 160),
    (0, 207, 117),
    (250, 230, 160),
)


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    external_test: tuple[str, ...]


SHOWCASE_SPLIT = DatasetSplit(
    train=("helsinki_metro", "north_karelia_forest", "turku_coast", "oulu_mixed"),
    validation=("tampere_growth", "jyvaskyla_validation"),
    external_test=("stockholm_external", "tallinn_external"),
)

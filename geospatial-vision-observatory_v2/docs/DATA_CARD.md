# Data card: Sentinel-2 + WorldCover + Hansen geospatial curation

## Purpose

The data pipeline creates a compact, reproducible urban/forestry/land-cover benchmark from real public geospatial sources. It is designed for an auditable GitHub showcase and reproducible experimentation, not for global statistical representativeness.

## Primary imagery — Copernicus Sentinel-2 L2A

The project discovers Sentinel-2 L2A items through the Element 84 Earth Search STAC API. Scientific preparation reads spatial windows from Cloud-Optimized GeoTIFF assets rather than downloading full global scenes.

Six bands are aligned onto the selected red-band grid:

- red;
- green;
- blue;
- NIR;
- SWIR1 (`swir16`);
- SWIR2 (`swir22`).

Each band is converted to reflectance using its STAC `raster:bands` scale and offset metadata. This is important because processing-baseline changes can introduce non-zero offsets; raw digital numbers are not treated as calibrated reflectance.

When Sentinel-2 Scene Classification Layer (SCL) is available, no-data, defective, cloud-shadow, unclassified/cloud, medium/high cloud and cirrus classes are excluded from training/evaluation pixels.

## ESA WorldCover 2021 v200

WorldCover 2021 v200 is the segmentation reference. It provides 11 classes at approximately 10 m resolution and is distributed as COGs under CC BY 4.0.

WorldCover is a model-derived map, not perfect pixel ground truth. The showcase therefore:

- uses a 2021 Sentinel acquisition window by default;
- reports the label version in every manifest;
- keeps geographic AOIs isolated across train/validation/external test;
- describes the result as WorldCover-style/reference segmentation rather than authoritative land truth.

## Hansen Global Forest Change v1.13

The workflow aligns `treecover2000` and `lossyear` layers from Hansen Global Forest Change v1.13 to the Sentinel grid. These layers provide forestry context and descriptive measurements.

They are **not** used as a temporally aligned target for the flagship single-date 2021 segmentation model: `treecover2000` represents a different year, and historical `lossyear` does not mean a 2021 pixel can be interpreted as an instantaneous forest-loss observation. This explicit boundary prevents a common temporal-labeling error.

## Frozen spatial split

The public showcase split is declared in `src/geo_vision/ml/schema.py`:

### Train

- `helsinki_metro` — urban/forest interface;
- `north_karelia_forest` — managed boreal forest;
- `turku_coast` — urban/cropland/coastal landscape;
- `oulu_mixed` — boreal urban/wetland landscape.

### Validation

- `tampere_growth`;
- `jyvaskyla_validation`.

### External test

- `stockholm_external`;
- `tallinn_external`.

No patch from one AOI may cross a split boundary. Seed/model selection uses validation AOIs only; external AOIs are evaluated after selection.

## Temporal window

`prepare_training_data.py` defaults to 2021-05-01 through 2021-09-30. Earth Search scene-wide `eo:cloud_cover` is used only to rank a bounded candidate set; the strict 15% publication-quality ceiling is measured from Sentinel-2 SCL pixels over the actual AOI. This avoids rejecting a clear AOI merely because clouds occur elsewhere in the granule, while still failing closed on locally obscured data. The temporal window reduces, but does not eliminate, disagreement with WorldCover 2021.

## Derived files and integrity

Every curated AOI contains the multispectral stack, spectral indices, optional SCL, preview, aligned WorldCover and Hansen layers, and `manifest.json`.

The manifest records:

- STAC item identifier and acquisition timestamp;
- upstream COG URLs;
- scene-wide STAC cloud-cover estimate and AOI SCL obscured-pixel fraction;
- per-band scale/offset/nodata metadata;
- CRS, affine transform and dimensions;
- SCL masking policy;
- spectral-index summary statistics;
- WorldCover class fractions;
- Hansen summary statistics;
- processing configuration;
- SHA-256 and byte size for every generated raster/image output.

`prepare_training_data.py` reuses an existing AOI only when every listed hash and byte size verifies **and** the AOI identity, 2021 date window, AOI SCL obscured-pixel ceiling, Sentinel collection and raster-size policy match the current request. Selection-mode preparation excludes the external holdout entirely. Training derives a **dataset signature** from scene file identities and an **experiment signature** from the dataset signature plus training configuration.

## Bias and limitations

The split is deliberately Nordic-heavy. It provides clean spatial isolation and mixed urban/forest/water/cropland conditions, but it is not evidence of global performance. Climate, construction styles, topography, seasonality, atmospheric effects and class prevalence differ substantially elsewhere.

Rare WorldCover classes such as mangroves may be absent from the external AOIs. Absent classes are reported as unavailable rather than scored as zero.

## Privacy and responsible use

The project operates at land-surface mapping scales and prohibits person identification, facial recognition, biometric inference, household surveillance and parcel-level conclusions about people. Built-up segmentation is not cadastral/legal boundary mapping.

## Sentinel collection policy

Scientific curation uses Earth Search `sentinel-2-c1-l2a` Collection 1 COGs. The lightweight operational monitor may use the thumbnail-friendly `sentinel-2-l2a` collection; operational previews are not used as training evidence.

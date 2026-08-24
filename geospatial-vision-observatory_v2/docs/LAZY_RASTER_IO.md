# Lazy raster and ML access

Iteration 4 removes full-AOI raster materialization from the training data path while preserving the existing scientific dataset contract.

## Contract

`load_lazy_scene()` validates curated-manifest identity, file integrity, dimensions, CRS, band descriptions and raster alignment without reading complete raster pixel arrays. A `LazyScene` retains only metadata and opens raster windows when a patch or bounded statistics block is requested.

The training pipeline now uses `LazyScene` objects for train, validation and external-evaluation scenes. `PatchDataset` requests only the patch window needed for the current sample. The existing inference engine already performs tiled/windowed reads and therefore remains bounded by its configured tile size.

## Exact validity semantics

A lazy window applies the same validity rules as the previous eager loader:

- finite six-band reflectance;
- source nodata rejection;
- WorldCover label mapping and ignored-label rejection;
- optional Sentinel-2 SCL invalid/cloud class rejection;
- reflectance-range checks.

Parity tests compare eager and lazy image, target and validity arrays directly.

## Bounded statistics

Patch indexing reads only one patch-sized window at a time. Class-weight counts are accumulated from bounded raster blocks.

Normalization uses a deterministic two-pass sampler. The first pass counts valid pixels. If the valid population exceeds the configured sample cap, the sampler draws the same valid-pixel ranks as the former eager `Generator.choice` path and the second pass collects only those ranks. This preserves normalization sampling semantics while limiting retained pixel memory to `sample_pixels_per_scene`.

Mean and standard deviation accumulation is performed in float64 and persisted as float32. This avoids reduction-order differences caused by raster-window memory layout.

## Memory boundary

Lazy access bounds raster pixel memory, not all training memory. Model tensors, optimizer state, the patch-reference index and the configured normalization sample still consume memory. A later execution-planning iteration may shard or externalize the patch index if measured workloads make it necessary.

## Scientific identity

Lazy access is an execution optimization only. It does not create new scientific evidence units, change AOI roles, alter dataset signatures, or relax external-holdout isolation. Processing cells remain execution units; AOIs remain evidence units.

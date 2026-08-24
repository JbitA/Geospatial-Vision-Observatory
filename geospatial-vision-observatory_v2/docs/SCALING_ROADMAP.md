# Scaling roadmap

The scaling sequence is deliberately local-first. Infrastructure changes must not change scientific meaning.

## Completed foundation

### Iteration 1 — AOI v2

- Polygon/MultiPolygon AOIs and deterministic geometry identities.
- Scientific roles and positive-area evidence isolation.
- Exact polygon masking.
- WorldCover/Hansen source-tile enumeration.
- Provider-neutral artifact-store contract with a local atomic filesystem implementation.

### Iteration 2 — deterministic projected processing cells

- Deterministic metre-based working CRS.
- Origin-anchored fixed-size core cells and stable cell IDs.
- Optional halo/context bounds without overlapping scientific cores.
- Exact AOI∩cell geometry.
- Direct multi-tile WorldCover and Hansen reprojection.
- Same-overpass Sentinel multi-item coverage for boundary cells.
- Independent cell artifacts and collection-level provenance.

### Iteration 3 — resumability and idempotent job state

- Frozen acquisition windows persisted before cell execution begins.
- Deterministic request, run and per-cell recipe identities.
- Run-scoped immutable artifact directories under `runs/<run_id>/`.
- SHA-256 + byte-size verification before a completed cell is reused.
- Automatic rebuild of missing/corrupt cells while valid cells are skipped.
- Source-selection continuity checks that fail closed if rebuilding would change evidence.
- Atomic `run-state.json` updates after each completed cell.
- Explicit `--fresh-window` and `--force-rebuild` controls.
- Immutable local `ArtifactStore` semantics: same key/same bytes is idempotent; same key/different bytes is a conflict.

### Iteration 4 — lazy raster and ML access

- Curated training scenes are integrity-validated from metadata without loading complete raster arrays.
- Training patches are read by bounded raster windows on demand.
- Patch validity and class weights are computed from bounded reads.
- Normalization sampling reproduces eager valid-pixel sampling with a deterministic two-pass bounded-memory algorithm.
- Normalization reductions use float64 accumulation for stable eager/lazy parity.
- Full-scene inference remains tiled/windowed.
- Eager/lazy parity and no-unwindowed-read tests protect the execution contract.

See `docs/LAZY_RASTER_IO.md`.

## Next iterations

### Iteration 5 — scientific experiment state

Add durable holdout-exposure records, AOI-level aggregate metrics, spatial block uncertainty and explicit distance/overlap checks for newly introduced external AOIs.

### Iteration 6 — bounded parallel local execution

After serial cell execution is reproducible, add separate limits for network acquisition, CPU/raster reprojection and GPU work. Deterministic manifests and results must be independent of worker completion order.

### Iteration 7 — Rust orchestration

Move planning/job supervision/hashing/acquisition scheduling into Rust only after the Python contracts are stable. Keep PyTorch for ML and GDAL/PROJ/GEOS for geospatial kernels.

### Iteration 8 — shared metadata/storage

Introduce PostgreSQL/PostGIS and shared/object storage only when multiple hosts or researchers justify it. Retain the same logical records and artifact identities.

### Iteration 9 — AWS/Azure scale mode

Implement S3 and Azure Blob versions of the artifact contract plus distributed workers. Cloud execution must produce the same AOI, cell, recipe, metric and provenance identities as local execution.

### Iteration 10 — enterprise security hardening

Add workload identity, RBAC, authenticated APIs, signing/provenance, backup/restore, network segmentation and formal control mapping after scaled functionality is reliable.

## Scale triggers

Local execution remains preferred while data fits available disk, job durations are acceptable, one host has enough memory/GPU capacity, backup is manageable and concurrency is low.

Shared/cloud scale becomes justified by measured constraints such as tens of terabytes of reusable rasters, thousands of AOIs, multiple simultaneous experiments, sustained multi-GPU demand, several worker hosts, a centralized artifact library or operationally difficult local recovery.

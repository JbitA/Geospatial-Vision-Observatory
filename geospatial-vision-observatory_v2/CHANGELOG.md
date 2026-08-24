# Changelog

## 2.0.0 — 2026-08-24

Version 2 is the first major architecture release after the original GitHub baseline. It keeps the
existing scientific land-cover workflow while adding the scaling and evidence primitives required
for larger AOIs and future enterprise deployment.

### Added

- Polygon and MultiPolygon AOIs with canonical geometry identities and explicit scientific roles.
- Deterministic projected processing cells with stable IDs, exact AOI intersections, and optional
  context halos.
- Multi-source Sentinel-2, WorldCover, and Hansen cell execution without giant AOI mosaics.
- Run-scoped immutable processing artifacts, frozen acquisition windows, integrity-checked resume,
  corruption recovery, and deliberate fresh-window semantics.
- Lazy windowed raster access for ML training, preserving the scientific dataset contract without
  loading complete AOIs into memory.
- Provider-neutral artifact storage boundary with a hardened local filesystem implementation.
- Processing-cell, run-state, AOI, and plan JSON Schemas.
- Experimental scientific-governance primitives for external-holdout exposure tracking and
  untouched-AOI validation.
- Python 3.14 runtime contract and refreshed stable dependency/toolchain pins.
- Scaling, AOI v2, and lazy-I/O architecture documentation.

### Changed

- Public package version is now `2.0.0`.
- The scaled processing workflow is a first-class supported repository entry point.
- Scientific AOIs remain evidence units; processing cells are execution units only.
- Large-run outputs are stored under immutable run identities rather than one mutable AOI folder.

### Fixed

- Direct invocation of `scripts/prepare_processing_cells.py` now resolves repository-local imports
  correctly.
- Release metadata is consistent across `pyproject.toml`, package namespace, citation metadata, and
  validation output.

### Deliberately not included yet

- Distributed workers, PostGIS job coordination, S3/Azure Blob artifact backends, enterprise IAM,
  multi-tenancy, and remote authenticated APIs remain roadmap work. Version 2 establishes the
  scientific and execution contracts those capabilities must preserve.

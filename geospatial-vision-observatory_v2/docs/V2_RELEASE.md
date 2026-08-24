# Geospatial Vision Observatory v2

## Purpose

Version 2 turns the original single-workflow showcase into a deterministic, resumable geospatial ML
execution foundation while preserving the original research discipline. The central invariant is
that scaling infrastructure must not change scientific meaning.

## What is new

1. **AOI v2** — Polygon/MultiPolygon geometry, canonical SHA-256 identity, explicit train,
   validation, external-observed, external-unobserved, and inference-only roles.
2. **Projected processing cells** — deterministic core grids, exact AOI intersection geometry,
   optional halos, stable cell IDs, and bounded raster operations.
3. **Multi-reference execution** — direct cell-grid reprojection from all required Sentinel-2,
   WorldCover, and Hansen sources rather than assuming one source tile per AOI.
4. **Resumability** — frozen acquisition windows, immutable run IDs, artifact hash/size validation,
   idempotent reuse, corruption-triggered repair, and source-drift rejection.
5. **Lazy ML raster access** — bounded window reads for patch indexing, normalization, class weights,
   and batches so training memory does not scale with full AOI raster size.
6. **Evidence foundations** — deterministic plan/run identities, content hashes, scientific-role
   boundaries, and early holdout-exposure governance primitives.

## Compatibility

The original eight-AOI scientific workflow remains available. Version 2 changes the public runtime
contract to Python 3.14 and adds new schemas/entry points, so it is released as a major version rather
than pretending the changes are a transparent patch to v1.

## Enterprise direction

The v2 contracts are intentionally infrastructure-neutral. Local filesystem + SQLite remain the
implemented default. Future PostgreSQL/PostGIS, object-storage, distributed-worker, and authenticated
enterprise modes must produce the same scientific identities, recipes, metrics, and provenance.

## Validation policy

A GitHub release is accepted only when the normal test suite, compile checks, repository hygiene,
source-integrity manifest, and strict CI quality gates pass in the declared Python 3.14 environment.
Scientific showcase thresholds remain separate from software/runtime integrity and are never lowered
to make a model appear publishable.

# Architecture

## Design principles

1. **Separate operational observation from scientific training.** The always-on service only needs a small preview; multispectral COG preparation is heavier and operator-invoked.
2. **Make provenance part of the data model.** Source identifiers, timestamps, hashes and audit events are persisted with observations instead of reconstructed later.
3. **Prefer one completed ML path.** The flagship six-band model owns a complete train → evaluate → export → infer lifecycle.
4. **Fail closed on runtime integrity.** Unknown hosts, invalid payloads, hash mismatches and split-policy violations stop the relevant operation. Research-score thresholds control publication eligibility rather than local runtime availability.
5. **Keep evidence reviewable on GitHub.** Large rasters stay out of Git; compact reports, model bundle, model card and visual results remain reviewable.

## Operational path

`Sentinel-2 STAC metadata → validated preview → SHA-256 content-addressed storage → SQLite provenance/audit → deterministic preview processors → FastAPI/Prometheus/dashboard`

The source client enforces bounded request behavior, exact destination hosts, HTTPS, redirect rejection and image validation.

## Scientific path

`train/validation COG windows + WorldCover/Hansen → calibrated common grid → validation-only seed candidates → frozen selection record → external AOI curation → selection-data re-verification → selected final training → post-training external load/evaluation → PT2 bundle → verified inference → generated GitHub evidence`

### Data identity

Each AOI manifest hashes all generated outputs. A dataset signature hashes scene identities. An experiment signature hashes the dataset signature plus the exact training configuration. These identities flow into evaluation and the model bundle.

### Model identity

`bundle.json` hashes every deployment file. Inference verifies the bundle before loading. The dashboard model-summary endpoint additionally hashes the mounted model and compares it with the generated showcase summary.

## Storage

Operational previews use content-addressed files and SQLite. Curated scientific rasters are directory-based artifacts because GeoTIFF/rasterio workflows benefit from normal filesystem semantics. Large curated files are intentionally ignored by Git.

## Integration boundaries

- Source integration: STAC-compatible discovery plus allowlisted COG hosts.
- Training integration: `data/curated/<aoi>` contract and `manifest.json`.
- Inference integration: six-band GeoTIFF + verified `models/landcover` bundle.
- API integration: read-only JSON/metrics endpoints; no arbitrary uploads/fetch URLs.
- Automation integration: Windows `.cmd`/PowerShell launchers, `scripts/run_showcase_pipeline.py`, and GitHub Actions.

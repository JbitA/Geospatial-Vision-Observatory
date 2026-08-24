# AOI v2, deterministic processing cells, and resumable runs

AOI v2 is the compatibility-preserving path from the fixed eight-area research baseline toward large, irregular and multi-reference-tile study regions.

## Iterations 2–3 scope

Iteration 2 adds a real projected execution unit rather than treating a large AOI as one raster rectangle:

- GeoJSON `Polygon` / `MultiPolygon` AOIs in WGS84;
- deterministic geometry identities and explicit scientific roles;
- deterministic working-CRS selection;
- origin-anchored processing cells with stable SHA-256 identities;
- non-overlapping scientific core cells plus optional halo/context bounds;
- exact AOI-intersection geometry for every cell;
- per-cell WorldCover and Hansen tile enumeration;
- direct multi-source reprojection onto the cell grid without building a giant intermediate mosaic;
- deterministic same-overpass Sentinel-2 set-cover when one STAC item does not cover a complete cell;
- independent per-cell raster artifacts and provenance manifests;
- a logical AOI collection manifest that preserves the AOI as the scientific unit;
- frozen per-run acquisition windows and deterministic request/run/recipe identities;
- restart-safe per-cell reuse based on manifest recipe identity plus SHA-256/byte-size validation;
- run-scoped immutable directories so a deliberately fresh acquisition window preserves prior evidence.

The existing eight-AOI training/evaluation workflow remains unchanged. Processing cells are execution and artifact units, **not independent train/validation/test samples**.

## Planning

```powershell
C:\Users\you\.gvo\venv-py314\Scripts\python.exe scripts\plan_aois.py examples\aoi-irregular.geojson --output reports\aoi-plan.json
```

The plan schema is now `2.0`. Each AOI entry contains its deterministic processing grid, working CRS, grid parameters, ordered cell records, exact source-tile requirements and legacy-engine compatibility information.

Default grid recipe:

```text
resolution:   10 m
core cell:    1024 x 1024 pixels
core extent:  10.24 km x 10.24 km
halo:         32 pixels / 320 m per side
```

Grid lines are anchored to the selected projected CRS origin, not to the AOI bounding box. Small edits near one AOI edge therefore do not renumber every unaffected cell.

## Working CRS policy

Iteration 2 uses a deterministic fail-closed policy:

1. AOIs fully contained in one ordinary UTM zone use that WGS84 UTM EPSG CRS.
2. Wider or multi-zone AOIs use an AOI-centred Lambert azimuthal equal-area CRS with a deterministically serialized centre.
3. Fully Arctic/Antarctic AOIs use standard polar stereographic CRSs.
4. Antimeridian-spanning AOIs remain explicitly unsupported; they are rejected rather than implicitly wrapped.

`pyproj` is now an explicit runtime dependency because projected planning is part of the public AOI contract rather than a transitive raster-library implementation detail.

## Scaled execution

Use the dedicated cell executor for new large-AOI work:

```powershell
C:\Users\you\.gvo\venv-py314\Scripts\python.exe scripts\prepare_processing_cells.py examples\aoi-irregular.geojson --output D:\gvo-data\processing-cells --start-date 2021-05-01 --end-date 2021-09-30 --require-cloud-threshold
```

Output layout:

```text
<output>/<aoi_id>/
|-- run-state.json
`-- runs/
    `-- <run_id>/
        |-- cells-manifest.json
        `-- cells/
            `-- <cell_id>/
                |-- sentinel2_multispectral.tif
                |-- sentinel2_indices.tif
                |-- sentinel2_scl.tif          # when available
                |-- sentinel2_preview.png
                |-- worldcover_2021_on_cell.tif
                |-- hansen_treecover2000_on_cell.tif
                |-- hansen_lossyear_on_cell.tif
                `-- manifest.json
```

For lookback-based acquisition, the UTC start/end window is resolved once and written to `run-state.json` before the first cell is fetched. Re-running the same request resumes that exact window. A cell is skipped only if its recipe identity matches and every listed artifact still has the recorded byte size and SHA-256. Missing, changed, symlinked, or stale-extra files invalidate the cell.

Use `--fresh-window` to intentionally create a new lookback run. The new run receives a new `run_id`; the previous run remains intact. `--force-rebuild` re-executes cells in the same run, but a source-selection mismatch is rejected so forced repair cannot silently change scientific evidence.

## Sentinel multi-item rule

A processing cell may intersect more than one Sentinel-2 STAC item footprint. Iteration 2 clusters candidates by acquisition time and accepts only a temporally coherent set whose footprints cover the complete AOI intersection for that cell. The set-cover choice is deterministic. If no candidate cluster covers the complete target under the cloud policy, execution fails rather than emitting a partial cell.

When strict cloud curation is enabled, SCL quality is measured over the actual item/cell intersection and the assembled cell is checked again after reprojection.

## Multi-reference raster rule

WorldCover and Hansen sources are enumerated per cell. Every required tile is reprojected directly into the fixed target grid. The merger distinguishes valid zero-valued Hansen pixels from source absence, so `lossyear=0` is preserved as valid “no recorded loss” rather than treated as nodata.

No AOI-sized WorldCover/Hansen mosaic is created.

## Scientific roles and isolation

- `train`
- `validation`
- `external_unobserved`
- `external_observed`
- `inference_only`

All evidence-producing AOIs must remain positive-area disjoint. A disconnected study area should be one `MultiPolygon` AOI. `inference_only` areas may overlap because they do not count as independent evidence.

Stockholm and Tallinn remain `external_observed` in the continuing baseline lineage. Future untouched publication-grade evaluation requires new `external_unobserved` geography.

## Deliberate boundaries after Iteration 4

Iteration 4 does **not** yet claim:

- bounded parallel cell execution;
- Rust orchestration;
- PostgreSQL/PostGIS metadata;
- S3 or Azure Blob artifact backends;
- distributed workers;
- antimeridian splitting.

Those capabilities build on the cell identity, resumability, and bounded raster-access contracts established here. Adding them before this contract is stable would couple orchestration complexity to an unproven scientific execution unit.

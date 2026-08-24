# Scientific method and evaluation protocol

## Primary question

Can a compact model trained from scratch on calibrated six-band Sentinel-2 imagery reproduce ESA WorldCover-style land-cover classes on geographically separated Nordic AOIs with useful performance, while preserving a fully auditable data/model lineage?

The flagship hypothesis is falsifiable: the trained model must pass declared external-holdout quality/calibration gates. Successful execution alone is not evidence of scientific performance.

## Experimental unit

The primary experimental unit is the acquisition/AOI, not an arbitrary image crop. Patches are computational samples derived from an acquisition, but they inherit the parent AOI split and never cross train/validation/external-test boundaries.

This prevents a common remote-sensing leakage mode where neighboring pixels from one scene are randomly divided into train and test sets.

## Data split

- Train: Helsinki, North Karelia, Turku, Oulu.
- Validation: Tampere, Jyväskylä.
- External test: Stockholm, Tallinn.

All are curated using the same declared 2021 seasonal window and processing rules.

## Seed selection without external-test tuning

The default workflow trains three seeds. Each candidate is evaluated only on the validation AOIs. The selected seed is the candidate with the largest validation macro IoU, breaking ties with weighted IoU.

The selection decision is written to `reports/landcover/seed-selection.json` and explicitly records that the external holdout was not used for selection.

Only after that selection record is written does the workflow curate the Stockholm/Tallinn external AOIs. It re-verifies that the train/validation dataset signature is unchanged, then retrains the selected seed deterministically. External pixel arrays are loaded only after the final model state has been restored, so they cannot influence normalization, class weights, optimization, early stopping, or seed selection. This external result is what the GitHub showcase displays.

## Preprocessing

1. Query bounded Sentinel-2 L2A items through STAC.
2. Select required red/green/blue/NIR/SWIR1/SWIR2 COG assets.
3. Read only the AOI window.
4. Apply each asset's STAC scale/offset to obtain calibrated reflectance.
5. Reproject continuous bands to the red-band grid with bilinear interpolation.
6. Reproject WorldCover/SCL/Hansen categorical layers with nearest-neighbor interpolation.
7. Mask invalid/cloud-shadow/cloud/cirrus SCL classes where available.
8. Derive normalization statistics from training AOIs only.

## Model

The flagship network is a compact U-Net-style semantic segmenter with residual depthwise-separable blocks and GroupNorm. It receives six reflectance bands and outputs 11 WorldCover-style classes.

No pretrained network weights are required. This reduces network/supply-chain dependence and ensures the repository demonstrates actual model training rather than only downloading a pretrained encoder.

Training uses bounded class weights, cross-entropy with small label smoothing, AdamW, OneCycleLR, gradient clipping, deterministic spatial augmentations and early stopping on validation macro IoU. Automatic mixed precision is used on CUDA.

## Metrics

Primary metric:

- macro IoU over classes present in the evaluated reference set.

Secondary:

- weighted IoU;
- macro Dice;
- pixel accuracy;
- per-class IoU/Dice/support;
- expected calibration error;
- patch-level inference latency p50/p95;
- tree-cover IoU (forestry slice);
- built-up IoU (urban slice).

The pipeline reports a patch-bootstrap 95% confidence interval for external macro IoU. Patches are not a substitute for additional independent geographic datasets, so this interval is an uncertainty summary within the frozen external scenes, not a claim of global confidence.

## Missing classes

A class absent from the reference pixels has no defined IoU denominator. It is serialized as `null`/unavailable rather than zero. This avoids penalizing a model for a class that did not occur and avoids pretending that the class was tested.

## Calibration

Expected calibration error uses model maximum-class probabilities against pixel correctness on a deterministic bounded sample of valid pixels. Calibration is included in research-showcase eligibility because an overconfident segmentation model can be misleading even when its average IoU is acceptable. Local runtime records a miss without presenting it as a passing research result.

## Selection and release gates

`config/showcase-policy.yaml` contains a minimum **research-showcase** gate. It exists to prevent publishing a trivially non-functional model as a successful GitHub result.

Passing this gate is not production authorization. Production claims require additional geography/seasons, independent labels, subgroup/failure analysis, operational hardware benchmarking, security review and human scientific approval.

## Integrity

Two signatures are kept separate:

- **dataset signature** — hash of curated scene file identities;
- **experiment signature** — hash of the dataset signature plus training configuration/seed.

The evaluation report and deployment bundle must agree on both. The model file must also match the SHA-256 recorded in `bundle.json` before inference or dashboard evidence is accepted.

## Claims boundary

WorldCover is a reference product rather than authoritative pixel ground truth. Hansen layers carry their own temporal consistency limitations. The dashboard's RGB preview proxies are descriptive health/land-surface measurements and are not substituted for the six-band model or spectral indices.

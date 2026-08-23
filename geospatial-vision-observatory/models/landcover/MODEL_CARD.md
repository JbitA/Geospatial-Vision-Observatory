# Model Card — Multispectral Land-Cover Segmentation

## Intended use

Research-grade semantic segmentation of Sentinel-2 L2A six-band imagery into ESA WorldCover-style
land-cover classes. The model is not an authoritative cadastral, environmental-compliance, emergency,
or person/household inference system.

## Inputs and outputs

- Input bands: red, green, blue, nir, swir16, swir22
- Output classes: 11
- Training labels: ESA WorldCover 2021 v200 reference labels
- Spatial split: train=('helsinki_metro', 'north_karelia_forest', 'turku_coast', 'oulu_mixed'), validation=('tampere_growth', 'jyvaskyla_validation'), external=('stockholm_external', 'tallinn_external')

## External-test metrics

- Macro IoU: 0.2855
- 95% patch-bootstrap CI: [0.2779, 0.3286]
- Weighted IoU: 0.5973
- Macro Dice: 0.3492
- Pixel accuracy: 0.7412
- Expected calibration error: 0.0653

| Class | IoU |
| --- | ---: |
| tree_cover | 0.707 |
| shrubland | 0.000 |
| grassland | 0.035 |
| cropland | 0.094 |
| built_up | 0.397 |
| bare_sparse_vegetation | 0.101 |
| snow_ice | n/a |
| permanent_water | 0.950 |
| herbaceous_wetland | 0.000 |
| mangroves | 0.000 |
| moss_lichen | 0.000 |

## Reproducibility

- Dataset signature: `4869766cff98f1f4c8d1091ba598ee3d5a455a7efe7993793854e3b431a597d6`
- Experiment signature: `714025888b3e200f6a33bb7cb8b80bf8e8334947df0365e5420e6982959e8854`
- Seed: `20260824`
- Training device: `cpu`
- Artifact integrity: every bundle file is SHA-256 listed in `bundle.json`

## Limitations

WorldCover is a model-derived reference map rather than perfect pixel ground truth. The training data
covers a deliberately small set of AOIs, so performance must not be generalized beyond the reported
spatial holdouts without additional independent validation. Rare classes absent from the held-out
regions are reported as unavailable rather than treated as zero-IoU evidence.

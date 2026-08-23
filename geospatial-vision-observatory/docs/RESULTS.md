# Results

## Measured model result

The flagship model is a compact six-band U-Net trained from scratch on spatially isolated Sentinel-2
scenes with ESA WorldCover reference labels. Validation and external testing use whole AOIs that are
never cropped into the training split.

| Metric | External holdout |
| --- | ---: |
| Macro IoU | **0.285** |
| 95% patch-bootstrap CI | 0.278–0.329 |
| Weighted IoU | 0.597 |
| Macro Dice | 0.349 |
| Pixel accuracy | 0.741 |
| ECE | 0.065 |
| Tree-cover IoU | 0.707 |
| Built-up IoU | 0.397 |

External AOIs: stockholm_external, tallinn_external. The repository reports absent classes as
unavailable rather than converting them into misleading zero-IoU values. The deployment artifact is
loaded only after SHA-256 verification against `models/landcover/bundle.json`.

![Sentinel-2, reference and prediction comparison](docs/assets/prediction-triptych.png)

![External class IoU](docs/assets/class-iou.svg)

![Training history](docs/assets/training-history.svg)

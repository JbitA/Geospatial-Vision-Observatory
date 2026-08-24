# Reproducibility and integrity

## Windows reproduction contract

The canonical host environment is Windows 10/11 x64 with Python 3.14.

```powershell
.\scripts\setup.cmd
.\scripts\showcase.cmd
```

No WSL or Bash environment is required. The launcher resolves a dedicated environment at `%LOCALAPPDATA%\GeospatialVisionObservatory\venv-py314` and invokes its Python directly; shell activation is unnecessary. Curated rasters default to `%LOCALAPPDATA%\GeospatialVisionObservatory\curated` so a repository stored under OneDrive does not continuously synchronize the environment or training data.

For an NVIDIA CUDA 13.0 PyTorch environment:

```powershell
.\scripts\setup.cmd -TorchBackend Cuda130
.\scripts\showcase.cmd -Device cuda -SkipSetup
```

The resolved Python package environment is written to `reports/environment-freeze.txt`. Direct project dependencies are pinned, and the release captures SBOM/evidence after a successful measured run.

## Determinism

The trainer sets Python/NumPy/PyTorch seeds and requests deterministic PyTorch algorithms with warning fallback for unsupported kernels. Data augmentation is derived deterministically from seed + epoch + patch index.

Three seeds compete on validation data only; the selected seed is retrained for the final external evaluation. Exact floating-point equivalence across CPU/GPU models or driver/runtime revisions is not promised. The actual device/runtime is recorded so differences remain investigable.

## Data and experiment identity

The training-data signature is computed over train + validation AOIs before external data is curated. The publication dataset signature additionally covers the frozen external AOIs. Each curated scene manifest records SHA-256 and byte size for generated outputs.

The experiment signature includes training-data identity, seed and training hyperparameters. A reused model is accepted only when the experiment signature matches.

## Generated evidence

The public showcase derives from:

- `reports/landcover/seed-selection.json`;
- `reports/landcover/evaluation.json`;
- `models/landcover/bundle.json`;
- `reports/landcover/showcase-summary.json`;
- generated qualitative/metric graphics.

The README results block is generated from those artifacts and is not manually pre-populated.

## Network failure

A failed live acquisition is a failed real-data build. Synthetic GeoTIFFs exist only in tests and must never be substituted into publication results.

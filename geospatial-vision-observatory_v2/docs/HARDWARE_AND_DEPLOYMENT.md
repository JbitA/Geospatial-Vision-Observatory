# Hardware and deployment

## Primary Windows host

The canonical development/training environment is Windows 10/11 x64 with Python 3.14. A discrete NVIDIA GPU is optional but recommended for faster training. CPU execution remains supported for correctness and smaller experiments.

- `-Device auto` chooses CUDA when PyTorch reports it available, otherwise CPU.
- CUDA uses automatic mixed precision.
- Default patch size is 128 and batch size is 8.
- The workflow selects among validation-only seeds, then retrains the winner before external evaluation.

Use `nvidia-smi` to inspect the installed driver. The setup launcher supports the PyTorch CUDA 13.0 index through `-TorchBackend Cuda130`; use the CPU backend when accelerator compatibility is uncertain.

## Operational service on Windows

The API and worker can run natively after `.\scripts\setup.cmd`:

```powershell
.\scripts\start_api.cmd
.\scripts\worker_once.cmd
```

Operational preview ingestion is CPU-light. SQLite and content-addressed previews live under the configured Windows data directory.

## Scientific data preparation

Rasterio wheels provide the GDAL-backed geospatial stack on Windows. The default preparation limits each AOI to 768 pixels on its largest dimension so the showcase remains bounded while preserving real multispectral/georeferenced behavior.

## Optional Docker Desktop deployment

Docker Desktop can run the API/worker using the hardened Compose definition. These are Linux containers by design, but they are optional and separate from the Windows-native training workflow. The API remains loopback-bound unless deliberately placed behind an authenticated TLS reverse proxy.

## Model artifact

The final model is exported with `torch.export` as `models/landcover/landcover_multispectral.pt2` and is loaded only after bundle verification. Full-scene inference streams image tiles rather than loading the full raster into RAM.

## Scaling beyond the showcase

For larger programs, move curated derivatives to versioned object storage, publish dataset/model signatures to immutable metadata infrastructure, parallelize AOI preparation/training through an orchestrator, centralize egress policy, add signed SBOM/container/model attestations, and benchmark throughput/RAM/VRAM/storage/energy on target Windows or server infrastructure before production use.

## Native API port selection

`.\scripts\start_api.cmd` binds loopback only. It prefers port 8080 and automatically selects a fallback loopback port when Windows reports 8080 as unavailable or reserved. The selected URL is printed before Uvicorn starts. Pass `-Port <port>` to require a specific port; an unavailable explicit port fails closed rather than silently changing it.

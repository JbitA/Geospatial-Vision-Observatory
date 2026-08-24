from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .data import (
    PatchRef,
    Scene,
    SceneLike,
    class_weights,
    compute_normalization,
    dataset_signature,
    experiment_signature,
    load_lazy_scene,
    patch_index,
)
from .integrity import sha256_file, verify_bundle
from .metrics import (
    bootstrap_macro_iou_ci,
    confusion_matrix,
    expected_calibration_error,
    metrics_from_confusion,
)
from .model import build_model
from .schema import CLASS_NAMES, IGNORE_INDEX, INPUT_BANDS, SHOWCASE_SPLIT, WORLDCOVER_CODES


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20260823
    epochs: int = 14
    batch_size: int = 8
    patch_size: int = 128
    stride: int = 96
    learning_rate: float = 0.0015
    weight_decay: float = 0.0001
    base_channels: int = 20
    label_smoothing: float = 0.02
    min_valid_fraction: float = 0.75
    patience: int = 4
    workers: int = 0
    bootstrap_iterations: int = 1000

    def __post_init__(self) -> None:
        if not 1 <= self.epochs <= 500:
            raise ValueError("epochs must be between 1 and 500")
        if not 1 <= self.batch_size <= 128:
            raise ValueError("batch_size must be between 1 and 128")
        if not 32 <= self.patch_size <= 1024:
            raise ValueError("patch_size must be between 32 and 1024")
        if not 16 <= self.stride <= self.patch_size:
            raise ValueError("stride must be between 16 and patch_size")
        if not math.isfinite(self.learning_rate) or not 0.0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate must be finite and in (0, 0.1]")
        if not math.isfinite(self.weight_decay) or not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("weight_decay must be finite and in [0, 1]")
        if not 4 <= self.base_channels <= 128:
            raise ValueError("base_channels must be between 4 and 128")
        if not math.isfinite(self.label_smoothing) or not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be finite and in [0, 1)")
        if (
            not math.isfinite(self.min_valid_fraction)
            or not 0.0 < self.min_valid_fraction <= 1.0
        ):
            raise ValueError("min_valid_fraction must be finite and in (0, 1]")
        if not 1 <= self.patience <= self.epochs:
            raise ValueError("patience must be between 1 and epochs")
        if not 0 <= self.workers <= 32:
            raise ValueError("workers must be between 0 and 32")
        if not 10 <= self.bootstrap_iterations <= 10_000:
            raise ValueError("bootstrap_iterations must be between 10 and 10000")


def _seed_everything(seed: int) -> None:
    # CUBLAS_WORKSPACE_CONFIG must be present before CUDA libraries execute deterministic GEMMs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
    return requested


class PatchDataset:
    def __init__(
        self,
        scenes: list[SceneLike],
        refs: list[PatchRef],
        mean: npt.NDArray[np.float32],
        std: npt.NDArray[np.float32],
        *,
        patch_size: int,
        augment: bool,
        seed: int,
    ) -> None:
        self.scenes = scenes
        self.refs = refs
        self.mean = mean[:, None, None].astype(np.float32)
        self.std = std[:, None, None].astype(np.float32)
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.refs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[Any, Any, int]:
        import torch

        ref = self.refs[index]
        scene = self.scenes[ref.scene_index]
        y0, x0, size = ref.y, ref.x, self.patch_size
        if isinstance(scene, Scene):
            image = scene.image[:, y0 : y0 + size, x0 : x0 + size].copy()
            target = scene.target[y0 : y0 + size, x0 : x0 + size].copy()
        else:
            image, target, _valid = scene.read_patch(y0, x0, size, size)
        image = np.nan_to_num((image - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            if rng.random() < 0.5:
                image = image[:, :, ::-1]
                target = target[:, ::-1]
            if rng.random() < 0.5:
                image = image[:, ::-1, :]
                target = target[::-1, :]
            turns = int(rng.integers(0, 4))
            if turns:
                image = np.rot90(image, turns, axes=(1, 2))
                target = np.rot90(target, turns, axes=(0, 1))
            if rng.random() < 0.35:
                gain = rng.normal(1.0, 0.025, size=(image.shape[0], 1, 1)).astype(np.float32)
                image = image * gain
        return (
            torch.from_numpy(np.ascontiguousarray(image)).float(),
            torch.from_numpy(np.ascontiguousarray(target)).long(),
            index,
        )


def _loader(dataset: PatchDataset, *, batch_size: int, shuffle: bool, workers: int) -> Any:
    import torch

    generator = torch.Generator()
    generator.manual_seed(dataset.seed + dataset.epoch)
    typed_dataset = cast(torch.utils.data.Dataset[Any], dataset)
    return torch.utils.data.DataLoader(
        typed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        # The dataset epoch controls deterministic augmentation. Recreate workers per epoch so
        # subprocess copies see the current epoch rather than stale persistent state.
        persistent_workers=False,
        generator=generator,
        drop_last=False,
    )


def _evaluate(
    model: Any,
    dataset: PatchDataset,
    *,
    device: str,
    batch_size: int,
    workers: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    patch_matrices: list[npt.NDArray[np.int64]] = []
    confidences: list[npt.NDArray[np.float32]] = []
    correctness: list[npt.NDArray[np.bool_]] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for images, targets, _ in _loader(
            dataset, batch_size=batch_size, shuffle=False, workers=workers
        ):
            images = images.to(device, non_blocking=True)
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(images)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0 / max(len(images), 1)
            latencies.extend([elapsed] * len(images))
            probability = torch.softmax(logits, dim=1)
            confidence, prediction = probability.max(dim=1)
            prediction_np = prediction.cpu().numpy()
            target_np = targets.numpy()
            confidence_np = confidence.cpu().numpy()
            for truth_patch, pred_patch, conf_patch in zip(
                target_np, prediction_np, confidence_np, strict=True
            ):
                patch_matrix = confusion_matrix(
                    truth_patch, pred_patch, len(CLASS_NAMES), ignore_index=IGNORE_INDEX
                )
                patch_matrices.append(patch_matrix)
                matrix += patch_matrix
                valid = truth_patch != IGNORE_INDEX
                if valid.any():
                    # Cap calibration memory per patch while preserving deterministic sampling.
                    flat = np.flatnonzero(valid.ravel())
                    if flat.size > 4096:
                        step = max(flat.size // 4096, 1)
                        flat = flat[::step][:4096]
                    truth_flat = truth_patch.ravel()[flat]
                    pred_flat = pred_patch.ravel()[flat]
                    confidences.append(conf_patch.ravel()[flat])
                    correctness.append(pred_flat == truth_flat)
    metrics = metrics_from_confusion(matrix)
    confidence_all = np.concatenate(confidences) if confidences else np.array([], dtype=float)
    correct_all = np.concatenate(correctness) if correctness else np.array([], dtype=bool)
    low, high = bootstrap_macro_iou_ci(
        patch_matrices,
        iterations=bootstrap_iterations,
    )
    metrics.update(
        {
            "confusion_matrix": matrix.tolist(),
            "macro_iou_ci95": [low, high],
            "ece": expected_calibration_error(confidence_all, correct_all),
            "latency_p50_ms_per_patch": (
                float(np.quantile(latencies, 0.50)) if latencies else math.nan
            ),
            "latency_p95_ms_per_patch": (
                float(np.quantile(latencies, 0.95)) if latencies else math.nan
            ),
            "patches": len(dataset),
        }
    )
    return metrics


def _write_bundle_manifest(bundle_dir: Path, metadata: dict[str, Any]) -> Path:
    manifest_path = bundle_dir / "bundle.json"
    files = {}
    for path in sorted(bundle_dir.iterdir()):
        if path.is_file() and path.name != manifest_path.name:
            files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {**metadata, "files": files}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest_path



def _model_card(
    *,
    evaluation: dict[str, Any],
    dataset_sha256: str,
    experiment_sha256: str,
    config: TrainConfig,
    device: str,
) -> str:
    ext = evaluation["external_test"]
    per_iou = ext["iou"]
    rows = "\n".join(
        f"| {name} | {('n/a' if value is None or not np.isfinite(value) else f'{value:.3f}')} |"
        for name, value in zip(CLASS_NAMES, per_iou, strict=True)
    )
    return f"""# Model Card — Multispectral Land-Cover Segmentation

## Intended use

Research-grade semantic segmentation of Sentinel-2 L2A six-band imagery into ESA WorldCover-style
land-cover classes. The model is not an authoritative cadastral, environmental-compliance, emergency,
or person/household inference system.

## Inputs and outputs

- Input bands: {', '.join(INPUT_BANDS)}
- Output classes: {len(CLASS_NAMES)}
- Training labels: ESA WorldCover 2021 v200 reference labels
- Spatial split: train={SHOWCASE_SPLIT.train}, validation={SHOWCASE_SPLIT.validation}, external={SHOWCASE_SPLIT.external_test}

## External-test metrics

- Macro IoU: {ext['macro_iou']:.4f}
- 95% patch-bootstrap CI: [{ext['macro_iou_ci95'][0]:.4f}, {ext['macro_iou_ci95'][1]:.4f}]
- Weighted IoU: {ext['weighted_iou']:.4f}
- Macro Dice: {ext['macro_dice']:.4f}
- Pixel accuracy: {ext['pixel_accuracy']:.4f}
- Expected calibration error: {ext['ece']:.4f}

| Class | IoU |
| --- | ---: |
{rows}

## Reproducibility

- Dataset signature: `{dataset_sha256}`
- Experiment signature: `{experiment_sha256}`
- Seed: `{config.seed}`
- Training device: `{device}`
- Artifact integrity: every bundle file is SHA-256 listed in `bundle.json`

## Limitations

WorldCover is a model-derived reference map rather than perfect pixel ground truth. The training data
covers a deliberately small set of AOIs, so performance must not be generalized beyond the reported
spatial holdouts without additional independent validation. Rare classes absent from the held-out
regions are reported as unavailable rather than treated as zero-IoU evidence.
"""


def train_landcover(
    data_root: Path,
    bundle_dir: Path,
    reports_dir: Path,
    *,
    config: TrainConfig,
    requested_device: str = "auto",
    force: bool = False,
    evaluate_external: bool = True,
    export_bundle: bool = True,
) -> dict[str, Any]:
    import torch

    _seed_everything(config.seed)
    device = _device(requested_device)
    selection_dirs = {
        "train": [data_root / key for key in SHOWCASE_SPLIT.train],
        "validation": [data_root / key for key in SHOWCASE_SPLIT.validation],
    }
    external_dirs = [data_root / key for key in SHOWCASE_SPLIT.external_test]
    signature_dirs = [path for paths in selection_dirs.values() for path in paths]
    if evaluate_external or export_bundle:
        # Fingerprints bind the frozen external files into the final experiment identity without
        # opening their pixels. Pixel data is loaded only after the final model state is fixed.
        signature_dirs.extend(external_dirs)
    missing = [str(path) for path in signature_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "missing curated AOIs; run the training-data preparation first: " + ", ".join(missing)
        )
    data_signature = dataset_signature(signature_dirs)
    signature = experiment_signature(data_signature, asdict(config))
    bundle_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    existing = bundle_dir / "bundle.json"
    if not force and existing.is_file():
        try:
            payload = verify_bundle(bundle_dir)
        except ValueError:
            payload = {}
        if payload.get("experiment_signature") == signature:
            evaluation_path = reports_dir / "evaluation.json"
            if evaluation_path.is_file():
                cached_raw = json.loads(evaluation_path.read_text())
                if not isinstance(cached_raw, dict):
                    raise ValueError("cached evaluation must be a JSON object")
                cached = cast(dict[str, Any], cached_raw)
                if (
                    cached.get("dataset_signature") == data_signature
                    and cached.get("experiment_signature") == signature
                    and (not evaluate_external or "external_test" in cached)
                ):
                    return cached

    # The frozen external holdout is deliberately absent from this mapping during optimization.
    scenes = {name: [load_lazy_scene(path) for path in paths] for name, paths in selection_dirs.items()}
    mean, std = compute_normalization(scenes["train"], seed=config.seed)
    weights = class_weights(scenes["train"])
    refs = {
        name: patch_index(
            scene_list,
            patch_size=config.patch_size,
            stride=config.stride,
            min_valid_fraction=config.min_valid_fraction,
        )
        for name, scene_list in scenes.items()
    }
    datasets = {
        name: PatchDataset(
            scenes[name],
            refs[name],
            mean,
            std,
            patch_size=config.patch_size,
            augment=name == "train",
            seed=config.seed,
        )
        for name in scenes
    }

    model = build_model(
        in_channels=len(INPUT_BANDS), classes=len(CLASS_NAMES), base_channels=config.base_channels
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device),
        ignore_index=IGNORE_INDEX,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(math.ceil(len(datasets["train"]) / config.batch_size), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.20,
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_state: dict[str, Any] | None = None
    bad_epochs = 0

    for epoch in range(config.epochs):
        datasets["train"].set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        samples = 0
        for images, targets, _ in _loader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            workers=config.workers,
        ):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(images)
            samples += len(images)
        validation = _evaluate(
            model,
            datasets["validation"],
            device=device,
            batch_size=config.batch_size,
            workers=config.workers,
            bootstrap_iterations=min(config.bootstrap_iterations, 250),
        )
        record = {
            "epoch": epoch + 1,
            "training_loss": loss_sum / max(samples, 1),
            "validation_macro_iou": validation["macro_iou"],
            "validation_weighted_iou": validation["weighted_iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        metric = float(validation["macro_iou"])
        if metric > best_metric + 1e-4:
            best_metric = metric
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no candidate state")
    model.load_state_dict(best_state)
    model.to(device).eval()

    evaluation_names = ["train", "validation"]
    if evaluate_external:
        # Holdout pixels become visible to the process only after training/early-stopping has ended
        # and the selected best state has been restored. They cannot influence normalization,
        # class weights, augmentation, optimization, early stopping, or seed selection.
        external_scenes = [load_lazy_scene(path) for path in external_dirs]
        external_refs = patch_index(
            external_scenes,
            patch_size=config.patch_size,
            stride=config.stride,
            min_valid_fraction=config.min_valid_fraction,
        )
        datasets["external_test"] = PatchDataset(
            external_scenes,
            external_refs,
            mean,
            std,
            patch_size=config.patch_size,
            augment=False,
            seed=config.seed,
        )
        evaluation_names.append("external_test")
    evaluation = {
        name: _evaluate(
            model,
            datasets[name],
            device=device,
            batch_size=config.batch_size,
            workers=config.workers,
            bootstrap_iterations=config.bootstrap_iterations,
        )
        for name in evaluation_names
    }
    evaluation_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "task": "worldcover_landcover_segmentation",
        "dataset_signature": data_signature,
        "experiment_signature": signature,
        "device": device,
        "config": asdict(config),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "class_names": list(CLASS_NAMES),
        "worldcover_codes": list(WORLDCOVER_CODES),
        "class_weights": weights.tolist(),
        "splits": {
            "train": list(SHOWCASE_SPLIT.train),
            "validation": list(SHOWCASE_SPLIT.validation),
            **(
                {"external_test": list(SHOWCASE_SPLIT.external_test)}
                if evaluate_external
                else {}
            ),
        },
        "patch_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "history": history,
        **evaluation,
    }
    evaluation_path = reports_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(evaluation_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (reports_dir / "training-history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if not export_bundle:
        return evaluation_payload

    if not evaluate_external:
        raise ValueError("exported showcase bundle requires external evaluation")
    model_cpu = model.to("cpu").eval()
    example = torch.zeros(1, len(INPUT_BANDS), config.patch_size, config.patch_size)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_dir.name}.tmp-", dir=bundle_dir.parent))
    try:
        model_path = temporary / "landcover_multispectral.pt2"
        exported = torch.export.export(model_cpu, (example,))
        torch.export.save(exported, model_path)
        (temporary / "normalization.json").write_text(
            json.dumps(
                {"bands": INPUT_BANDS, "mean": mean.tolist(), "std": std.tolist()},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        (temporary / "classes.json").write_text(
            json.dumps(
                {"class_names": CLASS_NAMES, "worldcover_codes": WORLDCOVER_CODES},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        (temporary / "training-config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (temporary / "MODEL_CARD.md").write_text(
            _model_card(
                evaluation=evaluation_payload,
                dataset_sha256=data_signature,
                experiment_sha256=signature,
                config=config,
                device=device,
            )
        )
        _write_bundle_manifest(
            temporary,
            {
                "schema_version": "1.0",
                "task": "worldcover_landcover_segmentation",
                "dataset_signature": data_signature,
                "experiment_signature": signature,
                "model_file": model_path.name,
                "input_bands": list(INPUT_BANDS),
                "class_names": list(CLASS_NAMES),
                "safe_loading": (
                    "SHA-256 verify bundle before torch.export.load; "
                    "no pickle checkpoint required"
                ),
            },
        )
        verify_bundle(temporary)

        backup = bundle_dir.with_name(f".{bundle_dir.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        if bundle_dir.exists():
            os.replace(bundle_dir, backup)
        try:
            os.replace(temporary, bundle_dir)
        except BaseException:
            if backup.exists() and not bundle_dir.exists():
                os.replace(backup, bundle_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return evaluation_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the flagship six-band land-cover model")
    parser.add_argument("--data-root", type=Path, default=Path("data/curated"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("models/landcover"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/landcover"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--base-channels", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        base_channels=args.base_channels,
        patience=args.patience,
        workers=args.workers,
    )
    result = train_landcover(
        args.data_root,
        args.bundle_dir,
        args.reports_dir,
        config=config,
        requested_device=args.device,
        force=args.force,
        evaluate_external=not args.validation_only,
        export_bundle=not args.validation_only,
    )
    key = "validation" if args.validation_only else "external_test"
    print(json.dumps({f"{key}_macro_iou": result[key]["macro_iou"]}, indent=2))


if __name__ == "__main__":
    main()

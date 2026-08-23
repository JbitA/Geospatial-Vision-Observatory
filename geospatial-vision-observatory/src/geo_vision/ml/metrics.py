from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def confusion_matrix(
    truth: npt.NDArray[np.integer[Any]],
    prediction: npt.NDArray[np.integer[Any]],
    classes: int,
    *,
    ignore_index: int = 255,
) -> npt.NDArray[np.int64]:
    valid = truth != ignore_index
    truth_v = truth[valid].astype(np.int64)
    pred_v = prediction[valid].astype(np.int64)
    include = (truth_v >= 0) & (truth_v < classes) & (pred_v >= 0) & (pred_v < classes)
    packed = truth_v[include] * classes + pred_v[include]
    return np.bincount(packed, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(matrix: npt.NDArray[np.integer[Any]]) -> dict[str, Any]:
    matrix_f = matrix.astype(np.float64)
    tp = np.diag(matrix_f)
    truth_count = matrix_f.sum(axis=1)
    pred_count = matrix_f.sum(axis=0)
    union = truth_count + pred_count - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    dice_den = truth_count + pred_count
    dice = np.divide(2 * tp, dice_den, out=np.full_like(tp, np.nan), where=dice_den > 0)
    present = truth_count > 0
    total = matrix_f.sum()
    pixel_accuracy = float(tp.sum() / total) if total else math.nan
    macro_iou = float(np.nanmean(iou[present])) if present.any() else math.nan
    macro_dice = float(np.nanmean(dice[present])) if present.any() else math.nan
    weighted_iou = (
        float(np.nansum(iou[present] * truth_count[present]) / truth_count[present].sum())
        if present.any()
        else math.nan
    )
    def finite_list(values: npt.NDArray[np.float64]) -> list[float | None]:
        return [float(value) if np.isfinite(value) else None for value in values]

    return {
        "pixel_accuracy": pixel_accuracy,
        "macro_iou": macro_iou,
        "macro_dice": macro_dice,
        "weighted_iou": weighted_iou,
        "iou": finite_list(iou),
        "dice": finite_list(dice),
        "support": truth_count.astype(int).tolist(),
    }


def expected_calibration_error(
    confidence: npt.NDArray[np.floating[Any]],
    correct: npt.NDArray[np.bool_],
    *,
    bins: int = 15,
) -> float:
    if confidence.size == 0:
        return math.nan
    confidence = np.clip(confidence.astype(np.float64), 0, 1)
    correct_f = correct.astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct_f[mask].mean()) - float(confidence[mask].mean())
            )
    return result


def bootstrap_macro_iou_ci(
    patch_confusions: Sequence[npt.NDArray[np.int64]],
    *,
    iterations: int = 1_000,
    seed: int = 20260823,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not patch_confusions:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(patch_confusions)
    stack = np.stack(patch_confusions)
    for _ in range(iterations):
        sampled = rng.integers(0, n, size=n)
        matrix = stack[sampled].sum(axis=0)
        value = float(metrics_from_confusion(matrix)["macro_iou"])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)

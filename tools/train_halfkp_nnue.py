#!/usr/bin/env python3
"""Train a runtime-compatible nonlinear HalfKP KNNUE model.

The trainer mirrors ``src/nnue.cpp`` exactly:

    score = bias + V * clipped_relu(b + W * features_for_us)
                  - V * clipped_relu(b + W * features_for_them)

Validation is grouped by complete games (or supplied as a separately curated
game split), the selected output is the epoch with the best held-out RMSE, and
training can stop when that metric no longer improves. Current teacher data
appends game_id and sample_kind columns after the existing result, score, FEN,
ply, bucket, opening, and termination fields. Legacy files infer contiguous
game groups from their increasing ply column.
"""

import argparse
import hashlib
import json
import math
import os
import random
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from train_bootstrap_nnue import (
    HALFKP_INPUT_SIZE,
    HALFKP_STRIDE,
    halfkp_piece_index,
    oriented_sq,
    parse_board_state,
)
from curate_dataset import material_class_from_fen, phase_from_position, ply_from_fen


@dataclass
class HalfKPSample:
    us: List[int]
    them: List[int]
    target: float
    group: str
    kind: str = "regular"
    bucket: str = "unknown"
    phase: str = "unknown"
    material: str = "unknown"
    result_stm: int = 0
    teacher_cp: int = 0


def require_torch():
    """Import PyTorch only when the optional backend is selected."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "the torch backend requires PyTorch; enter `nix develop .#cuda` "
            "or install a CPU-only PyTorch build for CPU testing"
        ) from error
    return torch


def resolve_torch_device(requested: str):
    torch = require_torch()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def create_torch_model(
    hidden_size: int,
    feature_weights: np.ndarray | None = None,
    hidden_bias: np.ndarray | None = None,
    output_weights: np.ndarray | None = None,
    output_bias: float | None = None,
):
    """Create the sparse PyTorch reference implementation of Kepler HalfKP."""
    torch = require_torch()
    padding_index = HALFKP_INPUT_SIZE

    class TorchHalfKP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_weights = torch.nn.Embedding(
                HALFKP_INPUT_SIZE + 1,
                hidden_size,
                padding_idx=padding_index,
                sparse=True,
            )
            self.hidden_bias = torch.nn.Parameter(torch.empty(hidden_size))
            self.output_weights = torch.nn.Parameter(torch.empty(hidden_size))
            self.output_bias = torch.nn.Parameter(torch.empty(()))

        def forward(self, us_features, them_features):
            us_pre = self.feature_weights(us_features).sum(dim=1) + self.hidden_bias
            them_pre = self.feature_weights(them_features).sum(dim=1) + self.hidden_bias
            difference = torch.relu(us_pre) - torch.relu(them_pre)
            return difference @ self.output_weights + self.output_bias

    model = TorchHalfKP()
    with torch.no_grad():
        model.feature_weights.weight.zero_()
        if feature_weights is not None:
            model.feature_weights.weight[:HALFKP_INPUT_SIZE].copy_(
                torch.from_numpy(np.asarray(feature_weights, dtype=np.float32))
            )
        if hidden_bias is None:
            model.hidden_bias.zero_()
        else:
            model.hidden_bias.copy_(
                torch.from_numpy(np.asarray(hidden_bias, dtype=np.float32))
            )
        if output_weights is None:
            model.output_weights.zero_()
        else:
            model.output_weights.copy_(
                torch.from_numpy(np.asarray(output_weights, dtype=np.float32))
            )
        model.output_bias.fill_(0.0 if output_bias is None else float(output_bias))
    return model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def halfkp_index(king_sq: int, perspective: int, piece: int, sq: int) -> int:
    mapped = halfkp_piece_index(piece, perspective)
    if mapped < 0:
        return -1
    return (
        oriented_sq(king_sq, perspective) * HALFKP_STRIDE
        + mapped * 64
        + oriented_sq(sq, perspective)
    )


def perspective_features(
    pieces: Sequence[Tuple[int, int]], kings: Sequence[int], perspective: int
) -> List[int]:
    king_sq = kings[perspective]
    if king_sq < 0:
        return []
    features: List[int] = []
    for piece, sq in pieces:
        feature = halfkp_index(king_sq, perspective, piece, sq)
        if feature >= 0:
            features.append(feature)
    return features


def parse_samples(
    path: str,
    result_weight: float,
    cp_scale: float,
    target_cp: float,
    target_mode: str = "legacy-tanh",
    target_clip: float = 2000.0,
    wdl_cp: float = 600.0,
) -> List[HalfKPSample]:
    samples: List[HalfKPSample] = []
    alpha = max(0.0, min(1.0, result_weight))
    cp_scale = max(1.0, cp_scale)
    target_cp = max(1.0, target_cp)
    legacy_group = 0
    legacy_rows = 0
    previous_ply = -1

    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                result_stm = float(int(parts[0]))
                score_cp_stm = float(int(parts[1]))
            except ValueError:
                continue

            pieces, side, kings = parse_board_state(parts[2])
            if kings[0] < 0 or kings[1] < 0:
                continue
            us = 0 if side == "w" else 1
            them = 1 - us
            us_features = perspective_features(pieces, kings, us)
            them_features = perspective_features(pieces, kings, them)
            if not us_features or not them_features:
                continue

            explicit_group = parts[7].strip() if len(parts) > 7 else ""
            if explicit_group:
                group = explicit_group
            else:
                try:
                    ply = int(parts[3]) if len(parts) > 3 else previous_ply + 1
                except ValueError:
                    ply = previous_ply + 1
                if legacy_rows and (ply <= previous_ply or legacy_rows >= 64):
                    legacy_group += 1
                    legacy_rows = 0
                group = f"legacy-inferred-{legacy_group}"
                previous_ply = ply
                legacy_rows += 1

            kind = (
                parts[8].strip() if len(parts) > 8 and parts[8].strip() else "regular"
            )
            bucket = (
                parts[4].strip() if len(parts) > 4 and parts[4].strip() else "unknown"
            )
            fen = parts[2]
            ply = ply_from_fen(fen)
            phase = (
                parts[9].strip()
                if len(parts) > 9 and parts[9].strip()
                else phase_from_position(fen, ply, 20, 60)
            )
            material = (
                parts[10].strip()
                if len(parts) > 10 and parts[10].strip()
                else material_class_from_fen(fen)
            )
            if target_mode == "centipawn":
                score_target = max(-target_clip, min(target_clip, score_cp_stm))
                result_target = result_stm * max(1.0, wdl_cp)
                target = alpha * result_target + (1.0 - alpha) * score_target
            else:
                score_target = math.tanh(score_cp_stm / cp_scale)
                target = (alpha * result_stm + (1.0 - alpha) * score_target) * target_cp
            samples.append(
                HalfKPSample(
                    us_features,
                    them_features,
                    target,
                    group,
                    kind,
                    bucket,
                    phase,
                    material,
                    int(result_stm),
                    int(score_cp_stm),
                )
            )

    return samples


def split_samples_by_group(
    samples: Sequence[HalfKPSample], validation_split: float, seed: int
) -> Tuple[List[HalfKPSample], List[HalfKPSample], List[str], List[str]]:
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 2:
        raise ValueError("grouped validation requires at least two game groups")
    random.Random(seed).shuffle(groups)
    validation_count = int(round(len(groups) * max(0.01, min(0.8, validation_split))))
    validation_count = min(max(1, validation_count), len(groups) - 1)
    validation_groups = set(groups[:validation_count])
    training_groups = set(groups[validation_count:])
    train = [sample for sample in samples if sample.group in training_groups]
    validation = [sample for sample in samples if sample.group in validation_groups]
    return train, validation, sorted(training_groups), sorted(validation_groups)


def padded_features(samples: Sequence[HalfKPSample]) -> Tuple[np.ndarray, np.ndarray]:
    max_us = max(len(sample.us) for sample in samples)
    max_them = max(len(sample.them) for sample in samples)
    us = np.full((len(samples), max_us), -1, dtype=np.int32)
    them = np.full((len(samples), max_them), -1, dtype=np.int32)
    for row, sample in enumerate(samples):
        us[row, : len(sample.us)] = sample.us
        them[row, : len(sample.them)] = sample.them
    return us, them


def hidden_values(
    feature_weights: np.ndarray,
    hidden_bias: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    mask = features >= 0
    indexes = np.maximum(features, 0)
    return (feature_weights[indexes] * mask[:, :, None]).sum(axis=1) + hidden_bias


def predict(
    feature_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
    us_features: np.ndarray,
    them_features: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    prediction = np.empty(len(us_features), dtype=np.float32)
    for start in range(0, len(us_features), max(1, batch_size)):
        stop = min(len(us_features), start + max(1, batch_size))
        us_hidden = np.maximum(
            hidden_values(feature_weights, hidden_bias, us_features[start:stop]), 0.0
        )
        them_hidden = np.maximum(
            hidden_values(feature_weights, hidden_bias, them_features[start:stop]), 0.0
        )
        prediction[start:stop] = (
            us_hidden - them_hidden
        ) @ output_weights + output_bias
    return prediction


def torch_predict(
    model,
    us_features: np.ndarray,
    them_features: np.ndarray,
    device: str | Any = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """Run a Torch model on padded HalfKP indexes and return FP32 NumPy output."""
    torch = require_torch()
    resolved = torch.device(device)
    model = model.to(resolved)
    model.eval()
    prediction = np.empty(len(us_features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(us_features), max(1, batch_size)):
            stop = min(len(us_features), start + max(1, batch_size))
            us = torch.from_numpy(
                np.where(
                    us_features[start:stop] >= 0,
                    us_features[start:stop],
                    HALFKP_INPUT_SIZE,
                ).astype(np.int64, copy=False)
            ).to(resolved)
            them = torch.from_numpy(
                np.where(
                    them_features[start:stop] >= 0,
                    them_features[start:stop],
                    HALFKP_INPUT_SIZE,
                ).astype(np.int64, copy=False)
            ).to(resolved)
            prediction[start:stop] = model(us, them).float().cpu().numpy()
    return prediction


def git_revision() -> Dict[str, str | bool | None]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def flake_revision() -> Dict[str, str | None]:
    root = Path(__file__).resolve().parents[1]
    lock = root / "flake.lock"
    if not lock.is_file():
        return {"lock_path": None, "lock_sha256": None, "nixpkgs_revision": None}
    revision = None
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
        nixpkgs = data.get("nodes", {}).get("nixpkgs", {}).get("locked", {})
        revision = nixpkgs.get("rev")
    except (OSError, ValueError, TypeError):
        pass
    return {
        "lock_path": str(lock),
        "lock_sha256": sha256_file(lock),
        "nixpkgs_revision": revision,
    }


def training_provenance(
    backend: str,
    device: str,
    seed: int,
    batch_size: int,
    amp: bool,
    training_data: Path,
    validation_data: Path | None,
) -> Dict[str, object]:
    provenance: Dict[str, object] = {
        "backend": backend,
        "device": device,
        "seed": seed,
        "batch_size": batch_size,
        "amp": amp,
        "optimizer": "numpy-adam" if backend == "numpy" else "SparseAdam+Adam",
        "training_data_sha256": sha256_file(training_data),
        "validation_data_sha256": sha256_file(validation_data)
        if validation_data
        else None,
        "git": git_revision(),
        "flake": flake_revision(),
    }
    if backend == "torch":
        torch = require_torch()
        provenance["torch_version"] = torch.__version__
        provenance["cuda_runtime"] = torch.version.cuda
        if device == "cuda":
            provenance["gpu"] = torch.cuda.get_device_name(0)
            provenance["compute_capability"] = list(torch.cuda.get_device_capability(0))
        else:
            provenance["gpu"] = None
            provenance["compute_capability"] = None
    else:
        provenance.update(
            {
                "torch_version": None,
                "cuda_runtime": None,
                "gpu": None,
                "compute_capability": None,
            }
        )
    return provenance


def weighted_rmse(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> float:
    squared = np.square(prediction - target)
    return float(np.sqrt(np.sum(squared * weights) / max(1e-9, float(np.sum(weights)))))


def sample_weights(
    samples: Sequence[HalfKPSample],
    tactical_weight: float,
    score_balance_power: float = 0.0,
) -> np.ndarray:
    bucket_counts = Counter(sample.bucket for sample in samples)
    average_count = len(samples) / max(1, len(bucket_counts))
    power = max(0.0, min(1.0, score_balance_power))
    bucket_factors = {
        bucket: (average_count / count) ** power
        for bucket, count in bucket_counts.items()
    }
    normalization = sum(
        bucket_counts[bucket] * factor for bucket, factor in bucket_factors.items()
    ) / max(1, len(samples))
    return np.array(
        [
            (tactical_weight if sample.kind == "tactical" else 1.0)
            * bucket_factors[sample.bucket]
            / max(1e-9, normalization)
            for sample in samples
        ],
        dtype=np.float32,
    )


def prediction_summary(
    prediction: np.ndarray, targets: np.ndarray, samples: Sequence[HalfKPSample]
) -> Dict[str, float | int | None]:
    if len(samples) == 0:
        return {"count": 0, "rmse": None, "mae": None, "bias": None}
    errors = prediction.astype(np.float64) - targets.astype(np.float64)
    return {
        "count": len(samples),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
    }


def segmented_prediction_summary(
    prediction: np.ndarray,
    targets: np.ndarray,
    samples: Sequence[HalfKPSample],
    attribute: str,
) -> Dict[str, Dict[str, float | int | None]]:
    report: Dict[str, Dict[str, float | int | None]] = {}
    for value in sorted({str(getattr(sample, attribute)) for sample in samples}):
        indexes = [
            index
            for index, sample in enumerate(samples)
            if str(getattr(sample, attribute)) == value
        ]
        report[value] = prediction_summary(
            prediction[indexes], targets[indexes], [samples[index] for index in indexes]
        )
    return report


def train(
    train_us: np.ndarray,
    train_them: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    validation_us: np.ndarray,
    validation_them: np.ndarray,
    validation_targets: np.ndarray,
    validation_weights: np.ndarray,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ridge: float,
    seed: int,
    patience: int,
    min_epochs: int,
    min_delta: float,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    # Unseen king buckets remain neutral instead of retaining random weights.
    feature_weights = np.zeros((HALFKP_INPUT_SIZE, hidden_size), dtype=np.float32)
    hidden_bias = rng.normal(0.10, 0.01, hidden_size).astype(np.float32)
    output_weights = rng.normal(0.0, 0.05, hidden_size).astype(np.float32)
    output_bias = float(np.average(train_targets, weights=train_weights))

    mw = np.zeros_like(feature_weights)
    vw = np.zeros_like(feature_weights)
    mb = np.zeros_like(hidden_bias)
    vb = np.zeros_like(hidden_bias)
    mo = np.zeros_like(output_weights)
    vo = np.zeros_like(output_weights)
    m_out_bias = 0.0
    v_out_bias = 0.0
    step = 0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    order = np.arange(len(train_targets))
    batch_size = max(1, batch_size)
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_parameters: Tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, max(1, epochs) + 1):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            us_idx = train_us[batch]
            them_idx = train_them[batch]
            us_mask = us_idx >= 0
            them_mask = them_idx >= 0
            us_idx = np.maximum(us_idx, 0)
            them_idx = np.maximum(them_idx, 0)

            us_pre = (feature_weights[us_idx] * us_mask[:, :, None]).sum(
                axis=1
            ) + hidden_bias
            them_pre = (feature_weights[them_idx] * them_mask[:, :, None]).sum(
                axis=1
            ) + hidden_bias
            us_hidden = np.maximum(us_pre, 0.0)
            them_hidden = np.maximum(them_pre, 0.0)
            prediction = (us_hidden - them_hidden) @ output_weights + output_bias
            batch_weights = train_weights[batch]
            weighted_error = (prediction - train_targets[batch]) * batch_weights
            scale = 1.0 / max(1e-9, float(np.sum(batch_weights)))

            grad_output = ((us_hidden - them_hidden).T @ weighted_error) * scale
            grad_output += ridge * output_weights
            grad_output_bias = float(np.sum(weighted_error) * scale)
            grad_us = weighted_error[:, None] * output_weights[None, :]
            grad_them = -grad_us
            grad_us[us_pre <= 0.0] = 0.0
            grad_them[them_pre <= 0.0] = 0.0
            grad_hidden_bias = np.sum(grad_us + grad_them, axis=0) * scale

            feature_indexes = np.concatenate((us_idx[us_mask], them_idx[them_mask]))
            feature_gradients = np.concatenate(
                (
                    np.broadcast_to(grad_us[:, None, :], (*us_idx.shape, hidden_size))[
                        us_mask
                    ],
                    np.broadcast_to(
                        grad_them[:, None, :], (*them_idx.shape, hidden_size)
                    )[them_mask],
                )
            )
            feature_order = np.argsort(feature_indexes, kind="stable")
            sorted_indexes = feature_indexes[feature_order]
            group_starts = np.concatenate(
                ([0], np.flatnonzero(sorted_indexes[1:] != sorted_indexes[:-1]) + 1)
            )
            touched = sorted_indexes[group_starts]
            grad_touched = (
                np.add.reduceat(feature_gradients[feature_order], group_starts, axis=0)
                * scale
            )
            grad_touched += ridge * feature_weights[touched]

            step += 1
            correction1 = 1.0 - beta1**step
            correction2 = 1.0 - beta2**step
            mw[touched] = beta1 * mw[touched] + (1.0 - beta1) * grad_touched
            vw[touched] = beta2 * vw[touched] + (1.0 - beta2) * np.square(grad_touched)
            mb = beta1 * mb + (1.0 - beta1) * grad_hidden_bias
            vb = beta2 * vb + (1.0 - beta2) * np.square(grad_hidden_bias)
            mo = beta1 * mo + (1.0 - beta1) * grad_output
            vo = beta2 * vo + (1.0 - beta2) * np.square(grad_output)
            m_out_bias = beta1 * m_out_bias + (1.0 - beta1) * grad_output_bias
            v_out_bias = beta2 * v_out_bias + (1.0 - beta2) * grad_output_bias**2

            feature_weights[touched] -= (
                learning_rate
                * (mw[touched] / correction1)
                / (np.sqrt(vw[touched] / correction2) + epsilon)
            )
            hidden_bias -= (
                learning_rate
                * (mb / correction1)
                / (np.sqrt(vb / correction2) + epsilon)
            )
            output_weights -= (
                learning_rate
                * (mo / correction1)
                / (np.sqrt(vo / correction2) + epsilon)
            )
            output_bias -= (
                learning_rate
                * (m_out_bias / correction1)
                / (math.sqrt(v_out_bias / correction2) + epsilon)
            )

        monitor_count = min(len(train_us), 10000)
        train_prediction = predict(
            feature_weights,
            hidden_bias,
            output_weights,
            output_bias,
            train_us[:monitor_count],
            train_them[:monitor_count],
        )
        validation_prediction = predict(
            feature_weights,
            hidden_bias,
            output_weights,
            output_bias,
            validation_us,
            validation_them,
        )
        train_error = weighted_rmse(
            train_prediction,
            train_targets[:monitor_count],
            train_weights[:monitor_count],
        )
        validation_error = weighted_rmse(
            validation_prediction, validation_targets, validation_weights
        )
        improved = validation_error < best_validation - min_delta
        if improved:
            best_validation = validation_error
            best_epoch = epoch
            stale_epochs = 0
            best_parameters = (
                feature_weights.copy(),
                hidden_bias.copy(),
                output_weights.copy(),
                output_bias,
            )
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_rmse": train_error,
                "validation_rmse": validation_error,
            }
        )
        print(
            f"epoch {epoch}/{epochs} train_rmse {train_error:.4f} "
            f"validation_rmse {validation_error:.4f} best_epoch {best_epoch}",
            flush=True,
        )
        if epoch >= min_epochs and patience > 0 and stale_epochs >= patience:
            break

    assert best_parameters is not None
    return {
        "feature_weights": best_parameters[0],
        "hidden_bias": best_parameters[1],
        "output_weights": best_parameters[2],
        "output_bias": best_parameters[3],
        "optimizer_steps": step,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < epochs,
        "history": history,
    }


def _write_torch_checkpoint(
    torch, payload: Dict[str, object], directory: Path, epoch: int
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f"epoch-{epoch:04d}.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)

    latest = directory / "latest.pt"
    latest_temporary = directory / ".latest.pt.tmp"
    try:
        latest_temporary.unlink(missing_ok=True)
        latest_temporary.symlink_to(checkpoint.name)
        os.replace(latest_temporary, latest)
    except OSError:
        latest_temporary.unlink(missing_ok=True)
        torch.save(payload, latest_temporary)
        os.replace(latest_temporary, latest)
    return checkpoint


def train_torch(
    train_us: np.ndarray,
    train_them: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    validation_us: np.ndarray,
    validation_them: np.ndarray,
    validation_targets: np.ndarray,
    validation_weights: np.ndarray,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ridge: float,
    seed: int,
    patience: int,
    min_epochs: int,
    min_delta: float,
    device: str = "auto",
    amp: bool = False,
    checkpoint_dir: Path | None = None,
    resume_checkpoint: Path | None = None,
    checkpoint_metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Train HalfKP with sparse PyTorch embedding updates on CPU or CUDA."""
    torch = require_torch()
    resolved = resolve_torch_device(device)
    if amp and resolved.type != "cuda":
        raise ValueError("--amp requires --device cuda")

    torch.manual_seed(seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    initial_hidden = rng.normal(0.10, 0.01, hidden_size).astype(np.float32)
    initial_output = rng.normal(0.0, 0.05, hidden_size).astype(np.float32)
    initial_output_bias = float(np.average(train_targets, weights=train_weights))
    model = create_torch_model(
        hidden_size=hidden_size,
        hidden_bias=initial_hidden,
        output_weights=initial_output,
        output_bias=initial_output_bias,
    ).to(resolved)

    sparse_optimizer = torch.optim.SparseAdam(
        [model.feature_weights.weight], lr=learning_rate
    )
    dense_optimizer = torch.optim.Adam(
        [model.hidden_bias, model.output_weights, model.output_bias],
        lr=learning_rate,
    )
    use_amp = bool(amp and resolved.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    order = np.arange(len(train_targets))
    batch_size = max(1, batch_size)
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    optimizer_steps = 0
    best_state: Dict[str, Any] | None = None
    history: List[Dict[str, float]] = []
    start_epoch = 0
    configuration = {
        "backend": "torch",
        "input_size": HALFKP_INPUT_SIZE,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "ridge": ridge,
        "seed": seed,
    }

    if resume_checkpoint is not None:
        if not resume_checkpoint.is_file():
            raise ValueError(f"resume checkpoint does not exist: {resume_checkpoint}")
        checkpoint = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        if checkpoint.get("format") != "kepler-halfkp-torch-v1":
            raise ValueError(f"unsupported checkpoint format: {resume_checkpoint}")
        saved_configuration = checkpoint.get("configuration", {})
        for key, expected in configuration.items():
            if saved_configuration.get(key) != expected:
                raise ValueError(
                    f"checkpoint configuration mismatch for {key}: "
                    f"{saved_configuration.get(key)!r} != {expected!r}"
                )
        expected_metadata = checkpoint_metadata or {}
        saved_metadata = checkpoint.get("metadata", {})
        for key in ("training_data_sha256", "validation_data_sha256"):
            if expected_metadata.get(key) != saved_metadata.get(key):
                raise ValueError(f"checkpoint dataset mismatch for {key}")

        model.load_state_dict(checkpoint["model_state"])
        sparse_optimizer.load_state_dict(checkpoint["sparse_optimizer_state"])
        dense_optimizer.load_state_dict(checkpoint["dense_optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"])
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        history = list(checkpoint["history"])
        best_state = checkpoint["best_model_state"]
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if resolved.type == "cuda" and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

    train_target_tensor = torch.from_numpy(train_targets)
    train_weight_tensor = torch.from_numpy(train_weights)
    padding_index = HALFKP_INPUT_SIZE

    for epoch in range(start_epoch + 1, max(1, epochs) + 1):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            batch_us = torch.from_numpy(
                np.where(train_us[batch] >= 0, train_us[batch], padding_index).astype(
                    np.int64, copy=False
                )
            ).to(resolved)
            batch_them = torch.from_numpy(
                np.where(
                    train_them[batch] >= 0, train_them[batch], padding_index
                ).astype(np.int64, copy=False)
            ).to(resolved)
            targets = train_target_tensor[batch].to(resolved)
            weights = train_weight_tensor[batch].to(resolved)

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(batch_us, batch_them)
                error = prediction - targets
                loss = (
                    0.5
                    * torch.sum(weights * error.square())
                    / weights.sum().clamp_min(1e-9)
                )
                if ridge > 0.0:
                    touched = torch.unique(
                        torch.cat(
                            (
                                batch_us[batch_us != padding_index],
                                batch_them[batch_them != padding_index],
                            )
                        )
                    )
                    loss = loss + 0.5 * ridge * (
                        model.feature_weights(touched).square().sum()
                        + model.output_weights.square().sum()
                    )

            scaler.scale(loss).backward()
            scaler.step(sparse_optimizer)
            scaler.step(dense_optimizer)
            scaler.update()
            optimizer_steps += 1

        monitor_count = min(len(train_us), 10000)
        train_prediction = torch_predict(
            model,
            train_us[:monitor_count],
            train_them[:monitor_count],
            resolved,
        )
        validation_prediction = torch_predict(
            model,
            validation_us,
            validation_them,
            resolved,
        )
        train_error = weighted_rmse(
            train_prediction,
            train_targets[:monitor_count],
            train_weights[:monitor_count],
        )
        validation_error = weighted_rmse(
            validation_prediction, validation_targets, validation_weights
        )
        improved = validation_error < best_validation - min_delta
        if improved:
            best_validation = validation_error
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_rmse": train_error,
                "validation_rmse": validation_error,
            }
        )
        print(
            f"epoch {epoch}/{epochs} train_rmse {train_error:.4f} "
            f"validation_rmse {validation_error:.4f} best_epoch {best_epoch}",
            flush=True,
        )

        if checkpoint_dir is not None:
            payload: Dict[str, object] = {
                "format": "kepler-halfkp-torch-v1",
                "epoch": epoch,
                "configuration": configuration,
                "metadata": checkpoint_metadata or {},
                "model_state": model.state_dict(),
                "best_model_state": best_state,
                "sparse_optimizer_state": sparse_optimizer.state_dict(),
                "dense_optimizer_state": dense_optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "optimizer_steps": optimizer_steps,
                "history": history,
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all()
                if resolved.type == "cuda"
                else None,
            }
            saved = _write_torch_checkpoint(torch, payload, checkpoint_dir, epoch)
            print(f"checkpoint {saved}", flush=True)

        if epoch >= min_epochs and patience > 0 and stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Torch training did not produce a best-validation checkpoint"
        )
    model.load_state_dict(best_state)
    model = model.cpu()
    return {
        "feature_weights": model.feature_weights.weight[:HALFKP_INPUT_SIZE]
        .detach()
        .numpy()
        .copy(),
        "hidden_bias": model.hidden_bias.detach().numpy().copy(),
        "output_weights": model.output_weights.detach().numpy().copy(),
        "output_bias": float(model.output_bias.detach().item()),
        "optimizer_steps": optimizer_steps,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < epochs,
        "history": history,
        "device": resolved.type,
    }


def quantize(
    feature_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
    train_us: np.ndarray,
    train_them: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    sample_count = min(len(train_us), 1024)
    us_pre = hidden_values(feature_weights, hidden_bias, train_us[:sample_count])
    them_pre = hidden_values(feature_weights, hidden_bias, train_them[:sample_count])
    positive = np.maximum(
        np.concatenate((us_pre.reshape(-1), them_pre.reshape(-1))), 0.0
    )
    activation_p995 = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    hidden_scale = max(1.0, min(256.0, 220.0 / max(0.25, activation_p995)))
    output_max = float(np.max(np.abs(output_weights)))
    output_scale = max(16.0, min(256.0, 30000.0 / max(1.0, output_max)))
    scale = max(1, int(round(hidden_scale * output_scale)))

    feature_q = np.clip(np.rint(feature_weights * hidden_scale), -32768, 32767).astype(
        np.int16
    )
    hidden_q = np.clip(np.rint(hidden_bias * hidden_scale), -32768, 32767).astype(
        np.int16
    )
    output_q = np.clip(np.rint(output_weights * output_scale), -32767, 32767).astype(
        np.int16
    )
    output_both = np.concatenate((output_q, -output_q)).astype(np.int16)
    output_bias_q = int(
        np.clip(
            round(output_bias * hidden_scale * output_scale), -2147483648, 2147483647
        )
    )
    return feature_q, hidden_q, output_both, output_bias_q, scale


def quantized_predict(
    feature_q: np.ndarray,
    hidden_q: np.ndarray,
    output_q: np.ndarray,
    output_bias_q: int,
    scale: int,
    us_features: np.ndarray,
    them_features: np.ndarray,
) -> np.ndarray:
    feature_i32 = feature_q.astype(np.int32)
    hidden_i32 = hidden_q.astype(np.int32)
    hidden_size = len(hidden_q)
    first = output_q[:hidden_size].astype(np.int64)
    second = output_q[hidden_size:].astype(np.int64)
    prediction = np.empty(len(us_features), dtype=np.float64)
    for start in range(0, len(us_features), 1024):
        stop = min(len(us_features), start + 1024)
        us = np.clip(
            hidden_values(feature_i32, hidden_i32, us_features[start:stop]), 0, 255
        )
        them = np.clip(
            hidden_values(feature_i32, hidden_i32, them_features[start:stop]), 0, 255
        )
        raw = (
            us.astype(np.int64) @ first + them.astype(np.int64) @ second + output_bias_q
        )
        prediction[start:stop] = raw.astype(np.float64) / scale
    return prediction


def write_numpy_model(
    path: Path,
    feature_q: np.ndarray,
    hidden_q: np.ndarray,
    output_q: np.ndarray,
    output_bias_q: int,
    scale: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        output.write(b"KNNUEv1\x00")
        output.write(struct.pack("<iii", HALFKP_INPUT_SIZE, len(hidden_q), scale))
        output.write(feature_q.astype("<i2", copy=False).tobytes(order="C"))
        output.write(hidden_q.astype("<i2", copy=False).tobytes(order="C"))
        output.write(output_q.astype("<i2", copy=False).tobytes(order="C"))
        output.write(struct.pack("<i", output_bias_q))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a nonlinear HalfKP KNNUE model."
    )
    parser.add_argument(
        "--data", required=True, help="TSV path: result<TAB>score<TAB>fen..."
    )
    parser.add_argument(
        "--validation-data",
        default="",
        help="Optional separately curated validation TSV with disjoint game IDs.",
    )
    parser.add_argument(
        "--out", required=True, help="Best-validation .nnue output path."
    )
    parser.add_argument(
        "--metrics-out", default="", help="Defaults to OUT.metrics.json."
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision (Torch CUDA backend only).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Write an atomic Torch checkpoint after every completed epoch.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Resume a Torch run from an epoch checkpoint or latest.pt.",
    )
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--result-weight", type=float, default=0.1)
    parser.add_argument("--cp-scale", type=float, default=400.0)
    parser.add_argument("--target-cp", type=float, default=100.0)
    parser.add_argument(
        "--target-mode",
        choices=["centipawn", "legacy-tanh"],
        default="centipawn",
        help="Centipawn mode preserves eval units; legacy-tanh reproduces older runs.",
    )
    parser.add_argument("--target-clip", type=float, default=2000.0)
    parser.add_argument("--wdl-cp", type=float, default=600.0)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--tactical-weight", type=float, default=1.5)
    parser.add_argument(
        "--score-balance-power",
        type=float,
        default=0.5,
        help="Inverse-frequency score-bucket weighting power, 0 disables and 1 fully balances.",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-training-samples", type=int, default=0)
    parser.add_argument("--min-training-groups", type=int, default=0)
    parser.add_argument(
        "--max-generalization-gap",
        type=float,
        default=0.0,
        help="Maximum validation-minus-training RMSE; 0 disables the gate.",
    )
    parser.add_argument(
        "--strength-ready",
        action="store_true",
        help="Require the scale/diversity expected for a promotion-quality network.",
    )
    args = parser.parse_args()

    if args.backend == "numpy":
        if args.device == "cuda":
            raise SystemExit("the NumPy backend only supports --device auto or cpu")
        if args.amp:
            raise SystemExit("--amp is only supported by the Torch CUDA backend")
        if args.checkpoint_dir or args.resume_checkpoint:
            raise SystemExit(
                "checkpointing is currently supported by the Torch backend only"
            )

    if args.strength_ready:
        args.min_training_samples = max(args.min_training_samples, 800_000)
        args.min_training_groups = max(args.min_training_groups, 16_000)
        args.max_generalization_gap = (
            min(args.max_generalization_gap, 25.0)
            if args.max_generalization_gap > 0.0
            else 25.0
        )

    samples = parse_samples(
        args.data,
        args.result_weight,
        args.cp_scale,
        args.target_cp,
        args.target_mode,
        max(1.0, args.target_clip),
        max(1.0, args.wdl_cp),
    )
    if not samples:
        raise SystemExit("No valid HalfKP rows found in data file.")
    if args.validation_data:
        train_samples = samples
        validation_samples = parse_samples(
            args.validation_data,
            args.result_weight,
            args.cp_scale,
            args.target_cp,
            args.target_mode,
            max(1.0, args.target_clip),
            max(1.0, args.wdl_cp),
        )
        if not validation_samples:
            raise SystemExit("No valid HalfKP rows found in validation data file.")
        train_groups = sorted({sample.group for sample in train_samples})
        validation_groups = sorted({sample.group for sample in validation_samples})
        overlap = set(train_groups) & set(validation_groups)
        if overlap:
            examples = ", ".join(sorted(overlap)[:3])
            raise SystemExit(f"training/validation game leakage detected: {examples}")
    else:
        try:
            train_samples, validation_samples, train_groups, validation_groups = (
                split_samples_by_group(samples, args.validation_split, args.seed)
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

    if len(train_samples) < args.min_training_samples:
        raise SystemExit(
            f"training quality gate failed: {len(train_samples)} samples < {args.min_training_samples}"
        )
    if len(train_groups) < args.min_training_groups:
        raise SystemExit(
            f"training quality gate failed: {len(train_groups)} game groups < {args.min_training_groups}"
        )

    train_us, train_them = padded_features(train_samples)
    validation_us, validation_them = padded_features(validation_samples)
    train_targets = np.array(
        [sample.target for sample in train_samples], dtype=np.float32
    )
    validation_targets = np.array(
        [sample.target for sample in validation_samples], dtype=np.float32
    )
    train_weight_values = sample_weights(
        train_samples, max(1.0, args.tactical_weight), args.score_balance_power
    )
    validation_weight_values = sample_weights(
        validation_samples, max(1.0, args.tactical_weight), args.score_balance_power
    )
    hidden_size = max(8, min(1536, args.hidden_size))

    actual_device = (
        resolve_torch_device(args.device).type if args.backend == "torch" else "cpu"
    )
    training_path = Path(args.data).resolve()
    validation_path = (
        Path(args.validation_data).resolve() if args.validation_data else None
    )
    provenance = training_provenance(
        args.backend,
        actual_device,
        args.seed,
        max(1, args.batch_size),
        bool(args.amp),
        training_path,
        validation_path,
    )
    if args.backend == "torch":
        resume_checkpoint = (
            Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
        )
        checkpoint_dir = (
            Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
        )
        if resume_checkpoint is not None and checkpoint_dir is None:
            checkpoint_dir = resume_checkpoint.parent
        trained = train_torch(
            train_us,
            train_them,
            train_targets,
            train_weight_values,
            validation_us,
            validation_them,
            validation_targets,
            validation_weight_values,
            hidden_size,
            max(1, args.epochs),
            max(1, args.batch_size),
            max(1e-5, args.lr),
            max(0.0, args.ridge),
            args.seed,
            max(0, args.patience),
            max(1, args.min_epochs),
            max(0.0, args.min_delta),
            actual_device,
            bool(args.amp),
            checkpoint_dir,
            resume_checkpoint,
            provenance,
        )
    else:
        trained = train(
            train_us,
            train_them,
            train_targets,
            train_weight_values,
            validation_us,
            validation_them,
            validation_targets,
            validation_weight_values,
            hidden_size,
            max(1, args.epochs),
            max(1, args.batch_size),
            max(1e-5, args.lr),
            max(0.0, args.ridge),
            args.seed,
            max(0, args.patience),
            max(1, args.min_epochs),
            max(0.0, args.min_delta),
        )
    weights = trained["feature_weights"]
    hidden_bias = trained["hidden_bias"]
    output_weights = trained["output_weights"]
    output_bias = float(trained["output_bias"])
    train_prediction = predict(
        weights, hidden_bias, output_weights, output_bias, train_us, train_them
    )
    validation_prediction = predict(
        weights,
        hidden_bias,
        output_weights,
        output_bias,
        validation_us,
        validation_them,
    )
    train_error = weighted_rmse(train_prediction, train_targets, train_weight_values)
    validation_error = weighted_rmse(
        validation_prediction, validation_targets, validation_weight_values
    )
    feature_q, hidden_q, output_q, output_bias_q, scale = quantize(
        weights, hidden_bias, output_weights, output_bias, train_us, train_them
    )
    quantized_train = quantized_predict(
        feature_q, hidden_q, output_q, output_bias_q, scale, train_us, train_them
    )
    quantized_validation = quantized_predict(
        feature_q,
        hidden_q,
        output_q,
        output_bias_q,
        scale,
        validation_us,
        validation_them,
    )
    quantized_train_error = weighted_rmse(
        quantized_train, train_targets, train_weight_values
    )
    quantized_validation_error = weighted_rmse(
        quantized_validation, validation_targets, validation_weight_values
    )
    generalization_gap = quantized_validation_error - quantized_train_error
    output_path = Path(args.out).resolve()
    write_numpy_model(output_path, feature_q, hidden_q, output_q, output_bias_q, scale)

    metrics_path = (
        Path(args.metrics_out).resolve()
        if args.metrics_out
        else output_path.with_suffix(".metrics.json")
    )
    metrics = {
        "arch": "halfkp-dense",
        "backend": args.backend,
        "device": actual_device,
        "amp": bool(args.amp),
        "optimizer": provenance["optimizer"],
        "batch_size": max(1, args.batch_size),
        "seed": args.seed,
        "provenance": provenance,
        "input_size": HALFKP_INPUT_SIZE,
        "hidden_size": hidden_size,
        "samples": len(train_samples) + len(validation_samples),
        "training_data": str(Path(args.data).resolve()),
        "training_data_sha256": sha256_file(Path(args.data).resolve()),
        "validation_data": (
            str(Path(args.validation_data).resolve())
            if args.validation_data
            else "grouped split"
        ),
        "validation_data_sha256": (
            sha256_file(Path(args.validation_data).resolve())
            if args.validation_data
            else None
        ),
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "training_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "tactical_training_samples": sum(
            sample.kind == "tactical" for sample in train_samples
        ),
        "tactical_validation_samples": sum(
            sample.kind == "tactical" for sample in validation_samples
        ),
        "training_score_bucket_counts": dict(
            Counter(sample.bucket for sample in train_samples)
        ),
        "validation_score_bucket_counts": dict(
            Counter(sample.bucket for sample in validation_samples)
        ),
        "training_phase_counts": dict(
            Counter(sample.phase for sample in train_samples)
        ),
        "validation_phase_counts": dict(
            Counter(sample.phase for sample in validation_samples)
        ),
        "training_material_counts": dict(
            Counter(sample.material for sample in train_samples)
        ),
        "validation_material_counts": dict(
            Counter(sample.material for sample in validation_samples)
        ),
        "training_result_counts": dict(
            Counter(str(sample.result_stm) for sample in train_samples)
        ),
        "validation_result_counts": dict(
            Counter(str(sample.result_stm) for sample in validation_samples)
        ),
        "target": {
            "mode": args.target_mode,
            "result_weight": max(0.0, min(1.0, args.result_weight)),
            "teacher_cp_weight": 1.0 - max(0.0, min(1.0, args.result_weight)),
            "cp_scale": max(1.0, args.cp_scale),
            "target_cp": max(1.0, args.target_cp),
            "target_clip": max(1.0, args.target_clip),
            "wdl_cp": max(1.0, args.wdl_cp),
        },
        "score_balance_power": max(0.0, min(1.0, args.score_balance_power)),
        "best_epoch": trained["best_epoch"],
        "epochs_completed": trained["epochs_completed"],
        "early_stopped": trained["early_stopped"],
        "optimizer_steps": trained["optimizer_steps"],
        "epoch_training_monitor_samples": min(len(train_samples), 10000),
        "train_rmse": train_error,
        "validation_rmse": validation_error,
        "quantized_train_rmse": quantized_train_error,
        "quantized_validation_rmse": quantized_validation_error,
        "generalization_gap": generalization_gap,
        "quantized_validation_by_phase": segmented_prediction_summary(
            quantized_validation, validation_targets, validation_samples, "phase"
        ),
        "quantized_validation_by_material": segmented_prediction_summary(
            quantized_validation, validation_targets, validation_samples, "material"
        ),
        "quantized_validation_by_result": segmented_prediction_summary(
            quantized_validation, validation_targets, validation_samples, "result_stm"
        ),
        "quality_checks": {
            "minimum_training_samples_met": len(train_samples)
            >= args.min_training_samples,
            "minimum_training_groups_met": len(train_groups)
            >= args.min_training_groups,
            "generalization_gap_met": args.max_generalization_gap <= 0.0
            or generalization_gap <= args.max_generalization_gap,
            "strength_ready": args.strength_ready
            and len(train_samples) >= args.min_training_samples
            and len(train_groups) >= args.min_training_groups
            and generalization_gap <= args.max_generalization_gap,
        },
        "quant_scale": scale,
        "history": trained["history"],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if (
        args.max_generalization_gap > 0.0
        and generalization_gap > args.max_generalization_gap
    ):
        raise SystemExit(
            "training quality gate failed: generalization gap "
            f"{generalization_gap:.3f} > {args.max_generalization_gap:.3f}; metrics: {metrics_path}"
        )

    for key in (
        "arch",
        "backend",
        "device",
        "amp",
        "input_size",
        "hidden_size",
        "samples",
        "training_samples",
        "validation_samples",
        "training_groups",
        "validation_groups",
        "best_epoch",
        "epochs_completed",
        "early_stopped",
        "optimizer_steps",
        "train_rmse",
        "validation_rmse",
        "quantized_train_rmse",
        "quantized_validation_rmse",
        "quant_scale",
        "output",
    ):
        print(key, metrics[key])
    print("metrics", metrics_path)


if __name__ == "__main__":
    main()

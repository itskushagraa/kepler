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
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from train_bootstrap_nnue import (
    HALFKP_INPUT_SIZE,
    HALFKP_STRIDE,
    halfkp_piece_index,
    oriented_sq,
    parse_board_state,
)


@dataclass
class HalfKPSample:
    us: List[int]
    them: List[int]
    target: float
    group: str
    kind: str = "regular"


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

            kind = parts[8].strip() if len(parts) > 8 and parts[8].strip() else "regular"
            score_target = math.tanh(score_cp_stm / cp_scale)
            target = (alpha * result_stm + (1.0 - alpha) * score_target) * target_cp
            samples.append(HalfKPSample(us_features, them_features, target, group, kind))

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
        prediction[start:stop] = (us_hidden - them_hidden) @ output_weights + output_bias
    return prediction


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    squared = np.square(prediction - target)
    return float(np.sqrt(np.sum(squared * weights) / max(1e-9, float(np.sum(weights)))))


def sample_weights(samples: Sequence[HalfKPSample], tactical_weight: float) -> np.ndarray:
    return np.array(
        [tactical_weight if sample.kind == "tactical" else 1.0 for sample in samples],
        dtype=np.float32,
    )


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

            us_pre = (feature_weights[us_idx] * us_mask[:, :, None]).sum(axis=1) + hidden_bias
            them_pre = (feature_weights[them_idx] * them_mask[:, :, None]).sum(axis=1) + hidden_bias
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
                    np.broadcast_to(grad_us[:, None, :], (*us_idx.shape, hidden_size))[us_mask],
                    np.broadcast_to(grad_them[:, None, :], (*them_idx.shape, hidden_size))[them_mask],
                )
            )
            feature_order = np.argsort(feature_indexes, kind="stable")
            sorted_indexes = feature_indexes[feature_order]
            group_starts = np.concatenate(
                ([0], np.flatnonzero(sorted_indexes[1:] != sorted_indexes[:-1]) + 1)
            )
            touched = sorted_indexes[group_starts]
            grad_touched = np.add.reduceat(
                feature_gradients[feature_order], group_starts, axis=0
            ) * scale
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

            feature_weights[touched] -= learning_rate * (mw[touched] / correction1) / (
                np.sqrt(vw[touched] / correction2) + epsilon
            )
            hidden_bias -= learning_rate * (mb / correction1) / (
                np.sqrt(vb / correction2) + epsilon
            )
            output_weights -= learning_rate * (mo / correction1) / (
                np.sqrt(vo / correction2) + epsilon
            )
            output_bias -= learning_rate * (m_out_bias / correction1) / (
                math.sqrt(v_out_bias / correction2) + epsilon
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
            {"epoch": epoch, "train_rmse": train_error, "validation_rmse": validation_error}
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
    positive = np.maximum(np.concatenate((us_pre.reshape(-1), them_pre.reshape(-1))), 0.0)
    activation_p995 = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    hidden_scale = max(1.0, min(256.0, 220.0 / max(0.25, activation_p995)))
    output_max = float(np.max(np.abs(output_weights)))
    output_scale = max(16.0, min(256.0, 30000.0 / max(1.0, output_max)))
    scale = max(1, int(round(hidden_scale * output_scale)))

    feature_q = np.clip(np.rint(feature_weights * hidden_scale), -32768, 32767).astype(np.int16)
    hidden_q = np.clip(np.rint(hidden_bias * hidden_scale), -32768, 32767).astype(np.int16)
    output_q = np.clip(np.rint(output_weights * output_scale), -32767, 32767).astype(np.int16)
    output_both = np.concatenate((output_q, -output_q)).astype(np.int16)
    output_bias_q = int(
        np.clip(round(output_bias * hidden_scale * output_scale), -2147483648, 2147483647)
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
        us = np.clip(hidden_values(feature_i32, hidden_i32, us_features[start:stop]), 0, 255)
        them = np.clip(hidden_values(feature_i32, hidden_i32, them_features[start:stop]), 0, 255)
        raw = us.astype(np.int64) @ first + them.astype(np.int64) @ second + output_bias_q
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
    parser = argparse.ArgumentParser(description="Train a nonlinear HalfKP KNNUE model.")
    parser.add_argument("--data", required=True, help="TSV path: result<TAB>score<TAB>fen...")
    parser.add_argument(
        "--validation-data",
        default="",
        help="Optional separately curated validation TSV with disjoint game IDs.",
    )
    parser.add_argument("--out", required=True, help="Best-validation .nnue output path.")
    parser.add_argument("--metrics-out", default="", help="Defaults to OUT.metrics.json.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--result-weight", type=float, default=0.1)
    parser.add_argument("--cp-scale", type=float, default=400.0)
    parser.add_argument("--target-cp", type=float, default=100.0)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--tactical-weight", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = parse_samples(args.data, args.result_weight, args.cp_scale, args.target_cp)
    if not samples:
        raise SystemExit("No valid HalfKP rows found in data file.")
    if args.validation_data:
        train_samples = samples
        validation_samples = parse_samples(
            args.validation_data, args.result_weight, args.cp_scale, args.target_cp
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
            train_samples, validation_samples, train_groups, validation_groups = split_samples_by_group(
                samples, args.validation_split, args.seed
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

    train_us, train_them = padded_features(train_samples)
    validation_us, validation_them = padded_features(validation_samples)
    train_targets = np.array([sample.target for sample in train_samples], dtype=np.float32)
    validation_targets = np.array([sample.target for sample in validation_samples], dtype=np.float32)
    train_weight_values = sample_weights(train_samples, max(1.0, args.tactical_weight))
    validation_weight_values = sample_weights(validation_samples, max(1.0, args.tactical_weight))
    hidden_size = max(8, min(1536, args.hidden_size))

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
    train_prediction = predict(weights, hidden_bias, output_weights, output_bias, train_us, train_them)
    validation_prediction = predict(
        weights, hidden_bias, output_weights, output_bias, validation_us, validation_them
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
        feature_q, hidden_q, output_q, output_bias_q, scale, validation_us, validation_them
    )
    quantized_train_error = weighted_rmse(quantized_train, train_targets, train_weight_values)
    quantized_validation_error = weighted_rmse(
        quantized_validation, validation_targets, validation_weight_values
    )
    output_path = Path(args.out).resolve()
    write_numpy_model(output_path, feature_q, hidden_q, output_q, output_bias_q, scale)

    metrics_path = (
        Path(args.metrics_out).resolve()
        if args.metrics_out
        else output_path.with_suffix(".metrics.json")
    )
    metrics = {
        "arch": "halfkp-dense",
        "input_size": HALFKP_INPUT_SIZE,
        "hidden_size": hidden_size,
        "samples": len(train_samples) + len(validation_samples),
        "training_data": str(Path(args.data).resolve()),
        "training_data_sha256": sha256_file(Path(args.data).resolve()),
        "validation_data": (
            str(Path(args.validation_data).resolve()) if args.validation_data else "grouped split"
        ),
        "validation_data_sha256": (
            sha256_file(Path(args.validation_data).resolve()) if args.validation_data else None
        ),
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "training_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "tactical_training_samples": sum(sample.kind == "tactical" for sample in train_samples),
        "tactical_validation_samples": sum(sample.kind == "tactical" for sample in validation_samples),
        "best_epoch": trained["best_epoch"],
        "epochs_completed": trained["epochs_completed"],
        "early_stopped": trained["early_stopped"],
        "optimizer_steps": trained["optimizer_steps"],
        "epoch_training_monitor_samples": min(len(train_samples), 10000),
        "train_rmse": train_error,
        "validation_rmse": validation_error,
        "quantized_train_rmse": quantized_train_error,
        "quantized_validation_rmse": quantized_validation_error,
        "quant_scale": scale,
        "history": trained["history"],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    for key in (
        "arch", "input_size", "hidden_size", "samples", "training_samples",
        "validation_samples", "training_groups", "validation_groups", "best_epoch",
        "epochs_completed", "early_stopped", "optimizer_steps", "train_rmse",
        "validation_rmse", "quantized_train_rmse", "quantized_validation_rmse",
        "quant_scale", "output",
    ):
        print(key, metrics[key])
    print("metrics", metrics_path)


if __name__ == "__main__":
    main()

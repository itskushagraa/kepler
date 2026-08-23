#!/usr/bin/env python3
"""Train a small dense PSQT-style KNNUE model.

The original bootstrap trainer writes a sparse linear model into the KNNUE
container.  This trainer keeps the same file format and runtime compatibility,
but learns a real hidden layer with ReLU activations and Adam updates.
It intentionally starts with the classic 768-feature architecture; the
HalfKP model needs a separate memory-conscious trainer and validation plan.
"""

import argparse
import math
from typing import List, Sequence, Tuple

import numpy as np

from train_bootstrap_nnue import PSQT_INPUT_SIZE, Sample, parse_data, write_model


def samples_to_matrix(samples: Sequence[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    features = np.zeros((len(samples), PSQT_INPUT_SIZE), dtype=np.float32)
    targets = np.zeros(len(samples), dtype=np.float32)
    for row, sample in enumerate(samples):
        for feature, sign in sample.features:
            if 0 <= feature < PSQT_INPUT_SIZE:
                features[row, feature] += sign
        targets[row] = sample.target
    return features, targets


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def train(
    features: np.ndarray,
    targets: np.ndarray,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ridge: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    fan_in = max(1, features.shape[1])
    weights = rng.normal(0.0, math.sqrt(2.0 / fan_in), (fan_in, hidden_size)).astype(np.float32)
    hidden_bias = np.zeros(hidden_size, dtype=np.float32)
    output_weights = rng.normal(0.0, 0.05, hidden_size).astype(np.float32)
    output_bias = float(np.mean(targets))

    # Adam state for the three trainable tensors.
    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = np.zeros_like(hidden_bias)
    vb = np.zeros_like(hidden_bias)
    mo = np.zeros_like(output_weights)
    vo = np.zeros_like(output_weights)
    m_out_bias = 0.0
    v_out_bias = 0.0
    step = 0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    order = np.arange(len(features))
    for _ in range(max(1, epochs)):
        rng.shuffle(order)
        for start in range(0, len(order), max(1, batch_size)):
            batch = order[start : start + max(1, batch_size)]
            x = features[batch]
            y = targets[batch]

            hidden_pre = x @ weights + hidden_bias
            hidden = np.maximum(hidden_pre, 0.0)
            prediction = hidden @ output_weights + output_bias
            error = prediction - y

            scale = 1.0 / max(1, len(batch))
            grad_output = (hidden.T @ error) * scale + ridge * output_weights
            grad_output_bias = float(np.mean(error))
            grad_hidden = error[:, None] * output_weights[None, :]
            grad_hidden[hidden_pre <= 0.0] = 0.0
            grad_weights = (x.T @ grad_hidden) * scale + ridge * weights
            grad_hidden_bias = np.sum(grad_hidden, axis=0) * scale

            step += 1
            correction1 = 1.0 - beta1**step
            correction2 = 1.0 - beta2**step

            mw = beta1 * mw + (1.0 - beta1) * grad_weights
            vw = beta2 * vw + (1.0 - beta2) * np.square(grad_weights)
            mb = beta1 * mb + (1.0 - beta1) * grad_hidden_bias
            vb = beta2 * vb + (1.0 - beta2) * np.square(grad_hidden_bias)
            mo = beta1 * mo + (1.0 - beta1) * grad_output
            vo = beta2 * vo + (1.0 - beta2) * np.square(grad_output)
            m_out_bias = beta1 * m_out_bias + (1.0 - beta1) * grad_output_bias
            v_out_bias = beta2 * v_out_bias + (1.0 - beta2) * grad_output_bias**2

            rate = learning_rate
            weights -= rate * (mw / correction1) / (np.sqrt(vw / correction2) + epsilon)
            hidden_bias -= rate * (mb / correction1) / (np.sqrt(vb / correction2) + epsilon)
            output_weights -= rate * (mo / correction1) / (np.sqrt(vo / correction2) + epsilon)
            output_bias -= rate * (m_out_bias / correction1) / (math.sqrt(v_out_bias / correction2) + epsilon)

    hidden_pre = features @ weights + hidden_bias
    prediction = np.maximum(hidden_pre, 0.0) @ output_weights + output_bias
    return weights, hidden_bias, output_weights, output_bias, rmse(prediction, targets)


def quantize(
    weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
) -> Tuple[List[int], List[int], List[int], int, int]:
    # Keep most hidden activations below the runtime's 255 clamp while
    # retaining enough integer precision for a small model.
    max_hidden = float(np.max(np.abs(weights))) + float(np.max(np.abs(hidden_bias)))
    hidden_scale = max(1.0, min(32.0, 200.0 / max(1.0, max_hidden * 12.0)))
    output_scale = 128.0
    scale = max(1, int(round(hidden_scale * output_scale)))

    feature_weights = np.rint(weights * hidden_scale).astype(np.int16).reshape(-1).tolist()
    hidden_bias_q = np.rint(hidden_bias * hidden_scale).astype(np.int16).tolist()
    output_weights_q = np.rint(output_weights * output_scale).astype(np.int16).tolist()
    output_bias_q = int(round(output_bias * hidden_scale * output_scale))
    return feature_weights, hidden_bias_q, output_weights_q, output_bias_q, scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dense PSQT KNNUE model.")
    parser.add_argument("--data", required=True, help="TSV path: result<TAB>score<TAB>fen.")
    parser.add_argument("--out", required=True, help="Output .nnue path.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--result-weight", type=float, default=0.0)
    parser.add_argument("--cp-scale", type=float, default=400.0)
    parser.add_argument("--target-cp", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = parse_data(
        args.data,
        "psqt",
        args.result_weight,
        args.cp_scale,
        args.target_cp,
    )
    if not samples:
        raise SystemExit("No valid rows found in data file.")

    features, targets = samples_to_matrix(samples)
    weights, hidden_bias, output_weights, output_bias, train_rmse = train(
        features,
        targets,
        hidden_size=max(8, args.hidden_size),
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        learning_rate=max(1e-5, args.lr),
        ridge=max(0.0, args.ridge),
        seed=args.seed,
    )
    fw, hb, ow, ob, scale = quantize(weights, hidden_bias, output_weights, output_bias)
    write_model(
        path=args.out,
        input_size=PSQT_INPUT_SIZE,
        hidden_size=max(8, args.hidden_size),
        feature_weights=fw,
        hidden_bias=hb,
        output_weights=ow,
        output_bias=ob,
        scale=scale,
    )

    print("arch psqt-dense")
    print("input_size", PSQT_INPUT_SIZE)
    print("hidden_size", max(8, args.hidden_size))
    print("samples", len(samples))
    print("train_rmse", round(train_rmse, 4))
    print("quant_scale", scale)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

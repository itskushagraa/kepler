#!/usr/bin/env python3
import argparse
import math
import random
import struct
from dataclasses import dataclass
from typing import List, Tuple

INPUT_SIZE = 768
HIDDEN_SIZE = 128

# Piece order: WP WN WB WR WQ WK BP BN BB BR BQ BK
PIECE_COUNT_MAX = [8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1]


@dataclass
class Sample:
    features: List[int]  # sparse active feature indices (piece-square)
    target: float        # white-perspective target in centipawns


def parse_sparse_features(fen: str) -> Tuple[List[int], str]:
    parts = fen.split()
    board = parts[0]
    side = parts[1] if len(parts) > 1 else "w"

    features: List[int] = []
    sq = 56  # a8
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }
    for c in board:
        if c == "/":
            sq -= 16
            continue
        if c.isdigit():
            sq += int(c)
            continue
        p = piece_map.get(c)
        if p is not None:
            features.append(p * 64 + sq)
        sq += 1
    return features, side


def parse_data(path: str, result_weight: float, cp_scale: float, target_cp: float) -> List[Sample]:
    rows: List[Sample] = []
    alpha = max(0.0, min(1.0, result_weight))
    cp_scale = max(1.0, cp_scale)
    target_cp = max(1.0, target_cp)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            try:
                result_stm = float(int(parts[0]))      # -1/0/1 from side-to-move perspective
                score_cp_stm = float(int(parts[1]))    # cp from side-to-move perspective
            except ValueError:
                continue

            features, side = parse_sparse_features(parts[2])

            score_target = math.tanh(score_cp_stm / cp_scale)
            target_stm = (alpha * result_stm + (1.0 - alpha) * score_target) * target_cp

            # Convert to white perspective for training; engine flips sign for black-to-move at runtime.
            target_white = target_stm if side == "w" else -target_stm
            rows.append(Sample(features=features, target=target_white))

    return rows


def train_sparse_linear(
    samples: List[Sample],
    epochs: int,
    lr: float,
    ridge: float,
    seed: int,
) -> Tuple[List[float], float]:
    w = [0.0] * INPUT_SIZE
    b = 0.0
    g2 = [1e-6] * INPUT_SIZE  # AdaGrad accumulators
    b2 = 1e-6
    rng = random.Random(seed)

    idxs = list(range(len(samples)))
    for _ in range(max(1, epochs)):
        rng.shuffle(idxs)
        for i in idxs:
            s = samples[i]
            pred = b
            for fi in s.features:
                pred += w[fi]
            err = pred - s.target

            # Bias update
            gb = err
            b2 += gb * gb
            b -= lr * gb / math.sqrt(b2)

            # Sparse feature updates with L2 regularization
            for fi in s.features:
                g = err + ridge * w[fi]
                g2[fi] += g * g
                w[fi] -= lr * g / math.sqrt(g2[fi])

    return w, b


def rmse(samples: List[Sample], w: List[float], b: float) -> float:
    se = 0.0
    for s in samples:
        pred = b
        for fi in s.features:
            pred += w[fi]
        d = pred - s.target
        se += d * d
    return math.sqrt(se / max(1, len(samples)))


def build_model_params(w: List[float], b: float) -> Tuple[List[int], List[int], List[int], int, int]:
    feature_weights = [0] * (INPUT_SIZE * HIDDEN_SIZE)
    hidden_bias = [0] * HIDDEN_SIZE
    output_weights = [0] * HIDDEN_SIZE

    # Two channels per piece: positive and negative contributions.
    # channel = piece*2 for positive, piece*2+1 for negative.
    used_channels = 24
    for p in range(12):
        output_weights[p * 2] = 1
        output_weights[p * 2 + 1] = -1

    # Pick quantization scale to avoid hidden clamp saturation (0..255 in engine).
    max_abs_per_piece = [0.0] * 12
    for fi, ww in enumerate(w):
        p = fi // 64
        a = abs(ww)
        if a > max_abs_per_piece[p]:
            max_abs_per_piece[p] = a
    q_max = 64.0
    for p in range(12):
        denom = PIECE_COUNT_MAX[p] * max(1e-6, max_abs_per_piece[p])
        q_max = min(q_max, 220.0 / denom)
    q = int(max(1.0, min(32.0, q_max)))

    for fi, ww in enumerate(w):
        if ww == 0.0:
            continue
        p = fi // 64
        ch = p * 2 if ww > 0 else p * 2 + 1
        v = int(round(abs(ww) * q))
        v = max(0, min(32767, v))
        feature_weights[fi * HIDDEN_SIZE + ch] = v

    output_bias = int(round(b * q))
    scale = q

    # Keep unused channels explicitly zeroed for clarity.
    for ch in range(used_channels, HIDDEN_SIZE):
        output_weights[ch] = 0
        hidden_bias[ch] = 0

    return feature_weights, hidden_bias, output_weights, output_bias, scale


def write_model(
    path: str,
    feature_weights: List[int],
    hidden_bias: List[int],
    output_weights: List[int],
    output_bias: int,
    scale: int,
) -> None:
    with open(path, "wb") as out:
        out.write(b"KNNUEv1\x00")
        out.write(struct.pack("<iii", INPUT_SIZE, HIDDEN_SIZE, scale))
        out.write(struct.pack("<{}h".format(len(feature_weights)), *feature_weights))
        out.write(struct.pack("<{}h".format(len(hidden_bias)), *hidden_bias))
        out.write(struct.pack("<{}h".format(len(output_weights)), *output_weights))
        out.write(struct.pack("<i", output_bias))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a sparse piece-square bootstrap NNUE model.")
    ap.add_argument("--data", required=True, help="TSV path: result<TAB>score<TAB>fen.")
    ap.add_argument("--out", required=True, help="Output .nnue path.")
    ap.add_argument("--result-weight", type=float, default=0.6, help="Blend weight for game result target.")
    ap.add_argument("--cp-scale", type=float, default=300.0, help="Scale for tanh(score/cp_scale).")
    ap.add_argument("--target-cp", type=float, default=100.0, help="Scale blended target into centipawn-like units.")
    ap.add_argument("--ridge", type=float, default=1.0, help="L2 regularization strength.")
    ap.add_argument("--epochs", type=int, default=8, help="SGD epochs.")
    ap.add_argument("--lr", type=float, default=0.08, help="Base learning rate.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = ap.parse_args()

    samples = parse_data(args.data, args.result_weight, args.cp_scale, args.target_cp)
    if not samples:
        raise SystemExit("No valid rows found in data file.")

    w, b = train_sparse_linear(
        samples=samples,
        epochs=max(1, args.epochs),
        lr=max(1e-4, args.lr),
        ridge=max(0.0, args.ridge),
        seed=args.seed,
    )
    train_rmse = rmse(samples, w, b)

    fw, hb, ow, ob, scale = build_model_params(w, b)
    write_model(args.out, fw, hb, ow, ob, scale)

    # Diagnostics
    piece_values = []
    # Approx average per-piece contribution in white channels as a rough sanity check.
    for p in range(5):
        start = p * 64
        vals = [w[start + sq] for sq in range(64)]
        piece_values.append(round(sum(vals) / 64.0, 3))

    print("samples", len(samples))
    print("train_rmse", round(train_rmse, 4))
    print("approx_piece_avgs", piece_values)
    print("quant_scale", scale)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

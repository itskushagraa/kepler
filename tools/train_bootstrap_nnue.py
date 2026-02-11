#!/usr/bin/env python3
import argparse
import math
import random
import struct
from dataclasses import dataclass
from typing import List, Tuple

PSQT_INPUT_SIZE = 768
HALFKP_INPUT_SIZE = 41024
HALFKP_STRIDE = 641

DEFAULT_PSQT_HIDDEN = 128
DEFAULT_HALFKP_HIDDEN = 1536

# Piece order: WP WN WB WR WQ WK BP BN BB BR BQ BK
PIECE_COUNT_MAX_PSQT = [8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1]
PIECE_COUNT_MAX_HALFKP = [8, 2, 2, 2, 1, 8, 2, 2, 2, 1]


@dataclass
class Sample:
    # (feature index, sign) sparse pairs.
    features: List[Tuple[int, int]]
    # Side-to-move perspective target in centipawns.
    target: float


PIECE_MAP = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def parse_board_state(fen: str) -> Tuple[List[Tuple[int, int]], str, List[int]]:
    parts = fen.split()
    board = parts[0]
    side = parts[1] if len(parts) > 1 else "w"

    pieces: List[Tuple[int, int]] = []
    kings = [-1, -1]

    sq = 56  # a8
    for c in board:
        if c == "/":
            sq -= 16
            continue
        if c.isdigit():
            sq += int(c)
            continue
        p = PIECE_MAP.get(c)
        if p is not None:
            pieces.append((p, sq))
            if p == 5:
                kings[0] = sq
            elif p == 11:
                kings[1] = sq
        sq += 1

    return pieces, side, kings


def oriented_sq(sq: int, perspective: int) -> int:
    return sq if perspective == 0 else (sq ^ 56)


def halfkp_piece_index(piece: int, perspective: int) -> int:
    white_map = [0, 1, 2, 3, 4, -1, 5, 6, 7, 8, 9, -1]
    black_map = [5, 6, 7, 8, 9, -1, 0, 1, 2, 3, 4, -1]
    if piece < 0 or piece >= 12:
        return -1
    return white_map[piece] if perspective == 0 else black_map[piece]


def extract_psqt_features(pieces: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(p * 64 + sq, 1) for p, sq in pieces]


def extract_halfkp_features(
    pieces: List[Tuple[int, int]], kings: List[int], side: str
) -> List[Tuple[int, int]]:
    us = 0 if side == "w" else 1
    them = 1 - us

    if kings[0] < 0 or kings[1] < 0:
        return []

    out: List[Tuple[int, int]] = []
    for perspective, sign in ((us, 1), (them, -1)):
        king_sq = kings[perspective]
        king_oriented = oriented_sq(king_sq, perspective)
        base = king_oriented * HALFKP_STRIDE

        for piece, sq in pieces:
            mapped = halfkp_piece_index(piece, perspective)
            if mapped < 0:
                continue
            sq_oriented = oriented_sq(sq, perspective)
            fi = base + mapped * 64 + sq_oriented
            out.append((fi, sign))

    return out


def parse_data(
    path: str,
    arch: str,
    result_weight: float,
    cp_scale: float,
    target_cp: float,
) -> List[Sample]:
    rows: List[Sample] = []
    alpha = max(0.0, min(1.0, result_weight))
    cp_scale = max(1.0, cp_scale)
    target_cp = max(1.0, target_cp)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                result_stm = float(int(parts[0]))
                score_cp_stm = float(int(parts[1]))
            except ValueError:
                continue

            pieces, side, kings = parse_board_state(parts[2])
            if arch == "halfkp":
                features = extract_halfkp_features(pieces, kings, side)
            else:
                features = extract_psqt_features(pieces)

            if not features:
                continue

            score_target = math.tanh(score_cp_stm / cp_scale)
            target_stm = (alpha * result_stm + (1.0 - alpha) * score_target) * target_cp

            if arch == "halfkp":
                target = target_stm
            else:
                # Legacy psqt trainer predicts white perspective.
                target = target_stm if side == "w" else -target_stm

            rows.append(Sample(features=features, target=target))

    return rows


def train_sparse_linear(
    samples: List[Sample],
    input_size: int,
    epochs: int,
    lr: float,
    ridge: float,
    seed: int,
) -> Tuple[List[float], float]:
    w = [0.0] * input_size
    b = 0.0
    g2 = [1e-6] * input_size
    b2 = 1e-6
    rng = random.Random(seed)

    idxs = list(range(len(samples)))
    for _ in range(max(1, epochs)):
        rng.shuffle(idxs)
        for i in idxs:
            s = samples[i]
            pred = b
            for fi, sign in s.features:
                pred += sign * w[fi]
            err = pred - s.target

            gb = err
            b2 += gb * gb
            b -= lr * gb / math.sqrt(b2)

            for fi, sign in s.features:
                g = err * sign + ridge * w[fi]
                g2[fi] += g * g
                w[fi] -= lr * g / math.sqrt(g2[fi])

    return w, b


def rmse(samples: List[Sample], w: List[float], b: float) -> float:
    se = 0.0
    for s in samples:
        pred = b
        for fi, sign in s.features:
            pred += sign * w[fi]
        d = pred - s.target
        se += d * d
    return math.sqrt(se / max(1, len(samples)))


def feature_piece_bucket(fi: int, arch: str) -> int:
    if arch == "halfkp":
        inner = fi % HALFKP_STRIDE
        if inner >= 640:
            return -1
        return inner // 64
    return fi // 64


def build_model_params(
    w: List[float],
    b: float,
    input_size: int,
    hidden_size: int,
    arch: str,
) -> Tuple[List[int], List[int], List[int], int, int, int]:
    feature_weights = [0] * (input_size * hidden_size)
    hidden_bias = [0] * hidden_size

    if arch == "halfkp":
        output_weights = [0] * (hidden_size * 2)
        piece_count_max = PIECE_COUNT_MAX_HALFKP
        used_piece_types = 10
    else:
        output_weights = [0] * hidden_size
        piece_count_max = PIECE_COUNT_MAX_PSQT
        used_piece_types = 12

    used_channels = used_piece_types * 2

    if arch == "halfkp":
        for p in range(used_piece_types):
            pos_ch = p * 2
            neg_ch = p * 2 + 1
            output_weights[pos_ch] = 1
            output_weights[neg_ch] = -1
            output_weights[hidden_size + pos_ch] = -1
            output_weights[hidden_size + neg_ch] = 1
    else:
        for p in range(used_piece_types):
            output_weights[p * 2] = 1
            output_weights[p * 2 + 1] = -1

    max_abs_per_piece = [0.0] * used_piece_types
    for fi, ww in enumerate(w):
        p = feature_piece_bucket(fi, arch)
        if p < 0 or p >= used_piece_types:
            continue
        a = abs(ww)
        if a > max_abs_per_piece[p]:
            max_abs_per_piece[p] = a

    q_max = 64.0
    for p in range(used_piece_types):
        denom = piece_count_max[p] * max(1e-6, max_abs_per_piece[p])
        q_max = min(q_max, 220.0 / denom)
    q = int(max(1.0, min(32.0, q_max)))

    for fi, ww in enumerate(w):
        if ww == 0.0:
            continue
        p = feature_piece_bucket(fi, arch)
        if p < 0 or p >= used_piece_types:
            continue

        ch = p * 2 if ww > 0 else p * 2 + 1
        v = int(round(abs(ww) * q))
        v = max(0, min(32767, v))
        feature_weights[fi * hidden_size + ch] = v

    output_bias = int(round(b * q))
    scale = q

    return feature_weights, hidden_bias, output_weights, output_bias, scale, used_channels


def write_model(
    path: str,
    input_size: int,
    hidden_size: int,
    feature_weights: List[int],
    hidden_bias: List[int],
    output_weights: List[int],
    output_bias: int,
    scale: int,
) -> None:
    with open(path, "wb") as out:
        out.write(b"KNNUEv1\x00")
        out.write(struct.pack("<iii", input_size, hidden_size, scale))
        out.write(struct.pack("<{}h".format(len(feature_weights)), *feature_weights))
        out.write(struct.pack("<{}h".format(len(hidden_bias)), *hidden_bias))
        out.write(struct.pack("<{}h".format(len(output_weights)), *output_weights))
        out.write(struct.pack("<i", output_bias))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a sparse bootstrap NNUE model.")
    ap.add_argument("--data", required=True, help="TSV path: result<TAB>score<TAB>fen.")
    ap.add_argument("--out", required=True, help="Output .nnue path.")
    ap.add_argument("--arch", choices=["psqt", "halfkp"], default="halfkp", help="Network input architecture.")
    ap.add_argument("--hidden-size", type=int, default=0, help="Hidden layer size (default depends on arch).")
    ap.add_argument("--result-weight", type=float, default=0.4, help="Blend weight for game result target.")
    ap.add_argument("--cp-scale", type=float, default=300.0, help="Scale for tanh(score/cp_scale).")
    ap.add_argument("--target-cp", type=float, default=100.0, help="Scale blended target into centipawn-like units.")
    ap.add_argument("--ridge", type=float, default=0.5, help="L2 regularization strength.")
    ap.add_argument("--epochs", type=int, default=80, help="SGD epochs.")
    ap.add_argument("--lr", type=float, default=0.08, help="Base learning rate.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = ap.parse_args()

    input_size = HALFKP_INPUT_SIZE if args.arch == "halfkp" else PSQT_INPUT_SIZE
    default_hidden = DEFAULT_HALFKP_HIDDEN if args.arch == "halfkp" else DEFAULT_PSQT_HIDDEN
    hidden_size = args.hidden_size if args.hidden_size > 0 else default_hidden

    samples = parse_data(args.data, args.arch, args.result_weight, args.cp_scale, args.target_cp)
    if not samples:
        raise SystemExit("No valid rows found in data file.")

    w, b = train_sparse_linear(
        samples=samples,
        input_size=input_size,
        epochs=max(1, args.epochs),
        lr=max(1e-4, args.lr),
        ridge=max(0.0, args.ridge),
        seed=args.seed,
    )
    train_rmse = rmse(samples, w, b)

    fw, hb, ow, ob, scale, used_channels = build_model_params(
        w=w,
        b=b,
        input_size=input_size,
        hidden_size=max(24, hidden_size),
        arch=args.arch,
    )
    write_model(
        path=args.out,
        input_size=input_size,
        hidden_size=max(24, hidden_size),
        feature_weights=fw,
        hidden_bias=hb,
        output_weights=ow,
        output_bias=ob,
        scale=scale,
    )

    print("arch", args.arch)
    print("input_size", input_size)
    print("hidden_size", max(24, hidden_size))
    print("samples", len(samples))
    print("train_rmse", round(train_rmse, 4))
    print("trained_units", list(range(used_channels)))
    print("quant_scale", scale)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

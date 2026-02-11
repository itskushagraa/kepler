#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_bootstrap_nnue import (  # type: ignore
    DEFAULT_HALFKP_HIDDEN,
    DEFAULT_PSQT_HIDDEN,
    HALFKP_INPUT_SIZE,
    PSQT_INPUT_SIZE,
    build_model_params,
    parse_data,
    rmse,
    write_model,
)


def train_epoch(
    samples,
    w: List[float],
    b: float,
    g2: List[float],
    b2: float,
    lr: float,
    ridge: float,
    rng: random.Random,
) -> Tuple[List[float], float, List[float], float]:
    idxs = list(range(len(samples)))
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

    return w, b, g2, b2


def save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train NNUE with validation tracking and checkpoints.")
    ap.add_argument("--train", required=True, help="Training TSV dataset.")
    ap.add_argument("--val", required=True, help="Validation TSV dataset.")
    ap.add_argument("--out-nnue", required=True, help="Output .nnue path.")
    ap.add_argument("--out-metrics", required=True, help="Output metrics JSON path.")
    ap.add_argument("--checkpoint-dir", default="", help="Directory for checkpoints (optional).")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint every N epochs (0 disables).")
    ap.add_argument("--val-every", type=int, default=1, help="Evaluate val RMSE every N epochs.")
    ap.add_argument("--log-every", type=int, default=1, help="Print progress every N epochs (0 disables).")
    ap.add_argument("--verbose", action="store_true", help="Print per-epoch progress.")
    ap.add_argument("--arch", choices=["psqt", "halfkp"], default="halfkp", help="Network input architecture.")
    ap.add_argument("--hidden-size", type=int, default=0, help="Hidden layer size (default depends on arch).")
    ap.add_argument("--result-weight", type=float, default=0.0, help="Blend weight for game result target.")
    ap.add_argument("--cp-scale", type=float, default=400.0, help="Scale for tanh(score/cp_scale).")
    ap.add_argument("--target-cp", type=float, default=100.0, help="Scale blended target into centipawn-like units.")
    ap.add_argument("--ridge", type=float, default=0.2, help="L2 regularization strength.")
    ap.add_argument("--epochs", type=int, default=120, help="SGD epochs.")
    ap.add_argument("--lr", type=float, default=0.03, help="Base learning rate.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = ap.parse_args()

    input_size = HALFKP_INPUT_SIZE if args.arch == "halfkp" else PSQT_INPUT_SIZE
    default_hidden = DEFAULT_HALFKP_HIDDEN if args.arch == "halfkp" else DEFAULT_PSQT_HIDDEN
    hidden_size = args.hidden_size if args.hidden_size > 0 else default_hidden

    train_data = parse_data(args.train, args.arch, args.result_weight, args.cp_scale, args.target_cp)
    val_data = parse_data(args.val, args.arch, args.result_weight, args.cp_scale, args.target_cp)
    if not train_data:
        raise SystemExit("No valid training rows found.")
    if not val_data:
        raise SystemExit("No valid validation rows found.")

    w = [0.0] * input_size
    b = 0.0
    g2 = [1e-6] * input_size
    b2 = 1e-6

    epochs = max(1, args.epochs)
    lr = max(1e-4, args.lr)
    ridge = max(0.0, args.ridge)
    val_every = max(1, args.val_every)
    ckpt_every = max(0, args.checkpoint_every)
    log_every = max(0, args.log_every)

    metrics = {
        "config": {
            "train": args.train,
            "val": args.val,
            "arch": args.arch,
            "hidden_size": max(24, hidden_size),
            "result_weight": args.result_weight,
            "cp_scale": args.cp_scale,
            "target_cp": args.target_cp,
            "ridge": ridge,
            "epochs": epochs,
            "lr": lr,
            "seed": args.seed,
        },
        "epochs": [],
    }

    ckpt_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        w, b, g2, b2 = train_epoch(train_data, w, b, g2, b2, lr, ridge, rng)
        train_rmse = rmse(train_data, w, b)
        val_rmse = None
        if epoch % val_every == 0:
            val_rmse = rmse(val_data, w, b)
        metrics["epochs"].append(
            {
                "epoch": epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "elapsed_sec": round(time.time() - start, 2),
                "epoch_sec": round(time.time() - t0, 2),
            }
        )

        if ckpt_dir and ckpt_every > 0 and (epoch % ckpt_every == 0 or epoch == epochs):
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pkl"
            save_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch,
                    "w": w,
                    "b": b,
                    "g2": g2,
                    "b2": b2,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                },
            )
            if args.verbose:
                print(f"[ckpt] wrote {ckpt_path}")

        if args.verbose and log_every > 0 and (epoch % log_every == 0 or epoch == epochs):
            avg_epoch = (time.time() - start) / epoch
            eta = avg_epoch * (epochs - epoch)
            val_str = f"{val_rmse:.4f}" if val_rmse is not None else "n/a"
            print(
                f"[epoch {epoch:4d}/{epochs}] "
                f"train_rmse={train_rmse:.4f} val_rmse={val_str} "
                f"elapsed_sec={time.time() - start:.1f} eta_sec={eta:.1f}"
            )

    fw, hb, ow, ob, scale, used_channels = build_model_params(
        w=w,
        b=b,
        input_size=input_size,
        hidden_size=max(24, hidden_size),
        arch=args.arch,
    )
    write_model(
        path=args.out_nnue,
        input_size=input_size,
        hidden_size=max(24, hidden_size),
        feature_weights=fw,
        hidden_bias=hb,
        output_weights=ow,
        output_bias=ob,
        scale=scale,
    )

    metrics["final"] = {
        "train_rmse": metrics["epochs"][-1]["train_rmse"],
        "val_rmse": metrics["epochs"][-1]["val_rmse"],
        "trained_units": list(range(used_channels)),
        "quant_scale": scale,
        "out_nnue": args.out_nnue,
    }

    Path(args.out_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", args.out_nnue)
    print("wrote", args.out_metrics)


if __name__ == "__main__":
    main()

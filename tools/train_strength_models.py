#!/usr/bin/env python3
"""Train matched H128/H256 models from a strength-ready curated dataset."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "tools" / "train_halfkp_nnue.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train promotion-quality matched H128/H256 models.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--result-weight", type=float, default=0.15)
    parser.add_argument("--target-clip", type=float, default=2000.0)
    parser.add_argument("--wdl-cp", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-strength-gate", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset_dir).resolve()
    train = dataset / "train.tsv"
    validation = dataset / "validation.tsv"
    curation_manifest = dataset / "curation_manifest.json"
    for path in (train, validation, curation_manifest, TRAINER):
        if not path.is_file():
            raise SystemExit(f"missing path: {path}")
    curation = json.loads(curation_manifest.read_text(encoding="utf-8"))
    if not args.skip_strength_gate and not curation.get("quality_checks", {}).get("strength_ready"):
        raise SystemExit(f"dataset is not strength-ready: {curation_manifest}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for hidden in (128, 256):
        model = out_dir / f"halfkp_h{hidden}.nnue"
        metrics = out_dir / f"halfkp_h{hidden}.metrics.json"
        command = [
            sys.executable,
            str(TRAINER),
            "--data", str(train),
            "--validation-data", str(validation),
            "--out", str(model),
            "--metrics-out", str(metrics),
            "--hidden-size", str(hidden),
            "--epochs", str(max(1, args.epochs)),
            "--batch-size", str(max(1, args.batch_size)),
            "--lr", str(args.lr),
            "--ridge", str(max(0.0, args.ridge)),
            "--result-weight", str(max(0.0, min(1.0, args.result_weight))),
            "--target-mode", "centipawn",
            "--target-clip", str(max(1.0, args.target_clip)),
            "--wdl-cp", str(max(1.0, args.wdl_cp)),
            "--tactical-weight", "1.25",
            "--score-balance-power", "0.5",
            "--patience", "12",
            "--min-epochs", "20",
            "--min-delta", "0.03",
            "--seed", str(args.seed),
        ]
        if not args.skip_strength_gate:
            command.append("--strength-ready")
        print(f"training H{hidden}: {model}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        models.append(
            {
                "hidden_size": hidden,
                "model": str(model),
                "model_sha256": sha256_file(model),
                "metrics": str(metrics),
                "metrics_sha256": sha256_file(metrics),
            }
        )

    manifest = {
        "dataset_manifest": str(curation_manifest),
        "dataset_manifest_sha256": sha256_file(curation_manifest),
        "training_configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ridge": args.ridge,
            "result_weight": args.result_weight,
            "target_clip": args.target_clip,
            "wdl_cp": args.wdl_cp,
            "seed": args.seed,
        },
        "models": models,
    }
    manifest_path = out_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"training manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

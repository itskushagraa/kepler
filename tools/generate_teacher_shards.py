#!/usr/bin/env python3
"""Generate a resumable, parallel Stockfish-teacher dataset and curate it."""

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "tools" / "nnue_cycle.py"
CURATOR = ROOT / "tools" / "curate_dataset.py"


def completed_shard(directory: Path, expected_games: int) -> Path | None:
    for summary_path in sorted(directory.glob("summary_*.json"), reverse=True):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            data = Path(summary["data_path"])
            if summary.get("games") == expected_games and data.is_file() and data.stat().st_size > 0:
                return data.resolve()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a strength-scale teacher dataset in resumable shards.")
    parser.add_argument("--engine", default="build-release/kepler")
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=24_000)
    parser.add_argument("--games-per-shard", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--label-nodes", type=int, default=20_000)
    parser.add_argument("--play-movetime", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=220)
    parser.add_argument("--random-plies", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-abs-score", type=int, default=2000)
    parser.add_argument("--kepler-game-fraction", type=float, default=0.25)
    parser.add_argument("--teacher-threads", type=int, default=1)
    parser.add_argument("--teacher-hash", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23000)
    parser.add_argument("--curated-rows", type=int, default=1_000_000)
    parser.add_argument("--skip-curation", action="store_true")
    parser.add_argument("--skip-strength-gate", action="store_true")
    args = parser.parse_args()

    engine = Path(args.engine).resolve()
    stockfish_bin = shutil.which(args.stockfish)
    stockfish = Path(stockfish_bin).resolve() if stockfish_bin else Path(args.stockfish).resolve()
    for path in (engine, stockfish, CYCLE, CURATOR):
        if not path.is_file():
            raise SystemExit(f"missing path: {path}")
    if args.games < 1 or args.games_per_shard < 1:
        parser.error("--games and --games-per-shard must be positive")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_count = math.ceil(args.games / args.games_per_shard)
    detected = os.cpu_count() or 4
    workers = args.workers if args.workers > 0 else max(1, min(4, detected // 3))
    shard_specs = []
    remaining = args.games
    for index in range(shard_count):
        games = min(args.games_per_shard, remaining)
        remaining -= games
        shard_specs.append((index + 1, games, out_dir / f"shard-{index + 1:03d}"))

    data_paths: dict[int, Path] = {}
    pending = []
    for index, games, directory in shard_specs:
        directory.mkdir(parents=True, exist_ok=True)
        existing = completed_shard(directory, games)
        if existing:
            data_paths[index] = existing
            print(f"shard {index}/{shard_count} already complete: {existing}")
        else:
            pending.append((index, games, directory))

    def run_shard(spec: tuple[int, int, Path]) -> tuple[int, Path]:
        index, games, directory = spec
        command = [
            sys.executable,
            str(CYCLE),
            "--engine", str(engine),
            "--stockfish", str(stockfish),
            "--datagen-mode", "teacher",
            "--generate-only",
            "--workdir", str(directory),
            "--games", str(games),
            "--movetime", str(max(1, args.play_movetime)),
            "--maxply", str(max(20, args.max_plies)),
            "--randomplies", str(max(0, args.random_plies)),
            "--sampleevery", str(max(1, args.sample_every)),
            "--tactical-sampleevery", str(max(1, args.sample_every)),
            "--minsampleply", "8",
            "--filter-min-ply", "8",
            "--filter-max-ply", str(max(20, args.max_plies)),
            "--filter-max-abs-score-cp", str(args.max_abs_score),
            "--teacher-play-mode", "full",
            "--teacher-label-elo", "0",
            "--teacher-analyze-nodes", str(max(1, args.label_nodes)),
            "--teacher-kepler-game-fraction", str(max(0.0, min(1.0, args.kepler_game_fraction))),
            "--teacher-threads", str(max(1, args.teacher_threads)),
            "--teacher-hash", str(max(1, args.teacher_hash)),
            "--kepler-threads", "1",
            "--kepler-hash", "64",
            "--seed", str(args.seed + index),
        ]
        log_path = directory / "generation.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"shard {index} failed with code {completed.returncode}; see {log_path}")
        data = completed_shard(directory, games)
        if not data:
            raise RuntimeError(f"shard {index} finished without a valid summary; see {log_path}")
        return index, data

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(run_shard, spec): spec[0] for spec in pending}
            try:
                for future in concurrent.futures.as_completed(futures):
                    index, data = future.result()
                    data_paths[index] = data
                    print(f"shard {index}/{shard_count} complete: {data}", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    ordered_data = [data_paths[index] for index in range(1, shard_count + 1)]
    generation_manifest = {
        "games": args.games,
        "games_per_shard": args.games_per_shard,
        "shards": [str(path) for path in ordered_data],
        "workers": workers,
        "label_nodes": args.label_nodes,
        "engine": str(engine),
        "stockfish": str(stockfish),
    }
    (out_dir / "generation_manifest.json").write_text(
        json.dumps(generation_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.skip_curation:
        return 0

    curated_dir = out_dir / "curated"
    command = [
        sys.executable,
        str(CURATOR),
        "--out-dir", str(curated_dir),
        "--max-rows", str(max(1, args.curated_rows)),
        "--tactical-fraction", "0.35",
        "--val-fraction", "0.2",
        "--seed", "42",
        "--max-extreme-score-fraction", "0.50",
    ]
    for path in ordered_data:
        command.extend(["--input", str(path)])
    if not args.skip_strength_gate:
        command.append("--strength-ready")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(f"strength dataset ready: {curated_dir / 'curation_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def ply_from_fen(fen: str) -> Optional[int]:
    parts = fen.split()
    if len(parts) < 6:
        return None
    stm = parts[1]
    try:
        fullmove = int(parts[5])
    except ValueError:
        return None
    if fullmove < 1:
        return None
    return (fullmove - 1) * 2 + (0 if stm == "w" else 1)


def score_bucket(score_cp_stm: int) -> str:
    if score_cp_stm <= -300:
        return "le_-300"
    if score_cp_stm <= -151:
        return "-300_to_-151"
    if score_cp_stm <= -51:
        return "-150_to_-51"
    if score_cp_stm <= 50:
        return "-50_to_50"
    if score_cp_stm <= 150:
        return "51_to_150"
    if score_cp_stm <= 300:
        return "151_to_300"
    return "ge_301"


def phase_from_ply(ply: Optional[int], cut1: int, cut2: int) -> str:
    if ply is None:
        return "unknown"
    if ply <= cut1:
        return "opening"
    if ply <= cut2:
        return "middlegame"
    return "endgame"


def hash_fen(fen: str) -> bytes:
    return hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest()


def parse_line(line: str) -> Optional[Tuple[int, int, str, Optional[int], str, str, str, List[str]]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    try:
        result_stm = int(parts[0])
        score_cp = int(parts[1])
    except ValueError:
        return None

    fen = parts[2]
    ply = None
    if len(parts) > 3 and parts[3]:
        try:
            ply = int(parts[3])
        except ValueError:
            ply = None
    if ply is None:
        ply = ply_from_fen(fen)

    bucket = parts[4] if len(parts) > 4 and parts[4] else score_bucket(score_cp)
    opening_id = parts[5] if len(parts) > 5 and parts[5] else "unknown"
    termination = parts[6] if len(parts) > 6 and parts[6] else "unknown"
    return result_stm, score_cp, fen, ply, bucket, opening_id, termination, parts


def bump(hist: Dict[str, int], key: str, inc: int = 1) -> None:
    hist[key] = hist.get(key, 0) + inc


def iter_lines(path: Path) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate dataset: dedup, stratify, split by game ID.")
    ap.add_argument("--input", required=True, help="Input TSV dataset.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--val-fraction", type=float, default=0.05, help="Validation fraction by game.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    ap.add_argument("--phase-cut1", type=int, default=20, help="Opening/middlegame cutoff (ply).")
    ap.add_argument("--phase-cut2", type=int, default=60, help="Middlegame/endgame cutoff (ply).")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: dedup + game boundaries + per-game stats.
    seen = set()
    total_rows = 0
    unique_rows = 0
    duplicate_rows = 0
    bad_rows = 0
    game_id = 0
    prev_ply: Optional[int] = None

    game_total_rows: Dict[int, int] = defaultdict(int)
    game_phase_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    game_bucket_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for line in iter_lines(input_path):
        total_rows += 1
        parsed = parse_line(line)
        if parsed is None:
            bad_rows += 1
            continue
        _, score_cp, fen, ply, bucket, _, _, _ = parsed

        if prev_ply is not None and ply is not None and ply <= prev_ply:
            game_id += 1
        prev_ply = ply

        h = hash_fen(fen)
        if h in seen:
            duplicate_rows += 1
            continue
        seen.add(h)
        unique_rows += 1

        phase = phase_from_ply(ply, args.phase_cut1, args.phase_cut2)
        game_total_rows[game_id] += 1
        bump(game_phase_counts[game_id], phase)
        bump(game_bucket_counts[game_id], bucket)

    total_games = game_id + 1 if total_rows else 0

    # Build stratification buckets per game.
    strata: Dict[str, List[int]] = defaultdict(list)
    for gid in range(total_games):
        phase_counts = game_phase_counts.get(gid, {})
        bucket_counts = game_bucket_counts.get(gid, {})
        phase = max(phase_counts.items(), key=lambda kv: kv[1])[0] if phase_counts else "unknown"
        bucket = max(bucket_counts.items(), key=lambda kv: kv[1])[0] if bucket_counts else "unknown"
        strata[f"{phase}|{bucket}"].append(gid)

    rng = random.Random(args.seed)
    val_games = set()
    for key, gids in strata.items():
        rng.shuffle(gids)
        if len(gids) <= 1:
            continue
        val_n = max(1, int(round(len(gids) * args.val_fraction)))
        val_games.update(gids[:val_n])

    train_games = set(range(total_games)) - val_games

    # Pass 2: re-run dedup and write splits with counts.
    seen = set()
    game_id = 0
    prev_ply = None

    train_path = out_dir / "train.tsv"
    val_path = out_dir / "val.tsv"
    train_games_path = out_dir / "train_games.txt"
    val_games_path = out_dir / "val_games.txt"

    train_rows = 0
    val_rows = 0
    train_phase_counts: Dict[str, int] = defaultdict(int)
    val_phase_counts: Dict[str, int] = defaultdict(int)
    train_bucket_counts: Dict[str, int] = defaultdict(int)
    val_bucket_counts: Dict[str, int] = defaultdict(int)

    with open(train_path, "w", encoding="utf-8") as train, open(val_path, "w", encoding="utf-8") as val:
        for line in iter_lines(input_path):
            parsed = parse_line(line)
            if parsed is None:
                continue
            _, score_cp, fen, ply, bucket, _, _, _ = parsed

            if prev_ply is not None and ply is not None and ply <= prev_ply:
                game_id += 1
            prev_ply = ply

            h = hash_fen(fen)
            if h in seen:
                continue
            seen.add(h)

            phase = phase_from_ply(ply, args.phase_cut1, args.phase_cut2)
            target = val if game_id in val_games else train
            target.write(line if line.endswith("\n") else line + "\n")
            if game_id in val_games:
                val_rows += 1
                bump(val_phase_counts, phase)
                bump(val_bucket_counts, bucket)
            else:
                train_rows += 1
                bump(train_phase_counts, phase)
                bump(train_bucket_counts, bucket)

    train_games_path.write_text("\n".join(str(g) for g in sorted(train_games)) + "\n", encoding="utf-8")
    val_games_path.write_text("\n".join(str(g) for g in sorted(val_games)) + "\n", encoding="utf-8")

    duplicate_rate = (duplicate_rows / max(1, total_rows)) if total_rows else 0.0

    manifest = {
        "task_id": "SF-04",
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "settings": {
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "phase_cut1": args.phase_cut1,
            "phase_cut2": args.phase_cut2,
        },
        "rows": {
            "total": total_rows,
            "unique": unique_rows,
            "duplicates": duplicate_rows,
            "bad_rows": bad_rows,
            "duplicate_rate": duplicate_rate,
        },
        "games": {
            "total": total_games,
            "train": len(train_games),
            "val": len(val_games),
        },
        "splits": {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_phase_counts": dict(train_phase_counts),
            "val_phase_counts": dict(val_phase_counts),
            "train_bucket_counts": dict(train_bucket_counts),
            "val_bucket_counts": dict(val_bucket_counts),
        },
        "artifacts": {
            "train_tsv": str(train_path),
            "val_tsv": str(val_path),
            "train_games": str(train_games_path),
            "val_games": str(val_games_path),
        },
    }

    manifest_path = out_dir / "sf04_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_manifest = {
        "split": "train",
        "rows": train_rows,
        "games": len(train_games),
        "phase_counts": dict(train_phase_counts),
        "bucket_counts": dict(train_bucket_counts),
        "data_path": str(train_path),
    }
    val_manifest = {
        "split": "val",
        "rows": val_rows,
        "games": len(val_games),
        "phase_counts": dict(val_phase_counts),
        "bucket_counts": dict(val_bucket_counts),
        "data_path": str(val_path),
    }

    (out_dir / "sf04_train_manifest.json").write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    (out_dir / "sf04_val_manifest.json").write_text(json.dumps(val_manifest, indent=2), encoding="utf-8")

    print("wrote", manifest_path)


if __name__ == "__main__":
    main()

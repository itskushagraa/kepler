#!/usr/bin/env python3
"""Deduplicate, balance, and game-split Kepler NNUE teacher data.

Current rows use this tab-separated schema:

    result, score_cp, fen, ply, score_bucket, opening, termination,
    game_id, sample_kind

Multiple input files are supported. Game IDs are namespaced by source before
writing so independently generated shards cannot leak the same apparent game
across training and validation. Legacy rows without game IDs are grouped only
within their source file by ply resets.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCORE_BUCKETS = (
    "le_-300",
    "-300_to_-151",
    "-150_to_-51",
    "-50_to_50",
    "51_to_150",
    "151_to_300",
    "ge_301",
)


@dataclass
class Row:
    ordinal: int
    result_stm: int
    score_cp: int
    fen: str
    ply: Optional[int]
    bucket: str
    opening_id: str
    termination: str
    game_id: str
    kind: str
    phase: str
    fields: List[str]
    material: str = "unknown"

    @property
    def balance_kind(self) -> str:
        return "tactical" if self.kind == "tactical" else "regular"

    @property
    def stratum(self) -> str:
        return f"{self.balance_kind}|{self.phase}|{self.material}|{self.bucket}"

    def serialize(self) -> str:
        fields = list(self.fields)
        while len(fields) < 11:
            fields.append("")
        fields[:9] = [
            str(self.result_stm),
            str(self.score_cp),
            self.fen,
            "" if self.ply is None else str(self.ply),
            self.bucket,
            self.opening_id,
            self.termination,
            self.game_id,
            self.kind,
        ]
        fields[9] = self.phase
        fields[10] = self.material
        return "\t".join(fields) + "\n"


def ply_from_fen(fen: str) -> Optional[int]:
    parts = fen.split()
    if len(parts) < 6:
        return None
    try:
        fullmove = int(parts[5])
    except ValueError:
        return None
    if fullmove < 1 or parts[1] not in {"w", "b"}:
        return None
    return (fullmove - 1) * 2 + (0 if parts[1] == "w" else 1)


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


def material_class_from_fen(fen: str) -> str:
    board = fen.split()[0] if fen.split() else ""
    queens = board.count("q") + board.count("Q")
    rooks = board.count("r") + board.count("R")
    minors = sum(board.count(piece) for piece in "nNbB")
    pawns = board.count("p") + board.count("P")
    if queens:
        return "queens"
    if rooks:
        return "rooks_no_queens"
    if minors:
        return "minor_only"
    if pawns:
        return "pawn_only"
    return "bare_kings"


def phase_from_position(fen: str, ply: Optional[int], cut1: int, cut2: int) -> str:
    board = fen.split()[0] if fen.split() else ""
    values = {"n": 320, "b": 330, "r": 500, "q": 900}
    nonpawn_material = sum(
        value * (board.count(piece) + board.count(piece.upper()))
        for piece, value in values.items()
    )
    if ply is not None and ply <= cut1 and nonpawn_material >= 5200:
        return "opening"
    if nonpawn_material <= 2600 or (ply is not None and ply > cut2 and nonpawn_material <= 3600):
        return "endgame"
    return "middlegame"


def position_key(fen: str) -> str:
    """Ignore move clocks while retaining side, castling, and en-passant state."""
    parts = fen.split()
    return " ".join(parts[:4]) if len(parts) >= 4 else fen.strip()


def iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield line


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(
    paths: Sequence[Path], phase_cut1: int, phase_cut2: int
) -> Tuple[List[Row], Dict[str, int]]:
    rows: List[Row] = []
    seen_positions = set()
    stats: Counter[str] = Counter()
    ordinal = 0

    for source_index, path in enumerate(paths):
        legacy_group = 0
        previous_legacy_ply: Optional[int] = None
        source_name = f"source-{source_index + 1}:{path.name}"
        for line in iter_lines(path):
            stats["total"] += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                stats["bad_rows"] += 1
                continue
            try:
                result_stm = int(fields[0])
                score_cp = int(fields[1])
            except ValueError:
                stats["bad_rows"] += 1
                continue
            fen = fields[2].strip()
            if len(fen.split()) < 4:
                stats["bad_rows"] += 1
                continue
            try:
                ply = int(fields[3]) if len(fields) > 3 and fields[3] else None
            except ValueError:
                ply = None
            if ply is None:
                ply = ply_from_fen(fen)

            explicit_game = fields[7].strip() if len(fields) > 7 else ""
            if explicit_game:
                local_game = explicit_game
            else:
                if (
                    previous_legacy_ply is not None
                    and ply is not None
                    and ply <= previous_legacy_ply
                ):
                    legacy_group += 1
                local_game = f"legacy-{legacy_group}"
                previous_legacy_ply = ply

            key = position_key(fen)
            if key in seen_positions:
                stats["duplicates"] += 1
                continue
            seen_positions.add(key)

            bucket = score_bucket(score_cp)
            opening_id = fields[5].strip() if len(fields) > 5 and fields[5].strip() else "unknown"
            termination = fields[6].strip() if len(fields) > 6 and fields[6].strip() else "unknown"
            kind = fields[8].strip().lower() if len(fields) > 8 and fields[8].strip() else "regular"
            game_id = f"{source_name}::{local_game}"
            rows.append(
                Row(
                    ordinal=ordinal,
                    result_stm=result_stm,
                    score_cp=score_cp,
                    fen=fen,
                    ply=ply,
                    bucket=bucket,
                    opening_id=opening_id,
                    termination=termination,
                    game_id=game_id,
                    kind=kind,
                    phase=phase_from_position(fen, ply, phase_cut1, phase_cut2),
                    fields=fields,
                    material=material_class_from_fen(fen),
                )
            )
            ordinal += 1

    return rows, {
        "total": stats["total"],
        "unique": len(rows),
        "duplicates": stats["duplicates"],
        "bad_rows": stats["bad_rows"],
    }


def even_sample(rows: Sequence[Row], target: int, seed: int, key_name: str) -> List[Row]:
    """Water-fill strata so abundant cells cannot drown out scarce cells."""
    if target <= 0 or not rows:
        return []
    if target >= len(rows):
        return list(rows)

    groups: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        if key_name == "phase_bucket":
            key = f"{row.phase}|{row.material}|{row.bucket}"
        elif key_name == "full":
            key = row.stratum
        else:
            raise ValueError(f"unknown balance key: {key_name}")
        groups[key].append(row)

    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])

    selected: List[Row] = []
    level = 0
    while len(selected) < target:
        added = False
        for key in keys:
            if level < len(groups[key]):
                selected.append(groups[key][level])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        level += 1
    return selected


def balance_rows(
    rows: Sequence[Row],
    max_rows: int,
    tactical_fraction: float,
    seed: int,
    max_extreme_fraction: float = 1.0,
) -> List[Row]:
    target = len(rows) if max_rows <= 0 else min(len(rows), max_rows)
    if target >= len(rows):
        return sorted(rows, key=lambda row: row.ordinal)

    tactical = [row for row in rows if row.balance_kind == "tactical"]
    regular = [row for row in rows if row.balance_kind == "regular"]
    desired_tactical = int(round(target * max(0.0, min(1.0, tactical_fraction))))
    desired_regular = target - desired_tactical

    selected = even_sample(tactical, min(desired_tactical, len(tactical)), seed + 11, "phase_bucket")
    selected += even_sample(regular, min(desired_regular, len(regular)), seed + 23, "phase_bucket")
    selected_ordinals = {row.ordinal for row in selected}
    if len(selected) < target:
        remainder = [row for row in rows if row.ordinal not in selected_ordinals]
        selected += even_sample(remainder, target - len(selected), seed + 37, "full")

    extreme_limit = int(target * max(0.0, min(1.0, max_extreme_fraction)))
    selected_extreme = [row for row in selected if abs(row.score_cp) >= 301]
    if len(selected_extreme) > extreme_limit:
        selected_ordinals = {row.ordinal for row in selected}
        replacements = [
            row
            for row in rows
            if row.ordinal not in selected_ordinals and abs(row.score_cp) < 301
        ]
        replace_count = min(len(selected_extreme) - extreme_limit, len(replacements))
        remove_ordinals = {
            row.ordinal
            for row in even_sample(selected_extreme, replace_count, seed + 41, "full")
        }
        selected = [row for row in selected if row.ordinal not in remove_ordinals]
        selected += even_sample(replacements, replace_count, seed + 43, "full")
    return sorted(selected, key=lambda row: row.ordinal)


def dominant_stratum(rows: Sequence[Row]) -> str:
    counts = Counter(row.stratum for row in rows)
    return min((-count, key) for key, count in counts.items())[1]


def split_games(rows: Sequence[Row], val_fraction: float, seed: int) -> Tuple[set, set]:
    by_game: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        by_game[row.game_id].append(row)
    games = sorted(by_game)
    if len(games) < 2 or val_fraction <= 0.0:
        return set(games), set()

    target = int(round(len(games) * min(0.8, val_fraction)))
    target = min(max(1, target), len(games) - 1)
    strata: Dict[str, List[str]] = defaultdict(list)
    for game in games:
        strata[dominant_stratum(by_game[game])].append(game)

    rng = random.Random(seed)
    keys = sorted(strata)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(strata[key])

    val_games = set()
    level = 0
    while len(val_games) < target:
        added = False
        for key in keys:
            if level < len(strata[key]):
                val_games.add(strata[key][level])
                added = True
                if len(val_games) == target:
                    break
        if not added:
            break
        level += 1
    return set(games) - val_games, val_games


def histogram(rows: Sequence[Row], attribute: str) -> Dict[str, int]:
    return dict(sorted(Counter(getattr(row, attribute) for row in rows).items()))


def row_summary(rows: Sequence[Row]) -> Dict[str, object]:
    tactical_rows = sum(row.balance_kind == "tactical" for row in rows)
    extreme_rows = sum(abs(row.score_cp) >= 301 for row in rows)
    game_counts = Counter(row.game_id for row in rows)
    return {
        "rows": len(rows),
        "games": len({row.game_id for row in rows}),
        "tactical_fraction": tactical_rows / len(rows) if rows else None,
        "phase_counts": histogram(rows, "phase"),
        "material_counts": histogram(rows, "material"),
        "score_bucket_counts": histogram(rows, "bucket"),
        "result_counts": dict(sorted(Counter(str(row.result_stm) for row in rows).items())),
        "opening_counts": dict(sorted(Counter(row.opening_id for row in rows).items())),
        "termination_counts": dict(sorted(Counter(row.termination for row in rows).items())),
        "extreme_score_fraction": extreme_rows / len(rows) if rows else None,
        "largest_game_fraction": max(game_counts.values(), default=0) / len(rows) if rows else None,
        "sample_kind_counts": dict(sorted(Counter(row.balance_kind for row in rows).items())),
        "joint_stratum_counts": dict(sorted(Counter(row.stratum for row in rows).items())),
    }


def write_rows(path: Path, rows: Sequence[Row]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(row.serialize())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate, balance, and split NNUE TSV shards by complete games."
    )
    parser.add_argument("--input", action="append", required=True, help="Input TSV; repeat for shards.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="Balanced row cap; 0 keeps all unique rows.")
    parser.add_argument("--tactical-fraction", type=float, default=0.4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-cut1", type=int, default=20)
    parser.add_argument("--phase-cut2", type=int, default=60)
    parser.add_argument("--min-rows", type=int, default=0, help="Fail the quality gate below this row count.")
    parser.add_argument("--min-games", type=int, default=0, help="Fail the quality gate below this game count.")
    parser.add_argument("--min-opening-ids", type=int, default=0)
    parser.add_argument("--max-extreme-score-fraction", type=float, default=1.0)
    parser.add_argument("--max-game-row-fraction", type=float, default=1.0)
    parser.add_argument(
        "--strength-ready",
        action="store_true",
        help="Require at least 1M positions from 20k games and basic diversity checks.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.tactical_fraction <= 1.0:
        parser.error("--tactical-fraction must be between 0 and 1")
    if not 0.0 <= args.val_fraction <= 0.8:
        parser.error("--val-fraction must be between 0 and 0.8")
    if args.phase_cut2 <= args.phase_cut1:
        parser.error("--phase-cut2 must be greater than --phase-cut1")
    if not 0.0 <= args.max_extreme_score_fraction <= 1.0:
        parser.error("--max-extreme-score-fraction must be between 0 and 1")
    if not 0.0 <= args.max_game_row_fraction <= 1.0:
        parser.error("--max-game-row-fraction must be between 0 and 1")
    if args.strength_ready:
        args.min_rows = max(args.min_rows, 1_000_000)
        args.min_games = max(args.min_games, 20_000)
        args.min_opening_ids = max(args.min_opening_ids, 32)
        args.max_extreme_score_fraction = min(args.max_extreme_score_fraction, 0.50)
        args.max_game_row_fraction = min(args.max_game_row_fraction, 0.002)

    inputs = [Path(value).resolve() for value in args.input]
    for path in inputs:
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_rows, source_stats = load_rows(inputs, args.phase_cut1, args.phase_cut2)
    if not unique_rows:
        raise SystemExit("No valid unique rows found.")
    curated = balance_rows(
        unique_rows,
        max(0, args.max_rows),
        args.tactical_fraction,
        args.seed,
        args.max_extreme_score_fraction,
    )
    train_games, val_games = split_games(curated, args.val_fraction, args.seed)
    train = [row for row in curated if row.game_id in train_games]
    validation = [row for row in curated if row.game_id in val_games]

    curated_path = out_dir / "curated.tsv"
    train_path = out_dir / "train.tsv"
    validation_path = out_dir / "validation.tsv"
    write_rows(curated_path, curated)
    write_rows(train_path, train)
    write_rows(validation_path, validation)
    (out_dir / "train_games.txt").write_text("\n".join(sorted(train_games)) + "\n", encoding="utf-8")
    (out_dir / "validation_games.txt").write_text(
        "\n".join(sorted(val_games)) + ("\n" if val_games else ""), encoding="utf-8"
    )

    curated_summary = row_summary(curated)
    diversity_checks = {
        "minimum_rows_met": len(curated) >= args.min_rows,
        "minimum_games_met": len({row.game_id for row in curated}) >= args.min_games,
        "minimum_openings_met": len({row.opening_id for row in curated}) >= args.min_opening_ids,
        "extreme_score_fraction_met": (
            curated_summary["extreme_score_fraction"] is not None
            and curated_summary["extreme_score_fraction"] <= args.max_extreme_score_fraction
        ),
        "game_concentration_met": (
            curated_summary["largest_game_fraction"] is not None
            and curated_summary["largest_game_fraction"] <= args.max_game_row_fraction
        ),
    }

    manifest = {
        "schema_version": 3,
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in inputs
        ],
        "settings": {
            "max_rows": args.max_rows,
            "tactical_fraction": args.tactical_fraction,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "phase_cut1": args.phase_cut1,
            "phase_cut2": args.phase_cut2,
            "min_rows": args.min_rows,
            "min_games": args.min_games,
            "min_opening_ids": args.min_opening_ids,
            "max_extreme_score_fraction": args.max_extreme_score_fraction,
            "max_game_row_fraction": args.max_game_row_fraction,
            "strength_ready": args.strength_ready,
        },
        "source_rows": source_stats,
        "before_balancing": row_summary(unique_rows),
        "curated": curated_summary,
        "train": row_summary(train),
        "validation": row_summary(validation),
        "game_leakage": sorted(train_games & val_games),
        "quality_checks": {
            "row_target_met": len(curated)
            == (len(unique_rows) if args.max_rows <= 0 else min(len(unique_rows), args.max_rows)),
            "game_leakage_free": not bool(train_games & val_games),
            "both_splits_nonempty": bool(train)
            and (args.val_fraction == 0.0 or bool(validation)),
            **diversity_checks,
            "strength_ready": args.strength_ready and all(diversity_checks.values()),
        },
        "artifacts": {
            "curated_tsv": str(curated_path),
            "train_tsv": str(train_path),
            "validation_tsv": str(validation_path),
            "train_games": str(out_dir / "train_games.txt"),
            "validation_games": str(out_dir / "validation_games.txt"),
        },
        "artifact_sha256": {
            "curated_tsv": sha256_file(curated_path),
            "train_tsv": sha256_file(train_path),
            "validation_tsv": sha256_file(validation_path),
        },
    }
    manifest_path = out_dir / "curation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "curated": manifest["curated"]}, indent=2))
    if not all(diversity_checks.values()):
        failed = ", ".join(name for name, passed in diversity_checks.items() if not passed)
        raise SystemExit(f"dataset quality gate failed: {failed}")


if __name__ == "__main__":
    main()

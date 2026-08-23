#!/usr/bin/env python3
"""Build a protocol-matched reference ladder for Kepler.

This tool runs a full-strength round robin between supplied UCI engines, fits
all engine ratings with one externally anchored rating, and writes the
calibrated manifest consumed by rated_gauntlet.py. It never uses
UCI_LimitStrength or UCI_Elo.
"""

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine
import chess.pgn

from rated_gauntlet import (
    BenchmarkError,
    apply_opening,
    build_limit,
    engine_metadata,
    resolve_executable,
    select_opening,
)


@dataclass(frozen=True)
class EngineSpec:
    name: str
    path: Path
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationGame:
    white: str
    black: str
    white_score: float
    opening_index: int
    termination: str
    plies: int


def parse_engine_config(path: Path) -> List[EngineSpec]:
    with open(path, "r", encoding="utf-8") as source:
        raw = json.load(source)
    entries = raw.get("engines", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or len(entries) < 2:
        raise BenchmarkError("engine config must contain at least two engines")

    specs: List[EngineSpec] = []
    names = set()
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry or "path" not in entry:
            raise BenchmarkError("each engine must contain name and path")
        name = str(entry["name"])
        if name in names:
            raise BenchmarkError(f"duplicate engine name: {name}")
        names.add(name)
        options = entry.get("options", {})
        if not isinstance(options, dict):
            raise BenchmarkError(f"engine options must be an object: {name}")
        specs.append(
            EngineSpec(
                name=name,
                path=resolve_executable(str(entry["path"])),
                options=dict(options),
            )
        )
    return specs


def configure_reference(
    engine: chess.engine.SimpleEngine,
    spec: EngineSpec,
    hash_mb: int,
    threads: int,
) -> Dict[str, Any]:
    supported = engine.options
    options: Dict[str, Any] = {}
    if "Hash" in supported:
        options["Hash"] = max(1, hash_mb)
    if "Threads" in supported:
        options["Threads"] = max(1, threads)
    for name, value in spec.options.items():
        if name not in supported:
            raise BenchmarkError(f"{spec.name} does not support configured option {name}")
        options[name] = value
    if "UCI_LimitStrength" in supported:
        options["UCI_LimitStrength"] = False
    options.pop("UCI_Elo", None)
    engine.configure(options)
    return options


def result_for_white(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def build_calibration_pgn(
    board: chess.Board,
    white: str,
    black: str,
    result: float,
    opening_index: int,
    termination: str,
    round_name: str,
) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = "Kepler Reference Calibration"
    game.headers["Date"] = dt.datetime.now(dt.timezone.utc).strftime("%Y.%m.%d")
    game.headers["Round"] = round_name
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = "1-0" if result == 1.0 else "0-1" if result == 0.0 else "1/2-1/2"
    game.headers["Termination"] = termination
    game.headers["OpeningIndex"] = str(opening_index)
    game.headers["CalibrationMode"] = "full_strength"
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    return game


def play_pair(
    first: EngineSpec,
    second: EngineSpec,
    pair_index: int,
    args: argparse.Namespace,
    limit: chess.engine.Limit,
) -> Tuple[List[CalibrationGame], List[chess.pgn.Game], Dict[str, Any]]:
    games: List[CalibrationGame] = []
    pgns: List[chess.pgn.Game] = []
    metadata: Dict[str, Any] = {}
    with chess.engine.SimpleEngine.popen_uci(str(first.path)) as first_engine, chess.engine.SimpleEngine.popen_uci(str(second.path)) as second_engine:
        first_options = configure_reference(first_engine, first, args.hash, args.threads)
        second_options = configure_reference(second_engine, second, args.hash, args.threads)
        metadata[first.name] = engine_metadata(first.path, first_engine)
        metadata[first.name]["benchmark_options"] = first_options
        metadata[second.name] = engine_metadata(second.path, second_engine)
        metadata[second.name]["benchmark_options"] = second_options

        for game_index in range(args.games_per_pair):
            opening_index, opening = select_opening(
                game_index,
                args.seed + pair_index * 31,
            )
            board = chess.Board()
            apply_opening(board, opening)
            first_white = game_index % 2 == 0
            termination = "natural"

            while not board.is_game_over(claim_draw=True):
                if args.max_plies > 0 and board.ply() >= args.max_plies:
                    termination = "adjudicated_max_plies"
                    break
                mover = first_engine if ((board.turn == chess.WHITE) == first_white) else second_engine
                try:
                    played = mover.play(board, limit)
                except Exception as exc:
                    raise BenchmarkError(
                        f"{first.name} vs {second.name} game {game_index + 1}: engine failure: {exc}"
                    ) from exc
                if played.move is None or played.move not in board.legal_moves:
                    raise BenchmarkError(
                        f"{first.name} vs {second.name} game {game_index + 1}: illegal or missing move"
                    )
                board.push(played.move)

            white_score = result_for_white(board)
            if termination == "natural":
                outcome = board.outcome(claim_draw=True)
                termination = outcome.termination.name if outcome and outcome.termination else "draw"
            white_name = first.name if first_white else second.name
            black_name = second.name if first_white else first.name
            first_score = white_score if first_white else 1.0 - white_score
            games.append(
                CalibrationGame(
                    white=white_name,
                    black=black_name,
                    white_score=white_score,
                    opening_index=opening_index,
                    termination=termination,
                    plies=board.ply(),
                )
            )
            pgns.append(
                build_calibration_pgn(
                    board,
                    white_name,
                    black_name,
                    white_score,
                    opening_index,
                    termination,
                    f"{pair_index + 1}.{game_index + 1}",
                )
            )
            result_text = "win" if first_score == 1.0 else "loss" if first_score == 0.0 else "draw"
            print(
                f"  {first.name} vs {second.name} game {game_index + 1}/{args.games_per_pair} "
                f"{result_text} {termination} plies={board.ply()}",
                flush=True,
            )
    return games, pgns, metadata


def logistic_probability(white_rating: float, black_rating: float) -> float:
    exponent = (black_rating - white_rating) / 400.0
    if exponent > 50.0:
        return 0.0
    if exponent < -50.0:
        return 1.0
    return 1.0 / (1.0 + 10.0 ** exponent)


def solve_linear(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Optional[List[float]]:
    size = len(vector)
    if size == 0:
        return []
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def invert_matrix(matrix: Sequence[Sequence[float]]) -> Optional[List[List[float]]]:
    size = len(matrix)
    inverse: List[List[float]] = []
    for column in range(size):
        basis = [0.0] * size
        basis[column] = 1.0
        solved = solve_linear(matrix, basis)
        if solved is None:
            return None
        inverse.append(solved)
    return [list(row) for row in zip(*inverse)]


def fit_ratings(
    specs: Sequence[EngineSpec],
    games: Sequence[CalibrationGame],
    anchor_name: str,
    anchor_rating: float,
) -> Dict[str, Any]:
    names = [spec.name for spec in specs]
    if anchor_name not in names:
        raise BenchmarkError(f"anchor engine is not in config: {anchor_name}")
    free_names = [name for name in names if name != anchor_name]
    free_index = {name: index for index, name in enumerate(free_names)}
    ratings = {name: float(anchor_rating) for name in names}
    minimum = float(anchor_rating) - 2000.0
    maximum = float(anchor_rating) + 2000.0
    scale = math.log(10.0) / 400.0
    status = "converged"

    for _ in range(100):
        gradient = [0.0] * len(free_names)
        information = [[0.0] * len(free_names) for _ in free_names]
        for game in games:
            white_rating = ratings[game.white]
            black_rating = ratings[game.black]
            probability = logistic_probability(white_rating, black_rating)
            weight = scale * scale * probability * (1.0 - probability)
            gradient_delta = scale * (game.white_score - probability)
            white_free = free_index.get(game.white)
            black_free = free_index.get(game.black)
            if white_free is not None:
                gradient[white_free] += gradient_delta
                information[white_free][white_free] += weight
            if black_free is not None:
                gradient[black_free] -= gradient_delta
                information[black_free][black_free] += weight
            if white_free is not None and black_free is not None:
                information[white_free][black_free] -= weight
                information[black_free][white_free] -= weight

        step = solve_linear(information, gradient)
        if step is None:
            status = "singular"
            break
        largest_step = 0.0
        for name, index in free_index.items():
            proposed = ratings[name] + step[index]
            if proposed <= minimum or proposed >= maximum:
                status = "boundary"
            ratings[name] = max(minimum, min(maximum, proposed))
            largest_step = max(largest_step, abs(step[index]))
        if largest_step < 1e-5:
            break
    else:
        status = "not_converged"

    # Recompute the observed information at the final solution for standard
    # errors. The anchor's external uncertainty is intentionally not included.
    information = [[0.0] * len(free_names) for _ in free_names]
    for game in games:
        probability = logistic_probability(ratings[game.white], ratings[game.black])
        weight = scale * scale * probability * (1.0 - probability)
        white_free = free_index.get(game.white)
        black_free = free_index.get(game.black)
        if white_free is not None:
            information[white_free][white_free] += weight
        if black_free is not None:
            information[black_free][black_free] += weight
        if white_free is not None and black_free is not None:
            information[white_free][black_free] -= weight
            information[black_free][white_free] -= weight
    covariance = invert_matrix(information)

    output: Dict[str, Any] = {}
    for name in names:
        if name == anchor_name:
            output[name] = {
                "rating": anchor_rating,
                "ci95": None,
                "standard_error": None,
                "status": "external_anchor",
            }
            continue
        index = free_index[name]
        standard_error = math.sqrt(covariance[index][index]) if covariance else None
        output[name] = {
            "rating": ratings[name],
            "ci95": (
                [ratings[name] - 1.96 * standard_error, ratings[name] + 1.96 * standard_error]
                if standard_error is not None
                else [None, None]
            ),
            "standard_error": standard_error,
            "status": "boundary" if ratings[name] in (minimum, maximum) else status,
        }
    return {
        "status": status,
        "anchor": {"name": anchor_name, "rating": anchor_rating},
        "ratings": output,
        "games": len(games),
    }


def aggregate_games(games: Sequence[CalibrationGame]) -> Dict[str, Any]:
    pairs: Dict[str, Dict[str, Any]] = {}
    for game in games:
        names = sorted((game.white, game.black))
        key = f"{names[0]} vs {names[1]}"
        if key not in pairs:
            pairs[key] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        row = pairs[key]
        row["games"] += 1
        score_for_first = game.white_score if game.white == names[0] else 1.0 - game.white_score
        if score_for_first == 1.0:
            row["wins"] += 1
        elif score_for_first == 0.0:
            row["losses"] += 1
        else:
            row["draws"] += 1
    for row in pairs.values():
        row["score"] = row["wins"] + 0.5 * row["draws"]
        row["score_rate"] = row["score"] / row["games"] if row["games"] else None
    return pairs


def build_reference_config(
    specs: Sequence[EngineSpec],
    fitted: Dict[str, Any],
    target_name: Optional[str],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "opponents": [
            {
                "name": spec.name,
                "path": str(spec.path),
                "rating": fitted["ratings"][spec.name]["rating"],
                "options": spec.options,
            }
            for spec in specs
            if spec.name != target_name
        ],
        "anchor": fitted["anchor"],
        "calibration_status": fitted["status"],
        "target_name": target_name,
        "authoritative": fitted["status"] == "converged" and (
            target_name is None or fitted["ratings"][target_name]["status"] == "converged"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a full-strength UCI reference ladder under one fixed protocol."
    )
    parser.add_argument("--engines-config", required=True, help="JSON containing at least two full-strength engines.")
    parser.add_argument(
        "--target-name",
        default="",
        help="Optional engine to measure; it is excluded from the generated opponent config.",
    )
    parser.add_argument("--anchor-name", required=True, help="Engine whose external rating anchors the ladder.")
    parser.add_argument("--anchor-rating", type=float, required=True, help="Trusted external rating for the anchor.")
    parser.add_argument("--games-per-pair", type=int, default=40, help="Games per engine pair; use an even number.")
    parser.add_argument("--nodes", type=int, default=100000, help="Fixed node budget.")
    parser.add_argument("--depth", type=int, default=0, help="Fixed depth instead of nodes.")
    parser.add_argument("--movetime-ms", type=int, default=0, help="Fixed time instead of nodes/depth.")
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-pgn", required=True)
    parser.add_argument("--out-reference-config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        specs = parse_engine_config(Path(args.engines_config).resolve())
        target_name = args.target_name or None
        if target_name and target_name not in {spec.name for spec in specs}:
            raise BenchmarkError(f"target engine is not in config: {target_name}")
        if target_name and target_name == args.anchor_name:
            raise BenchmarkError("target engine cannot also be the external anchor")
        if args.games_per_pair < 2 or args.games_per_pair % 2:
            raise BenchmarkError("games-per-pair must be an even number >= 2")
        if args.hash < 1 or args.threads < 1:
            raise BenchmarkError("hash and threads must be positive")
        limit = build_limit(args.nodes, args.depth, args.movetime_ms)
        all_games: List[CalibrationGame] = []
        all_pgns: List[chess.pgn.Game] = []
        metadata: Dict[str, Any] = {}
        pair_index = 0
        for first_index, first in enumerate(specs):
            for second in specs[first_index + 1:]:
                print(
                    f"[ladder] {first.name} vs {second.name} "
                    f"games={args.games_per_pair} limit={limit}",
                    flush=True,
                )
                games, pgns, pair_metadata = play_pair(first, second, pair_index, args, limit)
                all_games.extend(games)
                all_pgns.extend(pgns)
                metadata.update(pair_metadata)
                pair_index += 1

        fitted = fit_ratings(specs, all_games, args.anchor_name, args.anchor_rating)
        target_rating = fitted["ratings"].get(target_name) if target_name else None
        authoritative = fitted["status"] == "converged" and (
            target_rating is None or target_rating["status"] == "converged"
        )
        out_json = Path(args.out_json).resolve()
        out_pgn = Path(args.out_pgn).resolve()
        out_config = Path(args.out_reference_config).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_pgn.parent.mkdir(parents=True, exist_ok=True)
        out_config.parent.mkdir(parents=True, exist_ok=True)
        with open(out_pgn, "w", encoding="utf-8") as output:
            for game in all_pgns:
                print(game, file=output, end="\n\n")

        references = build_reference_config(specs, fitted, target_name)
        with open(out_config, "w", encoding="utf-8") as output:
            json.dump(references, output, indent=2)

        payload = {
            "schema_version": 1,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "protocol": {
                "nodes": args.nodes,
                "depth": args.depth,
                "movetime_ms": args.movetime_ms,
                "max_plies": args.max_plies,
                "hash_mb": args.hash,
                "threads": args.threads,
                "seed": args.seed,
                "games_per_pair": args.games_per_pair,
                "engine_config": str(Path(args.engines_config).resolve()),
            },
            "anchor": fitted["anchor"],
            "target": target_rating,
            "target_name": target_name,
            "authoritative": authoritative,
            "ratings": fitted["ratings"],
            "calibration": {
                "status": fitted["status"],
                "games": fitted["games"],
                "pairs": aggregate_games(all_games),
            },
            "engines": metadata,
            "artifacts": {
                "json": str(out_json),
                "pgn": str(out_pgn),
                "reference_config": str(out_config),
            },
        }
        with open(out_json, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
        print(f"[ladder] wrote {out_json}")
        print(f"[ladder] wrote {out_pgn}")
        print(f"[ladder] wrote {out_config}")
    except (BenchmarkError, ValueError, chess.engine.EngineError) as exc:
        raise SystemExit(f"reference ladder failed: {exc}") from exc


if __name__ == "__main__":
    main()

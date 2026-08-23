#!/usr/bin/env python3
"""Reproducible Kepler strength benchmark.

The authoritative mode uses full-strength reference engines.  Stockfish's
UCI_LimitStrength/UCI_Elo options are retained only as an explicitly labeled
exploratory mode and never produce an authoritative rating.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import chess.engine
import chess.pgn


OPENINGS: List[List[str]] = [
    [],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["c2c4", "e7e5", "b1c3", "g8f6"],
    ["g1f3", "d7d5", "d2d4", "g8f6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["c2c4", "c7c5", "g1f3", "g8f6"],
    ["g1f3", "g8f6", "c2c4", "e7e5"],
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    ["d2d4", "g8f6", "c2c4", "g7g6"],
    ["c2c4", "e7e6", "g2g3", "d7d5"],
    ["g1f3", "d7d5", "c2c4", "c7c6"],
    ["e2e4", "e7e5", "f2f4", "e5f4"],
    ["d2d4", "d7d5", "c2c4", "c7c6"],
]


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce a valid game result."""


@dataclass(frozen=True)
class OpponentSpec:
    name: str
    path: Path
    rating: Optional[float] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameResult:
    opponent: str
    opponent_rating: Optional[float]
    result: float
    kepler_white: bool
    opening_index: int
    termination: str
    plies: int


@dataclass(frozen=True)
class QualitySample:
    """One Kepler decision replayed from a completed benchmark game."""

    fen: str
    played_move: str
    opponent: str
    opening_index: int
    ply: int
    phase: str


def apply_opening(board: chess.Board, opening: Sequence[str]) -> None:
    for move_text in opening:
        move = chess.Move.from_uci(move_text)
        if move not in board.legal_moves:
            raise BenchmarkError(f"invalid benchmark opening move: {move_text}")
        board.push(move)


def white_result_from_outcome(outcome: chess.Outcome) -> int:
    if outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def result_for_kepler(board: chess.Board, kepler_white: bool) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    kepler_won = (outcome.winner == chess.WHITE) == kepler_white
    return 1.0 if kepler_won else 0.0


def write_pgn(path: Path, games: Iterable[chess.pgn.Game]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        for game in games:
            print(game, file=output, end="\n\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_executable(value: str) -> Path:
    resolved = shutil.which(value)
    path = Path(resolved if resolved else value).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise BenchmarkError(f"engine not found: {value}")
    return path


def build_limit(nodes: int, depth: int, movetime_ms: int) -> chess.engine.Limit:
    selected = sum(value > 0 for value in (nodes, depth, movetime_ms))
    if selected != 1:
        raise ValueError("exactly one of nodes, depth, or movetime_ms must be positive")
    if nodes > 0:
        return chess.engine.Limit(nodes=nodes)
    if depth > 0:
        return chess.engine.Limit(depth=depth)
    return chess.engine.Limit(time=movetime_ms / 1000.0)


def parse_reference_config(path: Path) -> List[OpponentSpec]:
    with open(path, "r", encoding="utf-8") as source:
        raw = json.load(source)
    entries = raw.get("opponents", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise BenchmarkError("reference config must contain a non-empty opponents list")

    specs: List[OpponentSpec] = []
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry or "path" not in entry:
            raise BenchmarkError("each reference must contain name and path")
        rating = entry.get("rating")
        if rating is not None:
            rating = float(rating)
        options = entry.get("options", {})
        if not isinstance(options, dict):
            raise BenchmarkError(f"reference options must be an object: {entry['name']}")
        specs.append(
            OpponentSpec(
                name=str(entry["name"]),
                path=resolve_executable(str(entry["path"])),
                rating=rating,
                options=options,
            )
        )
    return specs


def select_opening(game_index: int, seed: int) -> Tuple[int, List[str]]:
    # Two consecutive games use the same opening and swap colors.
    pair_index = game_index // 2
    opening_index = (seed + pair_index) % len(OPENINGS)
    return opening_index, OPENINGS[opening_index]


def fit_rating(games: Sequence[GameResult]) -> Dict[str, Any]:
    """Fit one Elo parameter against all rated opponents.

    Draws are represented as half-points in the standard logistic Elo
    likelihood. This is the same model used for the familiar Elo expected
    score, but avoids averaging several noisy per-opponent estimates.
    """
    rated = [game for game in games if game.opponent_rating is not None]
    if not rated:
        return {"rating": None, "ci95": [None, None], "status": "unanchored", "rated_games": 0}

    k = math.log(10.0) / 400.0
    minimum = min(float(game.opponent_rating) for game in rated) - 2000.0
    maximum = max(float(game.opponent_rating) for game in rated) + 2000.0
    rating = sum(float(game.opponent_rating) for game in rated) / len(rated)
    status = "converged"

    for _ in range(100):
        residual = 0.0
        information = 0.0
        for game in rated:
            opponent = float(game.opponent_rating)
            probability = 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))
            residual += game.result - probability
            information += probability * (1.0 - probability)
        if information <= 1e-12:
            status = "boundary"
            break
        step = residual / (k * information)
        if abs(step) < 1e-7:
            break
        proposed = rating + step
        if proposed >= maximum and residual > 0:
            rating = maximum
            status = "boundary"
            break
        if proposed <= minimum and residual < 0:
            rating = minimum
            status = "boundary"
            break
        rating = max(minimum, min(maximum, proposed))
    else:
        status = "not_converged"

    information = 0.0
    for game in rated:
        opponent = float(game.opponent_rating)
        probability = 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))
        information += probability * (1.0 - probability)
    standard_error = 1.0 / (k * math.sqrt(information)) if information > 1e-12 else None
    if standard_error is None:
        ci95 = [None, None]
    else:
        ci95 = [rating - 1.96 * standard_error, rating + 1.96 * standard_error]

    return {
        "rating": rating,
        "ci95": ci95,
        "status": status,
        "rated_games": len(rated),
        "standard_error": standard_error,
    }


def configure_kepler(engine: chess.engine.SimpleEngine, args: argparse.Namespace) -> None:
    options = {
        "Hash": max(1, args.hash),
        "Threads": max(1, args.threads),
        "Contempt": max(-100, min(100, args.contempt)),
        "NNUEWeight": max(0, min(100, getattr(args, "nnue_weight", 25))),
        "NNUEClamp": max(0, min(10000, getattr(args, "nnue_clamp", 300))),
        "UseBaseline": False if args.eval_model else True,
        "BaselineEvalFile": str(args.baseline_model),
    }
    if args.eval_model:
        options["EvalFile"] = str(args.eval_model)
    engine.configure(options)


def configure_opponent(
    engine: chess.engine.SimpleEngine,
    spec: OpponentSpec,
    args: argparse.Namespace,
    limited_rating: Optional[int],
) -> None:
    supported = engine.options
    options: Dict[str, Any] = {}
    if "Hash" in supported:
        options["Hash"] = max(1, args.hash)
    if "Threads" in supported:
        options["Threads"] = max(1, args.opponent_threads)
    if args.mode == "limited":
        if limited_rating is None:
            raise BenchmarkError("limited mode requires opponent levels")
        if "UCI_LimitStrength" not in supported or "UCI_Elo" not in supported:
            raise BenchmarkError("limited mode requires UCI_LimitStrength and UCI_Elo")
        options["UCI_LimitStrength"] = True
        options["UCI_Elo"] = limited_rating
    options.update(spec.options)
    # Never let a reference config accidentally turn authoritative mode back
    # into Stockfish's intentionally weakened mode.
    if args.mode == "limited":
        options["UCI_LimitStrength"] = True
        options["UCI_Elo"] = limited_rating
    else:
        if "UCI_LimitStrength" in supported:
            options["UCI_LimitStrength"] = False
        options.pop("UCI_Elo", None)
    for name in options:
        if name not in supported:
            raise BenchmarkError(f"engine does not support option {name}")
    engine.configure(options)


def engine_metadata(path: Path, engine: chess.engine.SimpleEngine) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "id": dict(engine.id),
    }


def add_quality_sample(
    samples: List[QualitySample],
    candidate: QualitySample,
    maximum: int,
    rng: random.Random,
    seen: List[int],
) -> None:
    if maximum <= 0:
        return
    seen[0] += 1
    if len(samples) < maximum:
        samples.append(candidate)
        return
    replacement = rng.randrange(seen[0])
    if replacement < maximum:
        samples[replacement] = candidate


def score_to_cp(score: chess.engine.PovScore, side: chess.Color) -> int:
    value = score.pov(side).score(mate_score=100000)
    return int(value if value is not None else 0)


def score_to_win_probability(
    score: chess.engine.PovScore,
    side: chess.Color,
    ply: int,
) -> Optional[float]:
    """Convert a Stockfish score to its model WDL expected score."""
    try:
        return score.pov(side).wdl(model="sf16", ply=ply).expectation()
    except (AttributeError, TypeError, ValueError):
        return None


def phase_for_board(board: chess.Board) -> str:
    if board.ply() < 20:
        return "opening"
    non_pawn_material = sum(
        len(board.pieces(piece_type, color))
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for color in (chess.WHITE, chess.BLACK)
    )
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if non_pawn_material <= 4 or (queens == 0 and non_pawn_material <= 8):
        return "endgame"
    return "middlegame"


def collect_quality_samples(
    games: Sequence[chess.pgn.Game],
    maximum: int,
    rng: random.Random,
) -> List[QualitySample]:
    """Collect Kepler moves from saved games, optionally by reservoir sampling."""
    samples: List[QualitySample] = []
    seen = [0]
    for game in games:
        board = game.board()
        kepler_white = game.headers.get("White") == "Kepler"
        opponent = game.headers.get("Opponent", game.headers.get("Black", "Unknown"))
        try:
            opening_index = int(game.headers.get("OpeningIndex", "-1"))
        except ValueError:
            opening_index = -1
        for move in game.mainline_moves():
            kepler_turn = (board.turn == chess.WHITE) == kepler_white
            if kepler_turn:
                candidate = QualitySample(
                    fen=board.fen(),
                    played_move=move.uci(),
                    opponent=opponent,
                    opening_index=opening_index,
                    ply=board.ply(),
                    phase=phase_for_board(board),
                )
                if maximum <= 0:
                    samples.append(candidate)
                else:
                    add_quality_sample(samples, candidate, maximum, rng, seen)
            board.push(move)
    return samples


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_quality(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "samples": 0,
            "analyzed": 0,
            "different_best_move": 0,
            "different_best_move_rate": None,
            "best_move_rate": None,
            "near_best_move_rate": None,
            "minor_blunder_rate": None,
            "severe_blunder_rate": None,
            "mate_loss_count": 0,
            "mate_loss_rate": None,
            "average_cp_loss": None,
            "median_cp_loss": None,
            "p90_cp_loss": None,
            "p95_cp_loss": None,
            "p99_cp_loss": None,
            "max_cp_loss": None,
            "average_win_probability_loss": None,
        }
    losses = [float(record["cp_loss"]) for record in records]
    win_probability_losses = [
        float(record["win_probability_loss"])
        for record in records
        if record["win_probability_loss"] is not None
    ]
    count = len(records)
    different = sum(bool(record["different_best_move"]) for record in records)
    near_best = sum(float(record["cp_loss"]) <= 10 for record in records)
    minor = sum(float(record["cp_loss"]) >= 100 for record in records)
    severe = sum(float(record["cp_loss"]) >= 300 for record in records)
    mate_losses = sum(bool(record["mate_loss"]) for record in records)
    return {
        "samples": count,
        "analyzed": count,
        "different_best_move": different,
        "different_best_move_rate": different / count,
        "best_move_rate": 1.0 - different / count,
        "near_best_move_rate": near_best / count,
        "minor_blunder_rate": minor / count,
        "severe_blunder_rate": severe / count,
        "mate_loss_count": mate_losses,
        "mate_loss_rate": mate_losses / count,
        "average_cp_loss": sum(losses) / count,
        "median_cp_loss": percentile(losses, 0.50),
        "p90_cp_loss": percentile(losses, 0.90),
        "p95_cp_loss": percentile(losses, 0.95),
        "p99_cp_loss": percentile(losses, 0.99),
        "max_cp_loss": max(losses),
        "average_win_probability_loss": (
            sum(win_probability_losses) / len(win_probability_losses)
            if win_probability_losses
            else None
        ),
    }


def run_move_quality_audit(
    engine: chess.engine.SimpleEngine,
    samples: Sequence[QualitySample],
    limit: chess.engine.Limit,
    progress_interval: int = 0,
) -> Dict[str, Any]:
    if not samples:
        return {"status": "complete", **summarize_quality([]), "invalid": 0}

    records: List[Dict[str, Any]] = []
    invalid = 0
    for index, sample in enumerate(samples, start=1):
        board = chess.Board(sample.fen)
        played = chess.Move.from_uci(sample.played_move)
        if played not in board.legal_moves:
            invalid += 1
            if progress_interval > 0 and (index == 1 or index % progress_interval == 0 or index == len(samples)):
                print(f"  audit {index}/{len(samples)} analyzed={len(records)} invalid={invalid}", flush=True)
            continue
        best_info = engine.analyse(board, limit)
        principal_variation = best_info.get("pv") or []
        best_move = principal_variation[0] if principal_variation else None
        if best_move is None:
            invalid += 1
            if progress_interval > 0 and (index == 1 or index % progress_interval == 0 or index == len(samples)):
                print(f"  audit {index}/{len(samples)} analyzed={len(records)} invalid={invalid}", flush=True)
            continue
        played_info = engine.analyse(board, limit, root_moves=[played])
        best_cp = score_to_cp(best_info["score"], board.turn)
        played_cp = score_to_cp(played_info["score"], board.turn)
        best_score = best_info["score"].pov(board.turn)
        played_score = played_info["score"].pov(board.turn)
        best_probability = score_to_win_probability(best_info["score"], board.turn, sample.ply)
        played_probability = score_to_win_probability(played_info["score"], board.turn, sample.ply)
        raw_cp_loss = max(0, best_cp - played_cp)
        records.append(
            {
                # Mate scores are not finite centipawns. Keep their separate
                # mate-loss counter, and cap the CPL distribution so one mate
                # does not make the average meaningless.
                "cp_loss": min(raw_cp_loss, 1000),
                "win_probability_loss": (
                    max(
                        0.0,
                        best_probability - played_probability,
                    )
                    if best_probability is not None and played_probability is not None
                    else None
                ),
                "different_best_move": best_move != played,
                "mate_loss": best_score.is_mate() and not played_score.is_mate(),
                "phase": sample.phase,
                "opponent": sample.opponent,
            }
        )
        if progress_interval > 0 and (index == 1 or index % progress_interval == 0 or index == len(samples)):
            print(f"  audit {index}/{len(samples)} analyzed={len(records)} invalid={invalid}", flush=True)

    audit = summarize_quality(records)
    audit["status"] = "complete"
    audit["samples"] = len(samples)
    audit["analyzed"] = len(records)
    audit["invalid"] = invalid
    audit["by_phase"] = {
        phase: summarize_quality([record for record in records if record["phase"] == phase])
        for phase in sorted({record["phase"] for record in records})
    }
    audit["by_opponent"] = {
        opponent: summarize_quality([record for record in records if record["opponent"] == opponent])
        for opponent in sorted({record["opponent"] for record in records})
    }
    return audit


def build_game_pgn(
    board: chess.Board,
    opponent: OpponentSpec,
    kepler_white: bool,
    opening_index: int,
    result: float,
    termination: str,
    game_number: int,
    mode: str,
) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = "Kepler Authoritative Gauntlet"
    game.headers["Date"] = dt.datetime.now(dt.timezone.utc).strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_number)
    game.headers["White"] = "Kepler" if kepler_white else opponent.name
    game.headers["Black"] = opponent.name if kepler_white else "Kepler"
    if result == 0.5:
        game.headers["Result"] = "1/2-1/2"
    elif (result == 1.0) == kepler_white:
        game.headers["Result"] = "1-0"
    else:
        game.headers["Result"] = "0-1"
    game.headers["Termination"] = termination
    game.headers["Opponent"] = opponent.name
    game.headers["OpeningIndex"] = str(opening_index)
    game.headers["BenchmarkMode"] = mode

    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    return game


def play_opponent(
    spec: OpponentSpec,
    engine_path: Path,
    args: argparse.Namespace,
    opponent_index: int,
    opponent_rating: Optional[float],
    limit: chess.engine.Limit,
) -> Tuple[List[GameResult], List[chess.pgn.Game], Dict[str, Any]]:
    results: List[GameResult] = []
    games: List[chess.pgn.Game] = []
    kepler_path = engine_path
    with chess.engine.SimpleEngine.popen_uci(str(kepler_path)) as kepler, chess.engine.SimpleEngine.popen_uci(str(spec.path)) as opponent:
        configure_kepler(kepler, args)
        configure_opponent(opponent, spec, args, int(opponent_rating) if args.mode == "limited" else None)
        opponent_meta = engine_metadata(spec.path, opponent)

        for game_index in range(args.games_per_opponent):
            opening_index, opening = select_opening(game_index, args.seed + opponent_index * 17)
            board = chess.Board()
            apply_opening(board, opening)
            kepler_white = game_index % 2 == 0
            termination = "natural"

            while not board.is_game_over(claim_draw=True):
                if args.max_plies > 0 and board.ply() >= args.max_plies:
                    termination = "adjudicated_max_plies"
                    break
                kepler_turn = (board.turn == chess.WHITE) == kepler_white
                mover = kepler if kepler_turn else opponent
                try:
                    played = mover.play(board, limit)
                except Exception as exc:
                    raise BenchmarkError(
                        f"{spec.name} game {game_index + 1}: engine failure: {exc}"
                    ) from exc
                if played.move is None or played.move not in board.legal_moves:
                    raise BenchmarkError(f"{spec.name} game {game_index + 1}: illegal or missing move")
                board.push(played.move)

            result = result_for_kepler(board, kepler_white)
            if termination == "natural":
                outcome = board.outcome(claim_draw=True)
                termination = outcome.termination.name if outcome and outcome.termination else "draw"
            results.append(
                GameResult(
                    opponent=spec.name,
                    opponent_rating=opponent_rating,
                    result=result,
                    kepler_white=kepler_white,
                    opening_index=opening_index,
                    termination=termination,
                    plies=board.ply(),
                )
            )
            games.append(
                build_game_pgn(
                    board,
                    spec,
                    kepler_white,
                    opening_index,
                    result,
                    termination,
                    game_index + 1,
                    args.mode,
                )
            )
            result_text = "win" if result == 1.0 else "loss" if result == 0.0 else "draw"
            print(
                f"  game {game_index + 1}/{args.games_per_opponent} "
                f"{'W' if kepler_white else 'B'} {result_text} {termination} plies={board.ply()}",
                flush=True,
            )

    return results, games, opponent_meta


def aggregate_results(results: Sequence[GameResult]) -> Dict[str, Any]:
    wins = sum(result.result == 1.0 for result in results)
    losses = sum(result.result == 0.0 for result in results)
    draws = len(results) - wins - losses
    return {
        "games": len(results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": wins + 0.5 * draws,
        "score_rate": (wins + 0.5 * draws) / len(results) if results else None,
        "terminations": {
            termination: sum(result.termination == termination for result in results)
            for termination in sorted({result.termination for result in results})
        },
    }


def parse_levels(text: str) -> List[int]:
    try:
        levels = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise BenchmarkError(f"invalid levels: {text}") from exc
    if not levels or any(level < 1000 or level > 4000 for level in levels):
        raise BenchmarkError("limited-mode levels must be between 1000 and 4000")
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Kepler against full-strength references or explicitly limited Stockfish."
    )
    parser.add_argument("--engine", default="build/kepler", help="Path to Kepler binary.")
    parser.add_argument("--stockfish", default="stockfish", help="Path to default Stockfish binary.")
    parser.add_argument("--baseline-model", default="models/kepler_baseline_pst_v1.nnue")
    parser.add_argument("--eval-model", default="", help="Optional candidate NNUE to load with UseBaseline=false.")
    parser.add_argument("--reference-config", default="", help="JSON list of named reference engines and ratings.")
    parser.add_argument("--stockfish-anchor-rating", type=float, default=None, help="Explicit rating anchor for default Stockfish.")
    parser.add_argument("--mode", choices=["full", "limited"], default="full", help="Full-strength authoritative mode or exploratory UCI-Elo mode.")
    parser.add_argument("--levels", default="1400,1600,1800,2000", help="Limited-mode Stockfish UCI_Elo levels.")
    parser.add_argument("--games-per-opponent", "--games-per-level", dest="games_per_opponent", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=100000, help="Fixed node budget (default benchmark limit).")
    parser.add_argument("--depth", type=int, default=0, help="Use fixed depth instead of nodes.")
    parser.add_argument("--movetime-ms", type=int, default=0, help="Use fixed time instead of nodes/depth.")
    parser.add_argument("--max-plies", type=int, default=400, help="Adjudicate unfinished games as draws at this ply count; 0 disables.")
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1, help="Kepler search threads.")
    parser.add_argument("--opponent-threads", type=int, default=1)
    parser.add_argument("--contempt", type=int, default=0)
    parser.add_argument("--nnue-weight", type=int, default=25, help="Neural share of evaluation, 0-100.")
    parser.add_argument("--nnue-clamp", type=int, default=300, help="Maximum NNUE/classical difference; 0 disables.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic opening/color seed.")
    parser.add_argument(
        "--audit-samples",
        type=int,
        default=0,
        help="Audit this many Kepler moves from the saved PGNs with full-strength Stockfish.",
    )
    parser.add_argument(
        "--audit-all",
        action="store_true",
        help="Audit every Kepler move from the saved PGNs with full-strength Stockfish.",
    )
    parser.add_argument("--audit-hash", type=int, default=1024, help="Stockfish Hash size for move-quality analysis.")
    parser.add_argument("--audit-threads", type=int, default=1, help="Stockfish threads for move-quality analysis.")
    parser.add_argument("--audit-nodes", type=int, default=500000)
    parser.add_argument("--audit-depth", type=int, default=0)
    parser.add_argument("--audit-movetime-ms", type=int, default=0)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-pgn", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        engine_path = resolve_executable(args.engine)
        args.stockfish = resolve_executable(args.stockfish)
        args.baseline_model = Path(args.baseline_model).resolve() if hasattr(args, "baseline_model") else Path("models/kepler_baseline_pst_v1.nnue").resolve()
        args.eval_model = Path(args.eval_model).resolve() if hasattr(args, "eval_model") and args.eval_model else None
        if not args.baseline_model.exists():
            raise BenchmarkError(f"baseline model not found: {args.baseline_model}")
        if args.eval_model and not args.eval_model.exists():
            raise BenchmarkError(f"eval model not found: {args.eval_model}")
        if args.games_per_opponent < 1:
            raise BenchmarkError("games-per-opponent must be positive")
        if args.games_per_opponent % 2:
            print("[gauntlet] warning: odd game count leaves one color unpaired", flush=True)
        if args.hash < 1 or args.threads < 1 or args.opponent_threads < 1:
            raise BenchmarkError("hash and thread counts must be positive")
        if args.audit_samples < 0:
            raise BenchmarkError("audit-samples cannot be negative")
        if args.audit_all and args.audit_samples:
            raise BenchmarkError("use either --audit-all or --audit-samples, not both")
        if args.audit_hash < 1 or args.audit_threads < 1:
            raise BenchmarkError("audit hash and thread counts must be positive")
        limit = build_limit(args.nodes, args.depth, args.movetime_ms)
        quality_enabled = args.audit_all or args.audit_samples > 0
        audit_limit = (
            build_limit(args.audit_nodes, args.audit_depth, args.audit_movetime_ms)
            if quality_enabled
            else limit
        )

        if args.reference_config:
            if args.mode == "limited":
                raise BenchmarkError("reference-config is only valid in full mode")
            opponents = parse_reference_config(Path(args.reference_config).resolve())
        elif args.mode == "full":
            opponents = [OpponentSpec("Stockfish", args.stockfish, args.stockfish_anchor_rating)]
        else:
            opponents = [OpponentSpec("Stockfish", args.stockfish)]
        levels = parse_levels(args.levels) if args.mode == "limited" else [None]

        all_results: List[GameResult] = []
        all_games: List[chess.pgn.Game] = []
        metadata: Dict[str, Any] = {}
        results_by_opponent: Dict[str, Any] = {}
        opponent_index = 0

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = Path(args.out_json).resolve() if args.out_json else Path(f"/tmp/kepler_rating_{timestamp}.json")
        out_pgn = Path(args.out_pgn).resolve() if args.out_pgn else Path(f"/tmp/kepler_rating_{timestamp}.pgn")

        for opponent in opponents:
            ratings = levels if args.mode == "limited" else [opponent.rating]
            for limited_rating in ratings:
                display_name = (
                    f"{opponent.name}[{limited_rating}]"
                    if args.mode == "limited" and limited_rating is not None
                    else opponent.name
                )
                effective_spec = OpponentSpec(display_name, opponent.path, opponent.rating, opponent.options)
                print(f"[gauntlet] opponent {display_name} mode={args.mode} limit={limit}", flush=True)
                results, games, engine_meta = play_opponent(
                    effective_spec,
                    engine_path,
                    args,
                    opponent_index,
                    float(limited_rating) if limited_rating is not None else opponent.rating,
                    limit,
                )
                all_results.extend(results)
                all_games.extend(games)
                metadata[display_name] = engine_meta
                metadata[display_name]["benchmark_options"] = {
                    "Hash": max(1, args.hash),
                    "Threads": max(1, args.opponent_threads),
                    "UCI_LimitStrength": args.mode == "limited",
                    "UCI_Elo": int(limited_rating) if args.mode == "limited" else None,
                    "custom": effective_spec.options,
                }
                results_by_opponent[display_name] = aggregate_results(results)
                print(f"[opponent {display_name}] {results_by_opponent[display_name]}", flush=True)
                opponent_index += 1

        # Persist the raw game record before starting the optional, expensive
        # move-quality pass. This makes the benchmark recoverable even if the
        # evaluator is interrupted or fails after the games complete.
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_pgn.parent.mkdir(parents=True, exist_ok=True)
        write_pgn(out_pgn, all_games)

        quality: Dict[str, Any] = {"status": "disabled", "samples": 0}
        if quality_enabled:
            rng = random.Random(args.seed)
            quality_samples = collect_quality_samples(
                all_games,
                0 if args.audit_all else args.audit_samples,
                rng,
            )
            print(
                f"[gauntlet] move-quality audit: {len(quality_samples)} Kepler moves "
                f"with full-strength evaluator limit={audit_limit}",
                flush=True,
            )
            with chess.engine.SimpleEngine.popen_uci(str(args.stockfish)) as evaluator:
                evaluator.configure(
                    {
                        "Hash": max(1, args.audit_hash),
                        "Threads": max(1, args.audit_threads),
                        "UCI_LimitStrength": False,
                    }
                )
                quality = run_move_quality_audit(
                    evaluator,
                    quality_samples,
                    audit_limit,
                    progress_interval=25,
                )
                quality["evaluator"] = engine_metadata(args.stockfish, evaluator)
                quality["evaluator"]["benchmark_options"] = {
                    "Hash": max(1, args.audit_hash),
                    "Threads": max(1, args.audit_threads),
                    "UCI_LimitStrength": False,
                }
            print(
                f"[gauntlet] move-quality average_cpl={quality['average_cp_loss']} "
                f"best_move_rate={quality['best_move_rate']}",
                flush=True,
            )

        fitted = fit_rating(all_results)
        overall = aggregate_results(all_results)
        if args.mode == "limited":
            fitted["status"] = "limited_mode_not_authoritative"
        elif fitted["status"] == "unanchored":
            print("[gauntlet] no external rating anchor supplied; reporting score only", flush=True)
        else:
            print(
                f"[gauntlet] calibrated_rating={fitted['rating']:.1f} "
                f"95% CI=[{fitted['ci95'][0]:.1f}, {fitted['ci95'][1]:.1f}]",
                flush=True,
            )

        payload = {
            "schema_version": 3,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "artifacts": {"json": str(out_json), "pgn": str(out_pgn)},
            "engine": {"path": str(engine_path), "sha256": sha256_file(engine_path)},
            "baseline_model": {
                "path": str(args.baseline_model),
                "sha256": sha256_file(args.baseline_model),
            },
            "eval_model": (
                {"path": str(args.eval_model), "sha256": sha256_file(args.eval_model)}
                if args.eval_model
                else None
            ),
            "mode": args.mode,
            "authoritative": args.mode == "full" and fitted["status"] == "converged",
            "protocol": {
                "nodes": args.nodes,
                "depth": args.depth,
                "movetime_ms": args.movetime_ms,
                "max_plies": args.max_plies,
                "hash_mb": args.hash,
                "kepler_threads": args.threads,
                "opponent_threads": args.opponent_threads,
                "seed": args.seed,
                "games_per_opponent": args.games_per_opponent,
                "opponent_mode": args.mode,
                "limited_levels": levels if args.mode == "limited" else [],
                "reference_config": str(Path(args.reference_config).resolve()) if args.reference_config else "",
                "audit_samples": args.audit_samples,
                "audit_all": args.audit_all,
                "audit_hash_mb": args.audit_hash,
                "audit_threads": args.audit_threads,
                "audit_nodes": args.audit_nodes,
                "audit_depth": args.audit_depth,
                "audit_movetime_ms": args.audit_movetime_ms,
            },
            "opponents": metadata,
            "results": overall,
            "results_by_opponent": results_by_opponent,
            "rating": fitted,
            "move_quality_audit": quality,
            "calibration_note": (
                "Full-strength mode is authoritative only when explicit reference ratings are supplied. "
                "Limited UCI_Elo mode is exploratory and intentionally not an absolute rating."
            ),
            "results_by_game": [result.__dict__ for result in all_results],
        }
        with open(out_json, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
        print(f"[gauntlet] wrote {out_json}")
        print(f"[gauntlet] wrote {out_pgn}")
    except (BenchmarkError, ValueError, chess.engine.EngineError) as exc:
        raise SystemExit(f"benchmark failed: {exc}") from exc


if __name__ == "__main__":
    main()

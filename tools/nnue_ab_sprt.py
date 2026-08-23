#!/usr/bin/env python3
"""Paired nonlinear-NNUE A/B benchmark with sequential stopping.

Model A is the baseline and model B is the candidate.  Every opening is
played twice with the model colors swapped, so opening and color effects are
paired.  Results are scored from model B's perspective and fed to a
draw-aware, fixed-hypothesis Wald SPRT-style test.

The test hypotheses are explicit:

    H0: candidate gain is elo0 or worse
    H1: candidate gain is elo1 or better

The draw rate is a declared nuisance parameter (rather than silently
pretending every game is decisive).  The final result is also reported as a
raw score and an approximate Elo difference; the SPRT decision is the
promotion gate.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import chess.engine
import chess.pgn

from nnue_ab import OPENINGS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_probabilities(elo: float, draw_rate: float) -> Dict[str, float]:
    """Return win/draw/loss probabilities for the candidate.

    The logistic expected score is split around a fixed draw probability.  It
    is deliberately simple and transparent; callers should report the
    declared draw rate alongside any SPRT result.
    """

    draw = max(0.0, min(0.95, draw_rate))
    expected_score = 1.0 / (1.0 + 10.0 ** (-elo / 400.0))
    if draw >= 1.0:
        return {"win": 0.0, "draw": 1.0, "loss": 0.0}
    # Expected score is P(win) + 0.5*P(draw); no renormalization is needed.
    win = expected_score - 0.5 * draw
    win = max(0.0, min(1.0 - draw, win))
    return {"win": win, "draw": draw, "loss": 1.0 - draw - win}


def llr_increment(
    result: str,
    elo0: float,
    elo1: float,
    draw_rate: float,
) -> float:
    if result not in {"win", "draw", "loss"}:
        raise ValueError(f"unknown result: {result}")
    p0 = result_probabilities(elo0, draw_rate)[result]
    p1 = result_probabilities(elo1, draw_rate)[result]
    return math.log(max(1e-15, p1) / max(1e-15, p0))


def sprt_bounds(alpha: float, beta: float) -> Tuple[float, float]:
    alpha = max(1e-9, min(1.0 - 1e-9, alpha))
    beta = max(1e-9, min(1.0 - 1e-9, beta))
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def sprt_decision(llr: float, lower: float, upper: float) -> str:
    if llr <= lower:
        return "reject_candidate"
    if llr >= upper:
        return "accept_candidate"
    return "continue"


def paired_sprt_decision(
    games_played: int, minimum_games: int, llr: float, lower: float, upper: float
) -> str:
    if games_played < minimum_games or games_played % 2:
        return "continue"
    return sprt_decision(llr, lower, upper)


def candidate_score(result: str, candidate_is_white: bool) -> Tuple[str, float]:
    if result == "1-0":
        score = 1.0 if candidate_is_white else 0.0
    elif result == "0-1":
        score = 0.0 if candidate_is_white else 1.0
    else:
        score = 0.5
    outcome = "win" if score == 1.0 else ("loss" if score == 0.0 else "draw")
    return outcome, score


def choose_limit(args: argparse.Namespace) -> chess.engine.Limit:
    if args.nodes is not None:
        return chess.engine.Limit(nodes=max(1, args.nodes))
    if args.movetime is not None:
        return chess.engine.Limit(time=max(0.001, args.movetime / 1000.0))
    if args.depth is not None:
        return chess.engine.Limit(depth=max(1, args.depth))
    return chess.engine.Limit(nodes=100_000)


def kepler_options(
    hash_mb: int, threads: int, model: Path, nnue_weight: int, nnue_clamp: int
) -> Dict[str, object]:
    return {
        "Hash": max(1, hash_mb),
        "Threads": max(1, threads),
        "NNUEWeight": max(0, min(100, nnue_weight)),
        "NNUEClamp": max(0, min(10000, nnue_clamp)),
        "UseBaseline": False,
        "EvalFile": str(model),
    }


def apply_opening(opening: Sequence[str]) -> chess.Board:
    board = chess.Board()
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal opening move {uci}")
        board.push(move)
    return board


def play_game(
    engine_a: chess.engine.SimpleEngine,
    engine_b: chess.engine.SimpleEngine,
    opening: Sequence[str],
    a_is_white: bool,
    limit: chess.engine.Limit,
    max_plies: int,
) -> Tuple[str, List[str], str]:
    board = apply_opening(opening)
    moves = list(opening)
    termination = "max_plies"

    while len(moves) < max_plies and not board.is_game_over(claim_draw=True):
        use_a = board.turn == chess.WHITE and a_is_white
        use_a = use_a or (board.turn == chess.BLACK and not a_is_white)
        engine = engine_a if use_a else engine_b
        result = engine.play(board, limit)
        move = result.move
        if move is None or move not in board.legal_moves:
            raise RuntimeError(f"engine returned illegal/no move: {move}")
        board.push(move)
        moves.append(move.uci())

    if board.is_checkmate():
        termination = "checkmate"
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        termination = "draw_rule"
    return board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2", moves, termination


def game_to_pgn(record: dict) -> chess.pgn.Game:
    game = chess.pgn.Game()
    for key in (
        "Event", "Date", "Round", "White", "Black", "Result", "Termination",
        "OpeningIndex", "PairIndex", "SPRTDecision",
    ):
        if key in record:
            game.headers[key] = str(record[key])
    board = chess.Board()
    node = game
    for uci in record["moves"]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)
        node = node.add_variation(move)
    return game


def write_pgn(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            print(game_to_pgn(record), file=output, end="\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired A/B SPRT-style NNUE benchmark.")
    parser.add_argument("--engine", default="build-release/kepler")
    parser.add_argument(
        "--engine-b",
        default="",
        help="Optional candidate engine binary; defaults to --engine for NNUE-only tests.",
    )
    parser.add_argument("--model-a", required=True, help="Baseline model.")
    parser.add_argument("--model-b", required=True, help="Candidate model.")
    parser.add_argument("--games", type=int, default=64, help="Maximum games; must be even for pairing.")
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument("--nodes", type=int, default=None)
    limit_group.add_argument("--movetime", type=int, default=None, help="Milliseconds per move.")
    limit_group.add_argument("--depth", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--nnue-weight-a", type=int, default=25)
    parser.add_argument("--nnue-weight-b", type=int, default=25)
    parser.add_argument("--nnue-clamp-a", type=int, default=300, help="0 disables the baseline clamp.")
    parser.add_argument("--nnue-clamp-b", type=int, default=300, help="0 disables the candidate clamp.")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--min-games", type=int, default=8, help="Do not stop before this many games.")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=30.0)
    parser.add_argument("--draw-rate", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--pgn-out", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.games < 2 or args.games % 2:
        parser.error("--games must be an even number of paired games")
    if args.elo1 <= args.elo0:
        parser.error("--elo1 must be greater than --elo0")
    if args.min_games < 2 or args.min_games > args.games or args.min_games % 2:
        parser.error("--min-games must be even and between 2 and --games")
    for option, value, maximum in (
        ("--nnue-weight-a", args.nnue_weight_a, 100),
        ("--nnue-weight-b", args.nnue_weight_b, 100),
        ("--nnue-clamp-a", args.nnue_clamp_a, 10000),
        ("--nnue-clamp-b", args.nnue_clamp_b, 10000),
    ):
        if value < 0 or value > maximum:
            parser.error(f"{option} must be between 0 and {maximum}")

    engine_path = Path(args.engine).resolve()
    engine_b_path = Path(args.engine_b).resolve() if args.engine_b else engine_path
    model_a = Path(args.model_a).resolve()
    model_b = Path(args.model_b).resolve()
    for path in (engine_path, engine_b_path, model_a, model_b):
        if not path.exists():
            raise SystemExit(f"missing path: {path}")

    lower, upper = sprt_bounds(args.alpha, args.beta)
    limit = choose_limit(args)
    rng = random.Random(args.seed)
    opening_indexes = list(range(len(OPENINGS)))
    rng.shuffle(opening_indexes)

    a_wins = b_wins = draws = 0
    llr = 0.0
    decision = "continue"
    records: List[dict] = []

    with (
        chess.engine.SimpleEngine.popen_uci(str(engine_path), timeout=args.timeout) as engine_a,
        chess.engine.SimpleEngine.popen_uci(str(engine_b_path), timeout=args.timeout) as engine_b,
    ):
        engine_a.configure(
            kepler_options(
                args.hash, args.threads, model_a, args.nnue_weight_a, args.nnue_clamp_a
            )
        )
        engine_b.configure(
            kepler_options(
                args.hash, args.threads, model_b, args.nnue_weight_b, args.nnue_clamp_b
            )
        )

        for game_index in range(args.games):
            pair_index = game_index // 2
            opening_index = opening_indexes[pair_index % len(opening_indexes)]
            opening = OPENINGS[opening_index]
            a_is_white = game_index % 2 == 0
            result, moves, termination = play_game(
                engine_a, engine_b, opening, a_is_white, limit, args.max_plies
            )
            candidate_is_white = not a_is_white
            outcome, score = candidate_score(result, candidate_is_white)
            llr += llr_increment(outcome, args.elo0, args.elo1, args.draw_rate)
            if result == "1-0":
                if a_is_white:
                    a_wins += 1
                else:
                    b_wins += 1
            elif result == "0-1":
                if a_is_white:
                    b_wins += 1
                else:
                    a_wins += 1
            else:
                draws += 1

            record = {
                "Event": "Kepler NNUE Paired SPRT",
                "Date": dt.datetime.now(dt.timezone.utc).strftime("%Y.%m.%d"),
                "Round": game_index + 1,
                "White": "A-baseline" if a_is_white else "B-candidate",
                "Black": "B-candidate" if a_is_white else "A-baseline",
                "Result": result,
                "Termination": termination,
                "OpeningIndex": opening_index,
                "PairIndex": pair_index + 1,
                "SPRTDecision": decision,
                "moves": moves,
            }
            records.append(record)
            games_played = game_index + 1
            # Never stop between the two color-swapped games of an opening.
            decision = paired_sprt_decision(
                games_played, args.min_games, llr, lower, upper
            )
            record["SPRTDecision"] = decision
            print(
                f"game {games_played}/{args.games} pair={pair_index + 1} "
                f"opening={opening_index} result={result} candidate={outcome} "
                f"llr={llr:.3f} decision={decision}",
                flush=True,
            )
            if decision != "continue":
                break

    total = len(records)
    candidate_score_total = b_wins + 0.5 * draws
    score_percent = candidate_score_total / max(1, total)
    score_clamped = max(1e-6, min(1.0 - 1e-6, score_percent))
    observed_elo = 400.0 * math.log10(score_clamped / (1.0 - score_clamped))
    if decision == "continue":
        decision = "inconclusive"

    report = {
        "status": decision,
        "baseline": str(model_a),
        "candidate": str(model_b),
        "baseline_sha256": sha256_file(model_a),
        "candidate_sha256": sha256_file(model_b),
        "baseline_engine": str(engine_path),
        "candidate_engine": str(engine_b_path),
        "baseline_engine_sha256": sha256_file(engine_path),
        "candidate_engine_sha256": sha256_file(engine_b_path),
        "baseline_evaluation": {
            "nnue_weight": args.nnue_weight_a,
            "nnue_clamp": args.nnue_clamp_a,
        },
        "candidate_evaluation": {
            "nnue_weight": args.nnue_weight_b,
            "nnue_clamp": args.nnue_clamp_b,
        },
        "games": total,
        "paired_games": total // 2,
        "baseline_wins": a_wins,
        "candidate_wins": b_wins,
        "draws": draws,
        "candidate_score": candidate_score_total,
        "candidate_score_percent": score_percent,
        "approx_observed_elo_difference": observed_elo,
        "sprt": {
            "elo0": args.elo0,
            "elo1": args.elo1,
            "draw_rate": args.draw_rate,
            "alpha": args.alpha,
            "beta": args.beta,
            "lower_boundary": lower,
            "upper_boundary": upper,
            "final_llr": llr,
        },
        "seed": args.seed,
        "limit": {
            "nodes": args.nodes,
            "movetime_ms": args.movetime,
            "depth": args.depth,
        },
    }
    print(json.dumps(report, indent=2))

    if args.pgn_out:
        pgn_path = Path(args.pgn_out).resolve()
        write_pgn(pgn_path, records)
        print(f"PGN {pgn_path}")
    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON {json_path}")

    return {"accept_candidate": 0, "reject_candidate": 1, "inconclusive": 2}[decision]


if __name__ == "__main__":
    raise SystemExit(main())

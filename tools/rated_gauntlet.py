#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn


OPENINGS = [
    [],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["c2c4", "e7e5", "b1c3", "g8f6"],
    ["g1f3", "d7d5", "d2d4", "g8f6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
]


@dataclass
class LevelResult:
    elo: int
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def score_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.score / self.games


def apply_opening(board: chess.Board, opening: List[str]) -> bool:
    for mv in opening:
        move = chess.Move.from_uci(mv)
        if move not in board.legal_moves:
            return False
        board.push(move)
    return True


def result_from_board(board: chess.Board) -> int:
    # Returns +1 white win, -1 black win, 0 draw/unknown.
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0
    if outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def write_pgn(path: Path, games: List[chess.pgn.Game]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for g in games:
            print(g, file=f, end="\n\n")


def elo_from_score(opponent_elo: int, p: float) -> float:
    # Elo model: p = 1 / (1 + 10^((opp - self)/400))
    p = max(0.01, min(0.99, p))
    delta = 400.0 * math.log10(p / (1.0 - p))
    return opponent_elo + delta


def main() -> None:
    ap = argparse.ArgumentParser(description="Rate Kepler vs Stockfish UCI_Elo levels.")
    ap.add_argument("--engine", default="build/kepler", help="Path to Kepler binary.")
    ap.add_argument("--stockfish", default="stockfish", help="Path to Stockfish binary.")
    ap.add_argument("--baseline-model", default="models/kepler_baseline_pst_v1.nnue", help="Baseline NNUE file path.")
    ap.add_argument("--levels", default="1400,1600,1800,2000", help="Comma-separated Stockfish UCI_Elo levels.")
    ap.add_argument("--games-per-level", type=int, default=8)
    ap.add_argument("--movetime-ms", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=140)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--out-json", default="", help="Optional JSON output path.")
    ap.add_argument("--out-pgn", default="", help="Optional PGN output path.")
    args = ap.parse_args()

    engine_path = Path(args.engine).resolve()
    sf_path = Path(args.stockfish)
    model_path = Path(args.baseline_model).resolve()
    if not engine_path.exists():
        raise SystemExit(f"Engine not found: {engine_path}")
    if not model_path.exists():
        raise SystemExit(f"Baseline model not found: {model_path}")

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    if not levels:
        raise SystemExit("No valid levels provided.")

    limit = chess.engine.Limit(time=max(0.001, args.movetime_ms / 1000.0))
    all_games: List[chess.pgn.Game] = []
    level_results: Dict[int, LevelResult] = {elo: LevelResult(elo=elo) for elo in levels}

    for elo in levels:
        print(f"[gauntlet] level {elo}")
        with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as eng, chess.engine.SimpleEngine.popen_uci(str(sf_path)) as sf:
            eng.configure(
                {
                    "Hash": args.hash,
                    "UseBaseline": True,
                    "BaselineEvalFile": str(model_path),
                }
            )
            sf.configure({"Hash": args.hash, "UCI_LimitStrength": True, "UCI_Elo": elo})

            for g in range(args.games_per_level):
                board = chess.Board()
                opening = OPENINGS[g % len(OPENINGS)]
                if not apply_opening(board, opening):
                    continue

                kepler_white = (g % 2 == 0)
                ply = 0
                while ply < args.max_plies and not board.is_game_over(claim_draw=True):
                    mover = eng if (board.turn == chess.WHITE) == kepler_white else sf
                    result = mover.play(board, limit)
                    if result.move is None:
                        break
                    if result.move not in board.legal_moves:
                        break
                    board.push(result.move)
                    ply += 1

                white_result = result_from_board(board)
                lr = level_results[elo]
                lr.games += 1
                if white_result == 0:
                    lr.draws += 1
                else:
                    kepler_won = (white_result == 1 and kepler_white) or (white_result == -1 and not kepler_white)
                    if kepler_won:
                        lr.wins += 1
                    else:
                        lr.losses += 1

                game = chess.pgn.Game()
                game.headers["Event"] = "Kepler Rated Gauntlet"
                game.headers["Date"] = dt.datetime.utcnow().strftime("%Y.%m.%d")
                game.headers["Round"] = f"{elo}-{g+1}"
                game.headers["White"] = "Kepler" if kepler_white else f"Stockfish[{elo}]"
                game.headers["Black"] = f"Stockfish[{elo}]" if kepler_white else "Kepler"
                if white_result == 1:
                    game.headers["Result"] = "1-0"
                elif white_result == -1:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
                game.headers["StockfishElo"] = str(elo)
                game.headers["KeplerModel"] = model_path.name

                node = game
                replay_board = chess.Board()
                apply_opening(replay_board, opening)
                for mv in opening:
                    node = node.add_variation(chess.Move.from_uci(mv))
                for mv in board.move_stack[len(opening):]:
                    node = node.add_variation(mv)
                all_games.append(game)

                print(
                    f"  game {g+1}/{args.games_per_level} "
                    f"{'W' if kepler_white else 'B'} "
                    f"res={game.headers['Result']}"
                )

    estimates = []
    for elo in levels:
        lr = level_results[elo]
        p = lr.score_rate
        est = elo_from_score(elo, p)
        estimates.append((est, lr.games))
        print(
            f"[level {elo}] games={lr.games} W/L/D={lr.wins}/{lr.losses}/{lr.draws} "
            f"score={p:.3f} est_kepler={est:.1f}"
        )

    total_weight = sum(g for _, g in estimates)
    weighted_est = sum(est * g for est, g in estimates) / max(1, total_weight)
    print(f"[gauntlet] estimated_kepler_elo={weighted_est:.1f} (anchored to Stockfish UCI_Elo scale)")

    out_json = Path(args.out_json).resolve() if args.out_json else Path(f"/tmp/kepler_rating_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_pgn = Path(args.out_pgn).resolve() if args.out_pgn else Path(f"/tmp/kepler_rating_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn")

    payload = {
        "engine": str(engine_path),
        "stockfish": str(sf_path),
        "model": str(model_path),
        "levels": levels,
        "games_per_level": args.games_per_level,
        "movetime_ms": args.movetime_ms,
        "max_plies": args.max_plies,
        "results": {
            str(elo): {
                "wins": level_results[elo].wins,
                "losses": level_results[elo].losses,
                "draws": level_results[elo].draws,
                "games": level_results[elo].games,
                "score_rate": level_results[elo].score_rate,
                "estimated_elo": elo_from_score(elo, level_results[elo].score_rate),
            }
            for elo in levels
        },
        "estimated_kepler_elo": weighted_est,
        "note": "Estimate is anchored to Stockfish UCI_LimitStrength/UCI_Elo, so treat as approximate.",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_pgn(out_pgn, all_games)
    print(f"[gauntlet] wrote {out_json}")
    print(f"[gauntlet] wrote {out_pgn}")


if __name__ == "__main__":
    main()

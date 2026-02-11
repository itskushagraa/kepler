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
    truncated: int = 0

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


def white_result_from_outcome(outcome: chess.Outcome) -> int:
    # Returns +1 white win, -1 black win, 0 draw.
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
    ap.add_argument(
        "--eval-model",
        default="",
        help="Optional candidate NNUE to load via EvalFile with UseBaseline=false.",
    )
    ap.add_argument("--levels", default="1400,1600,1800,2000", help="Comma-separated Stockfish UCI_Elo levels.")
    ap.add_argument("--games-per-level", type=int, default=8)
    ap.add_argument("--movetime-ms", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=0, help="Maximum plies per game. Use 0 to disable the limit.")
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--threads", type=int, default=1, help="Kepler search threads.")
    ap.add_argument("--contempt", type=int, default=0, help="Kepler contempt in centipawns.")
    ap.add_argument("--out-json", default="", help="Optional JSON output path.")
    ap.add_argument("--out-pgn", default="", help="Optional PGN output path.")
    args = ap.parse_args()

    engine_path = Path(args.engine).resolve()
    sf_path = Path(args.stockfish)
    model_path = Path(args.baseline_model).resolve()
    eval_model_path: Optional[Path] = Path(args.eval_model).resolve() if args.eval_model else None
    if not engine_path.exists():
        raise SystemExit(f"Engine not found: {engine_path}")
    if not model_path.exists():
        raise SystemExit(f"Baseline model not found: {model_path}")
    if eval_model_path and not eval_model_path.exists():
        raise SystemExit(f"Eval model not found: {eval_model_path}")

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    if not levels:
        raise SystemExit("No valid levels provided.")

    limit = chess.engine.Limit(time=max(0.001, args.movetime_ms / 1000.0))
    max_plies = max(0, args.max_plies)
    all_games: List[chess.pgn.Game] = []
    level_results: Dict[int, LevelResult] = {elo: LevelResult(elo=elo) for elo in levels}

    for elo in levels:
        print(f"[gauntlet] level {elo}")
        with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as eng, chess.engine.SimpleEngine.popen_uci(str(sf_path)) as sf:
            eng.configure(
                {
                    "Hash": args.hash,
                    "Threads": max(1, args.threads),
                    "Contempt": max(-100, min(100, args.contempt)),
                    "UseBaseline": False if eval_model_path else True,
                    "BaselineEvalFile": str(model_path),
                    "EvalFile": str(eval_model_path) if eval_model_path else "",
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
                while not board.is_game_over(claim_draw=True):
                    if max_plies > 0 and ply >= max_plies:
                        break
                    mover = eng if (board.turn == chess.WHITE) == kepler_white else sf
                    result = mover.play(board, limit)
                    if result.move is None:
                        break
                    if result.move not in board.legal_moves:
                        break
                    board.push(result.move)
                    ply += 1

                lr = level_results[elo]
                outcome = board.outcome(claim_draw=True)
                game_truncated = outcome is None

                if game_truncated:
                    lr.truncated += 1
                else:
                    white_result = white_result_from_outcome(outcome)
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
                if game_truncated:
                    game.headers["Result"] = "*"
                    game.headers["Termination"] = "unterminated(max plies limit)"
                else:
                    white_result = white_result_from_outcome(outcome)
                    if white_result == 1:
                        game.headers["Result"] = "1-0"
                    elif white_result == -1:
                        game.headers["Result"] = "0-1"
                    else:
                        game.headers["Result"] = "1/2-1/2"
                    game.headers["Termination"] = str(outcome.termination)
                game.headers["StockfishElo"] = str(elo)
                game.headers["KeplerModel"] = eval_model_path.name if eval_model_path else model_path.name

                node = game
                replay_board = chess.Board()
                apply_opening(replay_board, opening)
                for mv in opening:
                    node = node.add_variation(chess.Move.from_uci(mv))
                for mv in board.move_stack[len(opening):]:
                    node = node.add_variation(mv)
                all_games.append(game)

                status = "truncated" if game_truncated else f"res={game.headers['Result']}"
                print(f"  game {g+1}/{args.games_per_level} {'W' if kepler_white else 'B'} {status}")

    estimates = []
    for elo in levels:
        lr = level_results[elo]
        if lr.games == 0:
            print(
                f"[level {elo}] completed=0 truncated={lr.truncated} "
                f"W/L/D={lr.wins}/{lr.losses}/{lr.draws} score=n/a est_kepler=n/a"
            )
            continue
        p = lr.score_rate
        est = elo_from_score(elo, p)
        estimates.append((est, lr.games))
        print(
            f"[level {elo}] games={lr.games} truncated={lr.truncated} W/L/D={lr.wins}/{lr.losses}/{lr.draws} "
            f"score={p:.3f} est_kepler={est:.1f}"
        )

    if estimates:
        total_weight = sum(g for _, g in estimates)
        weighted_est = sum(est * g for est, g in estimates) / max(1, total_weight)
        print(f"[gauntlet] estimated_kepler_elo={weighted_est:.1f} (anchored to Stockfish UCI_Elo scale)")
    else:
        weighted_est = None
        print("[gauntlet] estimated_kepler_elo=n/a (no completed games)")

    out_json = Path(args.out_json).resolve() if args.out_json else Path(f"/tmp/kepler_rating_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_pgn = Path(args.out_pgn).resolve() if args.out_pgn else Path(f"/tmp/kepler_rating_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn")

    payload = {
        "engine": str(engine_path),
        "stockfish": str(sf_path),
        "baseline_model": str(model_path),
        "eval_model": str(eval_model_path) if eval_model_path else "",
        "levels": levels,
        "games_per_level": args.games_per_level,
        "movetime_ms": args.movetime_ms,
        "max_plies": args.max_plies,
        "threads": max(1, args.threads),
        "contempt": max(-100, min(100, args.contempt)),
        "results": {
            str(elo): {
                "wins": level_results[elo].wins,
                "losses": level_results[elo].losses,
                "draws": level_results[elo].draws,
                "truncated": level_results[elo].truncated,
                "games": level_results[elo].games,
                "score_rate": level_results[elo].score_rate,
                "estimated_elo": (
                    elo_from_score(elo, level_results[elo].score_rate)
                    if level_results[elo].games > 0
                    else None
                ),
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

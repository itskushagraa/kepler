#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import chess
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


def run_uci(engine: Path, commands: List[str], timeout_s: int = 30) -> str:
    payload = "\n".join(commands + ["quit"]) + "\n"
    p = subprocess.run(
        [str(engine)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "engine exited with error")
    return p.stdout


def query_move(
    engine: Path,
    model: Path,
    moves: List[str],
    movetime_ms: int,
    depth: int,
    timeout_s: int,
) -> Tuple[Optional[str], int]:
    pos = "position startpos"
    if moves:
        pos += " moves " + " ".join(moves)
    probe = f"probe movetime {movetime_ms}"
    if depth > 0:
        probe += f" depth {depth}"
    out = run_uci(
        engine,
        ["uci", f"setoption name EvalFile value {model}", pos, probe],
        timeout_s=timeout_s,
    )
    m_best = re.search(r"bestmove (\S+)", out)
    if not m_best:
        return None, 0
    best = m_best.group(1)
    m_score = re.search(r"probe depth \d+ scorecp (-?\d+)", out)
    score = int(m_score.group(1)) if m_score else 0
    return best, score


def play_game(
    engine: Path,
    model_white: Path,
    model_black: Path,
    opening: List[str],
    movetime_ms: int,
    depth: int,
    max_plies: int,
    adjudicate_cp: int,
    adjudicate_after: int,
    timeout_s: int,
) -> Tuple[int, List[str], str]:
    moves = list(opening)
    ply = len(moves)
    termination = "draw"
    board = chess.Board()
    for mv in opening:
        try:
            m = chess.Move.from_uci(mv)
            if m not in board.legal_moves:
                return 0, moves, "bad_opening"
            board.push(m)
        except Exception:
            return 0, moves, "bad_opening"

    while ply < max_plies:
        white_to_move = (ply % 2 == 0)
        model = model_white if white_to_move else model_black
        best, score = query_move(engine, model, moves, movetime_ms, depth, timeout_s)
        if not best or best == "0000":
            if board.is_checkmate():
                termination = "checkmate"
                # side to move is checkmated
                return (-1 if board.turn == chess.WHITE else 1), moves, termination
            if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                termination = "draw_rule"
                return 0, moves, termination
            termination = "no_move"
            return 0, moves, termination

        if ply >= adjudicate_after and abs(score) >= adjudicate_cp:
            side_winning = white_to_move if score > 0 else (not white_to_move)
            termination = "adjudication"
            return (1 if side_winning else -1), moves, termination

        try:
            move_obj = chess.Move.from_uci(best)
            if move_obj not in board.legal_moves:
                termination = "illegal_move"
                return 0, moves, termination
            board.push(move_obj)
        except Exception:
            termination = "bad_uci"
            return 0, moves, termination

        moves.append(best)
        ply += 1

    if board.is_checkmate():
        termination = "checkmate"
        return (-1 if board.turn == chess.WHITE else 1), moves, termination
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        termination = "draw_rule"
        return 0, moves, termination
    termination = "max_plies"
    return 0, moves, termination


def write_games_pgn(path: Path, games: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for g in games:
            game = chess.pgn.Game()
            game.headers["Event"] = "Kepler NNUE A/B"
            game.headers["Date"] = dt.datetime.utcnow().strftime("%Y.%m.%d")
            game.headers["Round"] = str(g["index"])
            game.headers["White"] = g["white_name"]
            game.headers["Black"] = g["black_name"]
            game.headers["Result"] = g["result_str"]
            game.headers["Termination"] = g["termination"]
            game.headers["OpeningSeed"] = str(g["opening_index"])

            board = chess.Board()
            node = game
            for mv in g["moves"]:
                try:
                    move = chess.Move.from_uci(mv)
                    if move not in board.legal_moves:
                        node.comment = (node.comment + " " if node.comment else "") + f"illegal_uci:{mv}"
                        break
                    board.push(move)
                    node = node.add_variation(move)
                except Exception:
                    node.comment = (node.comment + " " if node.comment else "") + f"bad_uci:{mv}"
                    break

            print(game, file=f, end="\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B match runner for two NNUE models on Kepler.")
    ap.add_argument("--engine", default="build/kepler")
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--movetime", type=int, default=15)
    ap.add_argument("--depth", type=int, default=0, help="If >0, probe uses fixed depth.")
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--adjudicate-cp", type=int, default=300)
    ap.add_argument("--adjudicate-after", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--pgn-out", default="", help="Optional PGN output path (defaults in /tmp).")
    args = ap.parse_args()

    engine = Path(args.engine).resolve()
    model_a = Path(args.model_a).resolve()
    model_b = Path(args.model_b).resolve()
    if not engine.exists():
        raise SystemExit(f"missing engine: {engine}")
    if not model_a.exists():
        raise SystemExit(f"missing model-a: {model_a}")
    if not model_b.exists():
        raise SystemExit(f"missing model-b: {model_b}")

    a_wins = 0
    b_wins = 0
    draws = 0
    game_records = []

    for g in range(args.games):
        opening = OPENINGS[g % len(OPENINGS)]
        a_is_white = (g % 2 == 0)
        mw = model_a if a_is_white else model_b
        mb = model_b if a_is_white else model_a
        result, moves, termination = play_game(
            engine,
            mw,
            mb,
            opening,
            args.movetime,
            args.depth,
            args.max_plies,
            args.adjudicate_cp,
            args.adjudicate_after,
            args.timeout,
        )
        if result == 1:
            if a_is_white:
                a_wins += 1
            else:
                b_wins += 1
        elif result == -1:
            if a_is_white:
                b_wins += 1
            else:
                a_wins += 1
        else:
            draws += 1
        result_str = "1-0" if result == 1 else ("0-1" if result == -1 else "1/2-1/2")
        white_name = f"A:{model_a.stem}" if a_is_white else f"B:{model_b.stem}"
        black_name = f"B:{model_b.stem}" if a_is_white else f"A:{model_a.stem}"
        game_records.append(
            {
                "index": g + 1,
                "opening_index": g % len(OPENINGS),
                "white_name": white_name,
                "black_name": black_name,
                "result_str": result_str,
                "termination": termination,
                "moves": moves,
            }
        )
        print(
            f"game {g+1}/{args.games} opening_len={len(opening)} "
            f"a_is_white={int(a_is_white)} result={result_str} term={termination}"
        )

    print(f"RESULT A_wins={a_wins} B_wins={b_wins} Draws={draws}")
    score = a_wins + 0.5 * draws
    total = max(1, args.games)
    print(f"A_score={score:.1f}/{total}")

    pgn_path = Path(args.pgn_out).resolve() if args.pgn_out else Path(f"/tmp/kepler_ab_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn")
    write_games_pgn(pgn_path, game_records)
    print(f"PGN {pgn_path}")


if __name__ == "__main__":
    main()

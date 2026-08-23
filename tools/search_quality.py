#!/usr/bin/env python3
"""Measure Kepler move quality on reproducibly sampled PGN positions."""

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import List, Tuple

import chess
import chess.engine
import chess.pgn


def collect_positions(path: Path, name: str) -> List[Tuple[str, str]]:
    positions: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as source:
        while game := chess.pgn.read_game(source):
            board = game.board()
            for move in game.mainline_moves():
                player = game.headers.get("White" if board.turn else "Black", "")
                if name.lower() in player.lower() and len(board.move_stack) >= 8:
                    positions.append((board.fen(), move.uci()))
                board.push(move)
    return positions


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    blend = index - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit search move quality against Stockfish.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--engine-name", default="Kepler")
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--sf-nodes", type=int, default=500000)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    engine_path = Path(args.engine).resolve()
    stockfish_path = Path(args.stockfish).resolve()
    pgn_path = Path(args.pgn).resolve()
    for path in (engine_path, stockfish_path, pgn_path):
        if not path.exists():
            raise SystemExit(f"missing path: {path}")

    available = collect_positions(pgn_path, args.engine_name)
    rng = random.Random(args.seed)
    chosen = rng.sample(available, min(max(1, args.positions), len(available)))
    details = []
    losses: List[float] = []
    best_matches = 0
    engine_limit = chess.engine.Limit(nodes=max(1, args.nodes))
    sf_limit = chess.engine.Limit(nodes=max(1, args.sf_nodes))
    common = {"Hash": max(1, args.hash), "Threads": max(1, args.threads)}

    with (
        chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine,
        chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as stockfish,
    ):
        engine_options = dict(common)
        if args.model:
            engine_options["EvalFile"] = str(Path(args.model).resolve())
        engine.configure(engine_options)
        stockfish.configure({**common, "UCI_LimitStrength": False})

        for index, (fen, played_pgn) in enumerate(chosen, 1):
            board = chess.Board(fen)
            original_side = board.turn
            reference = stockfish.analyse(board, sf_limit)
            pv = reference.get("pv", [])
            if not pv or reference.get("score") is None:
                continue
            best_move = pv[0]
            before = reference["score"].pov(original_side).score(mate_score=20000)
            selected = engine.play(board, engine_limit).move
            if selected is None or selected not in board.legal_moves or before is None:
                continue
            if selected == best_move:
                best_matches += 1
            board.push(selected)
            after_info = stockfish.analyse(board, sf_limit)
            after_score = after_info.get("score")
            after = after_score.pov(original_side).score(mate_score=20000) if after_score else None
            if after is None:
                continue
            loss = float(max(0, before - after))
            losses.append(loss)
            details.append(
                {
                    "fen": fen,
                    "pgn_move": played_pgn,
                    "engine_move": selected.uci(),
                    "stockfish_move": best_move.uci(),
                    "cp_before": before,
                    "cp_after": after,
                    "cp_loss": loss,
                }
            )
            print(
                f"position {index}/{len(chosen)} move={selected.uci()} "
                f"best={best_move.uci()} loss={loss:.0f}",
                flush=True,
            )

    report = {
        "engine": str(engine_path),
        "model": str(Path(args.model).resolve()) if args.model else "",
        "pgn": str(pgn_path),
        "available_positions": len(available),
        "sampled_positions": len(chosen),
        "analyzed_positions": len(losses),
        "engine_nodes": args.nodes,
        "stockfish_nodes": args.sf_nodes,
        "best_move_rate": best_matches / max(1, len(losses)),
        "average_cp_loss": statistics.fmean(losses) if losses else 0.0,
        "median_cp_loss": statistics.median(losses) if losses else 0.0,
        "p90_cp_loss": percentile(losses, 0.90),
        "max_cp_loss": max(losses, default=0.0),
        "details": details,
    }
    output = Path(args.json_out).resolve()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
    print("JSON", output)


if __name__ == "__main__":
    main()

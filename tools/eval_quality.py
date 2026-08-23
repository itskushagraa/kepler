#!/usr/bin/env python3
"""Measure Kepler's static evaluation against held-out teacher labels.

This deliberately performs no search. It separates evaluation error from
move-selection/search error and reports calibration by phase, material class,
sample kind, and score bucket.
"""

import argparse
import hashlib
import json
import math
import queue
import random
import re
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import chess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_position(board: chess.Board) -> tuple[str, str]:
    nonpawn = {
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    nonpawn_material = sum(
        value * len(board.pieces(piece, color))
        for piece, value in nonpawn.items()
        for color in chess.COLORS
    )
    ply = board.ply()
    if ply <= 20 and nonpawn_material >= 5200:
        phase = "opening"
    elif nonpawn_material <= 2600:
        phase = "endgame"
    else:
        phase = "middlegame"

    queens = len(board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(chess.QUEEN, chess.BLACK))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE) | board.pieces(chess.ROOK, chess.BLACK))
    minors = sum(
        len(board.pieces(piece, color))
        for piece in (chess.BISHOP, chess.KNIGHT)
        for color in chess.COLORS
    )
    pawns = len(board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK))
    if queens:
        material = "queens"
    elif rooks:
        material = "rooks_no_queens"
    elif minors:
        material = "minor_only"
    elif pawns:
        material = "pawn_only"
    else:
        material = "bare_kings"
    return phase, material


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            try:
                teacher = int(fields[1])
                board = chess.Board(fields[2])
            except (ValueError, TypeError):
                continue
            phase, material = classify_position(board)
            rows.append(
                {
                    "line": line_number,
                    "teacher_cp": teacher,
                    "fen": board.fen(),
                    "phase": phase,
                    "material": material,
                    "bucket": fields[4] if len(fields) > 4 and fields[4] else "unknown",
                    "kind": fields[8] if len(fields) > 8 and fields[8] else "regular",
                }
            )
    return rows


def summarize(rows: Iterable[dict]) -> dict:
    values = list(rows)
    if not values:
        return {"count": 0}
    errors = [row["engine_cp"] - row["teacher_cp"] for row in values]
    abs_errors = [abs(error) for error in errors]
    squared = [error * error for error in errors]
    sign_correct = sum(
        (row["engine_cp"] > 0) == (row["teacher_cp"] > 0)
        for row in values
        if row["engine_cp"] != 0 and row["teacher_cp"] != 0
    )
    sign_total = sum(
        row["engine_cp"] != 0 and row["teacher_cp"] != 0 for row in values
    )
    engine_mean = sum(row["engine_cp"] for row in values) / len(values)
    teacher_mean = sum(row["teacher_cp"] for row in values) / len(values)
    covariance = sum(
        (row["engine_cp"] - engine_mean) * (row["teacher_cp"] - teacher_mean)
        for row in values
    )
    engine_variance = sum((row["engine_cp"] - engine_mean) ** 2 for row in values)
    teacher_variance = sum((row["teacher_cp"] - teacher_mean) ** 2 for row in values)
    denominator = math.sqrt(engine_variance * teacher_variance)
    return {
        "count": len(values),
        "mae_cp": sum(abs_errors) / len(values),
        "rmse_cp": math.sqrt(sum(squared) / len(values)),
        "bias_cp": sum(errors) / len(values),
        "sign_accuracy": sign_correct / sign_total if sign_total else None,
        "pearson_correlation": covariance / denominator if denominator else None,
    }


def grouped(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


class KeplerEvalClient:
    def __init__(self, engine: Path, options: dict[str, object], timeout: float):
        self.process = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.timeout = timeout
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        self._send("uci")
        self._read_until(lambda line: line == "uciok")
        for name, value in options.items():
            self._send(f"setoption name {name} value {str(value).lower() if isinstance(value, bool) else value}")
        self._send("isready")
        self._read_until(lambda line: line == "readyok")

    def _send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def _read_until(self, predicate) -> str:
        deadline = time.monotonic() + max(0.1, self.timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.process.kill()
                self.process.wait()
                raise TimeoutError(f"Kepler did not reply within {self.timeout:.1f} seconds")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as error:
                self.process.kill()
                self.process.wait()
                raise TimeoutError(
                    f"Kepler did not reply within {self.timeout:.1f} seconds"
                ) from error
            if line is None:
                raise RuntimeError(
                    f"Kepler exited before replying (return code {self.process.poll()})"
                )
            if predicate(line.strip()):
                return line.strip()

    def evaluate(self, fen: str) -> tuple[int, str]:
        self._send(f"position fen {fen}")
        self._send("eval")
        line = self._read_until(lambda value: value.startswith("info string eval score "))
        match = re.search(r"eval score (-?\d+) source (\S+)", line)
        if not match:
            raise RuntimeError(f"could not parse eval response: {line}")
        return int(match.group(1)), match.group(2)

    def close(self) -> None:
        if self.process.poll() is None:
            self._send("quit")
            try:
                self.process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit static eval against held-out teacher data.")
    parser.add_argument("--engine", default="build-release/kepler")
    parser.add_argument("--data", required=True, help="Held-out teacher TSV, preferably validation.tsv.")
    parser.add_argument("--model", default="", help="Optional non-baseline NNUE model.")
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--nnue-weight", type=int, default=25)
    parser.add_argument("--nnue-clamp", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    engine = Path(args.engine).resolve()
    data = Path(args.data).resolve()
    model = Path(args.model).resolve() if args.model else None
    for path in (engine, data, *([model] if model else [])):
        if path is None or not path.is_file():
            raise SystemExit(f"missing path: {path}")

    available = read_rows(data)
    if not available:
        raise SystemExit("no valid positions found")
    rng = random.Random(args.seed)
    chosen = rng.sample(available, min(max(1, args.positions), len(available)))
    options: dict[str, object] = {
        "NNUEWeight": max(0, min(100, args.nnue_weight)),
        "NNUEClamp": max(0, min(10000, args.nnue_clamp)),
    }
    if model:
        options.update({"UseBaseline": False, "EvalFile": str(model)})

    evaluated = []
    sources: dict[str, int] = defaultdict(int)
    with KeplerEvalClient(engine, options, args.timeout) as client:
        for index, row in enumerate(chosen, 1):
            engine_cp, source = client.evaluate(row["fen"])
            sources[source] += 1
            evaluated.append({**row, "engine_cp": engine_cp})
            if index % 250 == 0 or index == len(chosen):
                print(f"evaluated {index}/{len(chosen)}", flush=True)

    report = {
        "engine": str(engine),
        "engine_sha256": sha256_file(engine),
        "data": str(data),
        "data_sha256": sha256_file(data),
        "model": str(model) if model else "baseline",
        "model_sha256": sha256_file(model) if model else None,
        "settings": {
            "positions": args.positions,
            "nnue_weight": args.nnue_weight,
            "nnue_clamp": args.nnue_clamp,
            "seed": args.seed,
        },
        "available_positions": len(available),
        "evaluated_positions": len(evaluated),
        "evaluation_sources": dict(sorted(sources.items())),
        "overall": summarize(evaluated),
        "by_phase": grouped(evaluated, "phase"),
        "by_material": grouped(evaluated, "material"),
        "by_kind": grouped(evaluated, "kind"),
        "by_score_bucket": grouped(evaluated, "bucket"),
        "worst_positions": sorted(
            evaluated,
            key=lambda row: abs(row["engine_cp"] - row["teacher_cp"]),
            reverse=True,
        )[:50],
    }
    output = Path(args.json_out).resolve()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "worst_positions"}, indent=2))
    print("JSON", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

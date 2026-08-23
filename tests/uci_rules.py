#!/usr/bin/env python3
"""Protocol regressions for mate reporting and full-history repetition."""

import subprocess
import sys


def run(engine: str, commands: list[str]) -> str:
    completed = subprocess.run(
        [engine],
        input="\n".join(["uci", "isready", *commands, "quit"]) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=True,
    )
    return completed.stdout


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: uci_rules.py <kepler>")
    engine = sys.argv[1]

    false_mate = run(
        engine,
        [
            "setoption name Threads value 1",
            "position fen r3kb1r/ppp1pppp/2n5/4P3/6b1/5N2/PPP2PP1/RNBK1B1R b q - 0 8",
            "probe depth 2",
        ],
    )
    if "scoremate" in false_mate or "scorecp 29997" in false_mate:
        raise AssertionError(false_mate)

    mate = run(
        engine,
        [
            "setoption name Threads value 1",
            "position fen 7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            "probe depth 2",
        ],
    )
    if "scoremate 1" not in mate:
        raise AssertionError(mate)

    repetition = run(
        engine,
        [
            "position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8",
            "probe depth 4",
        ],
    )
    if "scorecp 0" not in repetition:
        raise AssertionError(repetition)

    print("UCI mate and repetition rules passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

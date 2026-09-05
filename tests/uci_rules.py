#!/usr/bin/env python3
"""Protocol regressions for mate reporting and full-history repetition."""

import subprocess
import sys
import time


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


def read_until(process: subprocess.Popen[str], marker: str) -> list[str]:
    lines: list[str] = []
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            raise AssertionError(f"engine exited before {marker}: {lines}")
        lines.append(line)
        if marker in line:
            return lines


def timed_bullet_search(engine: str) -> tuple[float, str]:
    process = subprocess.Popen(
        [engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    try:
        process.stdin.write("uci\n")
        process.stdin.flush()
        uci_output = "".join(read_until(process, "uciok"))
        expected = "option name MoveOverhead type spin default 10 min 0 max 5000"
        if expected not in uci_output:
            raise AssertionError(uci_output)

        process.stdin.write(
            "setoption name Threads value 4\n"
            "setoption name MoveOverhead value 900\n"
            "isready\n"
        )
        process.stdin.flush()
        read_until(process, "readyok")

        process.stdin.write(
            "position startpos\n"
            "go wtime 57000 btime 57000 winc 0 binc 0\n"
        )
        process.stdin.flush()
        start = time.monotonic()
        search_lines = read_until(process, "bestmove")
        elapsed = time.monotonic() - start
        return elapsed, "".join(search_lines)
    finally:
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
        process.wait(timeout=5)


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

    elapsed, bullet = timed_bullet_search(engine)
    if "bestmove 0000" in bullet or elapsed > 1.0:
        raise AssertionError(f"unsafe threaded bullet search ({elapsed:.3f}s):\n{bullet}")

    print(f"UCI rules and threaded bullet deadline passed ({elapsed:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

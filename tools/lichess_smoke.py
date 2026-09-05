#!/usr/bin/env python3
"""Offline end-to-end smoke test against the pinned lichess-bot bridge."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = REPO_ROOT / ".local" / "lichess-bot"
CONFIG_PATH = REPO_ROOT / "deploy" / "lichess" / "config.yml"

# lichess-bot is intentionally an ignored runtime dependency, not vendored
# Kepler source. Import its real configuration and engine wrappers here.
sys.path.insert(0, str(BRIDGE_DIR))
os.chdir(BRIDGE_DIR)

import chess  # noqa: E402
import chess.engine  # noqa: E402
from lib.config import load_config  # noqa: E402
from lib import engine_wrapper  # noqa: E402
from lib.timer import seconds  # noqa: E402


def main() -> int:
    config = load_config(str(CONFIG_PATH))

    # Kepler patches the pinned bridge's fixed ten-second first move. Verify
    # that 1+0 gets a bullet-sized movetime before starting an engine.
    first_move = engine_wrapper.first_move_time(
        SimpleNamespace(id="kepler-first-move-smoke", clock_initial=seconds(60))
    )
    if first_move.time is None or not 0.1 <= first_move.time <= 0.6:
        raise RuntimeError(f"unsafe bullet first-move limit: {first_move.time}")

    # This exercises the bridge's own config validation, UCI option discovery,
    # startup, isready exchange, and clean shutdown without contacting Lichess.
    with engine_wrapper.create_engine(config):
        pass

    engine_path = (BRIDGE_DIR / config.engine.dir / config.engine.name).resolve()
    working_dir = (BRIDGE_DIR / config.engine.working_dir).resolve()
    board = chess.Board()
    game_id = "kepler-lichess-offline-smoke"
    limit = chess.engine.Limit(
        white_clock=5.0,
        black_clock=5.0,
        white_inc=1.0,
        black_inc=1.0,
    )

    with chess.engine.SimpleEngine.popen_uci(
        str(engine_path), timeout=15, cwd=str(working_dir)
    ) as engine:
        engine.configure(dict(config.engine.uci_options.items()))
        for _ in range(2):
            result = engine.play(board, limit, game=game_id)
            if result.move not in board.legal_moves:
                raise RuntimeError(f"bridge-style search returned illegal move {result.move}")
            board.push(result.move)

    print(
        "PASS: patched lichess-bot loaded Kepler; bullet first-move and clock-managed searches completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

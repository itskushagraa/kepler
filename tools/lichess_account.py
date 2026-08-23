#!/usr/bin/env python3
"""Safely inspect the Lichess account selected by LICHESS_BOT_TOKEN."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = REPO_ROOT / ".local" / "lichess-bot"
TOKEN_ENV = "LICHESS_BOT_TOKEN"


def summarize_profile(profile: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    username = profile.get("username") or profile.get("id") or "unknown"
    title = profile.get("title")
    count = profile.get("count")
    games = count.get("all") if isinstance(count, dict) else None
    is_bot = title == "BOT"
    eligible = games == 0 and not is_bot

    if is_bot:
        state = "already_bot"
        safe = True
    elif eligible:
        state = "eligible_zero_games"
        safe = True
    elif games is None:
        state = "blocked_unknown_game_count"
        safe = False
    else:
        state = "blocked_has_played_games"
        safe = False

    return (
        {
            "username": username,
            "title": title or "none",
            "games_played": games if games is not None else "unknown",
            "bot_play_scope": True,
            "upgrade_state": state,
        },
        safe,
    )


def main() -> int:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if len(token) < 16:
        raise SystemExit(f"{TOKEN_ENV} is not set.")

    sys.path.insert(0, str(BRIDGE_DIR))
    os.chdir(BRIDGE_DIR)
    from lib.lichess import Lichess  # noqa: E402

    # Lichess validates bot:play in the constructor. Neither the token-test
    # response nor the token itself is printed or written to disk.
    client = Lichess(token, "https://lichess.org/", "kepler-account-check", logging.WARNING, 2)
    summary, safe = summarize_profile(client.get_profile())
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0 if safe else 2


if __name__ == "__main__":
    raise SystemExit(main())

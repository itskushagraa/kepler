#!/usr/bin/env python3
"""Install, validate, and run Kepler through the official lichess-bot bridge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = REPO_ROOT / ".local" / "lichess-bot"
BRIDGE_URL = "https://github.com/lichess-bot-devs/lichess-bot.git"
BRIDGE_COMMIT = "df7e730de58cc3ef2f1415a0dc2eeda842d39167"
VENV_PYTHON = BRIDGE_DIR / ".venv" / "bin" / "python"
CONFIG_PATH = REPO_ROOT / "deploy" / "lichess" / "config.yml"
LOCK_PATH = REPO_ROOT / "deploy" / "lichess" / "requirements.lock.txt"
SMOKE_PATH = REPO_ROOT / "tools" / "lichess_smoke.py"
ENGINE_PATH = REPO_ROOT / "build-release" / "kepler"
TOKEN_ENV = "LICHESS_BOT_TOKEN"


def run_checked(command: Sequence[str | Path], cwd: Path = REPO_ROOT) -> None:
    rendered = [str(part) for part in command]
    print("+", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=cwd, check=True)


def bridge_head() -> str | None:
    if not (BRIDGE_DIR / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=BRIDGE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_bridge() -> None:
    if not (BRIDGE_DIR / ".git").is_dir():
        BRIDGE_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "clone", "--depth", "1", BRIDGE_URL, BRIDGE_DIR])

    current = bridge_head()
    if current != BRIDGE_COMMIT:
        # This directory is managed, ignored deployment state. Do not use it
        # for local bridge development; the pin keeps production reproducible.
        run_checked(["git", "fetch", "--depth", "1", "origin", BRIDGE_COMMIT], BRIDGE_DIR)
        run_checked(["git", "checkout", "--detach", BRIDGE_COMMIT], BRIDGE_DIR)


def ensure_venv() -> None:
    if not VENV_PYTHON.is_file():
        run_checked([sys.executable, "-m", "venv", BRIDGE_DIR / ".venv"])
    run_checked([VENV_PYTHON, "-m", "pip", "install", "-r", LOCK_PATH])
    run_checked([VENV_PYTHON, "-m", "pip", "check"])


def build_engine(jobs: int) -> None:
    build_dir = REPO_ROOT / "build-release"
    if not (build_dir / "CMakeCache.txt").is_file():
        run_checked(
            ["cmake", "-S", REPO_ROOT, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"]
        )
    run_checked(["cmake", "--build", build_dir, "-j", str(max(1, jobs))])


def require_local_install() -> None:
    missing = []
    if bridge_head() != BRIDGE_COMMIT:
        missing.append(f"official bridge pinned at {BRIDGE_COMMIT[:8]}")
    if not VENV_PYTHON.is_file():
        missing.append("bridge virtual environment")
    if not ENGINE_PATH.is_file():
        missing.append("Release Kepler binary")
    if missing:
        raise SystemExit(
            "Missing " + ", ".join(missing) + ". Run: python3 tools/lichess_bot.py setup"
        )


def require_token() -> None:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if len(token) < 16 or token == "set-via-LICHESS_BOT_TOKEN":
        raise SystemExit(
            f"{TOKEN_ENV} is not set. Load the bot:play token into the environment; "
            "never put it in config.yml or a command-line argument."
        )


def smoke() -> None:
    require_local_install()
    (REPO_ROOT / "deploy" / "lichess" / "games").mkdir(parents=True, exist_ok=True)
    run_checked([VENV_PYTHON, SMOKE_PATH], BRIDGE_DIR)


def bridge_command(extra: Sequence[str]) -> int:
    require_local_install()
    require_token()
    command = [
        str(VENV_PYTHON),
        str(BRIDGE_DIR / "lichess-bot.py"),
        "--config",
        str(CONFIG_PATH),
        *extra,
    ]
    return subprocess.run(command, cwd=BRIDGE_DIR, check=False).returncode


def command_setup(args: argparse.Namespace) -> int:
    ensure_bridge()
    ensure_venv()
    build_engine(args.jobs)
    smoke()
    print("Lichess deployment setup and offline bridge smoke test passed.")
    return 0


def command_status(_: argparse.Namespace) -> int:
    checks = {
        "bridge_commit": bridge_head() or "missing",
        "bridge_pinned": bridge_head() == BRIDGE_COMMIT,
        "venv_python": VENV_PYTHON.is_file(),
        "release_engine": ENGINE_PATH.is_file(),
        "config": CONFIG_PATH.is_file(),
        "token_loaded": bool(os.environ.get(TOKEN_ENV, "").strip()),
    }
    for key, value in checks.items():
        print(f"{key}: {value}")
    return 0 if all(checks[key] for key in ["bridge_pinned", "venv_python", "release_engine", "config"]) else 1


def command_run(args: argparse.Namespace) -> int:
    return bridge_command(["-v"] if args.verbose else [])


def command_smoke(_: argparse.Namespace) -> int:
    smoke()
    return 0


def command_upgrade(args: argparse.Namespace) -> int:
    if not args.confirm_irreversible:
        raise SystemExit(
            "BOT conversion is irreversible. Re-run with --confirm-irreversible only after "
            "verifying this account has never played a game."
        )
    extra = ["-u"]
    if args.verbose:
        extra.append("-v")
    return bridge_command(extra)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Manage Kepler's pinned, secret-safe lichess-bot deployment."
    )
    sub = result.add_subparsers(dest="command", required=True)

    setup = sub.add_parser(
        "setup", help="Install the pinned bridge, build Kepler, and smoke-test it."
    )
    setup.add_argument("--jobs", type=int, default=4)
    setup.set_defaults(func=command_setup)

    status = sub.add_parser(
        "status", help="Show local deployment readiness without contacting Lichess."
    )
    status.set_defaults(func=command_status)

    smoke_parser = sub.add_parser(
        "smoke", help="Run the offline bridge and clock-management test."
    )
    smoke_parser.set_defaults(func=command_smoke)

    run_parser = sub.add_parser(
        "run", help="Run the already-upgraded bot account in the foreground."
    )
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.set_defaults(func=command_run)

    upgrade = sub.add_parser(
        "upgrade", help="Irreversibly upgrade the account, then run the bot."
    )
    upgrade.add_argument("--confirm-irreversible", action="store_true")
    upgrade.add_argument("--verbose", action="store_true")
    upgrade.set_defaults(func=command_upgrade)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

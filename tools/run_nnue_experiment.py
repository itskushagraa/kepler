#!/usr/bin/env python3
"""Run static-eval audits, A/A sanity, and resumable pentanomial A/B."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "tools" / "nnue_ab_sprt.py"
EVAL = ROOT / "tools" / "eval_quality.py"


def run(command: list[str], allowed: set[int]) -> int:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in allowed:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a complete trustworthy NNUE promotion experiment.")
    parser.add_argument("--engine", default="build-release/kepler")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validation-data", default="")
    parser.add_argument("--aa-games", type=int, default=100)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--min-games", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--baseline-weight", type=int, default=25)
    parser.add_argument("--candidate-weight", type=int, default=75)
    parser.add_argument("--baseline-clamp", type=int, default=300)
    parser.add_argument("--candidate-clamp", type=int, default=300)
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=10.0)
    parser.add_argument("--draw-rate", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = Path(args.engine).resolve()
    baseline = Path(args.baseline).resolve()
    candidate = Path(args.candidate).resolve()
    for path in (engine, baseline, candidate, AB, EVAL):
        if not path.is_file():
            raise SystemExit(f"missing path: {path}")
    if args.aa_games % 2 or args.games % 2 or args.min_games % 2:
        parser.error("game counts must be even")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.validation_data:
        validation = Path(args.validation_data).resolve()
        for name, model, weight, clamp in (
            ("baseline", baseline, args.baseline_weight, args.baseline_clamp),
            ("candidate", candidate, args.candidate_weight, args.candidate_clamp),
        ):
            run(
                [
                    sys.executable, str(EVAL),
                    "--engine", str(engine),
                    "--data", str(validation),
                    "--model", str(model),
                    "--positions", "2000",
                    "--nnue-weight", str(weight),
                    "--nnue-clamp", str(clamp),
                    "--seed", str(args.seed),
                    "--json-out", str(out_dir / f"static-eval-{name}.json"),
                ],
                {0},
            )

    aa_json = out_dir / "aa-sanity.json"
    aa_pgn = out_dir / "aa-sanity.pgn"
    common = [
        "--engine", str(engine),
        "--nodes", str(max(1, args.nodes)),
        "--threads", str(max(1, args.threads)),
        "--hash", str(max(1, args.hash)),
        "--max-plies", "240",
        "--seed", str(args.seed),
        "--draw-rate", str(args.draw_rate),
        "--elo0", str(args.elo0),
        "--elo1", str(args.elo1),
    ]
    run(
        [
            sys.executable, str(AB), *common,
            "--model-a", str(baseline),
            "--model-b", str(baseline),
            "--nnue-weight-a", str(args.baseline_weight),
            "--nnue-weight-b", str(args.baseline_weight),
            "--nnue-clamp-a", str(args.baseline_clamp),
            "--nnue-clamp-b", str(args.baseline_clamp),
            "--games", str(args.aa_games),
            "--min-games", str(min(args.aa_games, max(2, args.min_games))),
            "--aa-sanity", "--resume",
            "--json-out", str(aa_json),
            "--pgn-out", str(aa_pgn),
        ],
        {0},
    )

    ab_json = out_dir / "ab-result.json"
    ab_pgn = out_dir / "ab-games.pgn"
    ab_exit_code = run(
        [
            sys.executable, str(AB), *common,
            "--model-a", str(baseline),
            "--model-b", str(candidate),
            "--nnue-weight-a", str(args.baseline_weight),
            "--nnue-weight-b", str(args.candidate_weight),
            "--nnue-clamp-a", str(args.baseline_clamp),
            "--nnue-clamp-b", str(args.candidate_clamp),
            "--games", str(args.games),
            "--min-games", str(args.min_games),
            "--require-aa-report", str(aa_json),
            "--resume",
            "--json-out", str(ab_json),
            "--pgn-out", str(ab_pgn),
        ],
        {0, 1, 2},
    )
    result = json.loads(ab_json.read_text(encoding="utf-8"))
    pipeline = {
        "status": result["status"],
        "exit_code": ab_exit_code,
        "aa_report": str(aa_json),
        "ab_report": str(ab_json),
        "static_eval_baseline": str(out_dir / "static-eval-baseline.json") if args.validation_data else None,
        "static_eval_candidate": str(out_dir / "static-eval-candidate.json") if args.validation_data else None,
    }
    pipeline_path = out_dir / "experiment.json"
    pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
    print(f"experiment status: {result['status']}")
    print(f"pipeline report: {pipeline_path}")
    return ab_exit_code


if __name__ == "__main__":
    raise SystemExit(main())

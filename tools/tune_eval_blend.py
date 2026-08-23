#!/usr/bin/env python3
"""Grid-search NNUE blend/clamp calibration on held-out teacher positions."""

import argparse
import json
import random
from pathlib import Path

from eval_quality import KeplerEvalClient, read_rows, summarize


def integer_list(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune NNUE weight/clamp on held-out positions.")
    parser.add_argument("--engine", default="build-release/kepler")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--weights", type=integer_list, default=integer_list("25,50,75,100"))
    parser.add_argument("--clamps", type=integer_list, default=integer_list("0,150,300,600"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    engine = Path(args.engine).resolve()
    model = Path(args.model).resolve()
    data = Path(args.data).resolve()
    for path in (engine, model, data):
        if not path.is_file():
            raise SystemExit(f"missing path: {path}")
    available = read_rows(data)
    chosen = random.Random(args.seed).sample(
        available, min(max(1, args.positions), len(available))
    )

    candidates = []
    for weight in args.weights:
        for clamp in args.clamps:
            evaluated = []
            options = {
                "UseBaseline": False,
                "EvalFile": str(model),
                "NNUEWeight": max(0, min(100, weight)),
                "NNUEClamp": max(0, min(10000, clamp)),
            }
            with KeplerEvalClient(engine, options, 30.0) as client:
                for row in chosen:
                    engine_cp, _ = client.evaluate(row["fen"])
                    evaluated.append({**row, "engine_cp": engine_cp})
            metrics = summarize(evaluated)
            candidate = {"nnue_weight": weight, "nnue_clamp": clamp, **metrics}
            candidates.append(candidate)
            print(
                f"weight={weight} clamp={clamp} rmse={metrics['rmse_cp']:.2f} "
                f"mae={metrics['mae_cp']:.2f}",
                flush=True,
            )

    ranked = sorted(candidates, key=lambda item: (item["rmse_cp"], item["mae_cp"]))
    report = {
        "engine": str(engine),
        "model": str(model),
        "data": str(data),
        "positions": len(chosen),
        "seed": args.seed,
        "selection_metric": "static_teacher_rmse_only; final promotion still requires A/B",
        "best": ranked[0],
        "candidates": ranked,
    }
    output = Path(args.json_out).resolve()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"JSON {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

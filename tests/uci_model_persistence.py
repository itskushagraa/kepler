#!/usr/bin/env python3
"""Regression test: a custom EvalFile must survive ucinewgame."""

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: uci_model_persistence.py ENGINE NNUE_TOOL OUTPUT_MODEL BASELINE_MODEL"
        )
    engine, nnue_tool, output_model, baseline_model = map(Path, sys.argv[1:])
    subprocess.run([str(nnue_tool), str(output_model)], check=True, capture_output=True, text=True)
    commands = "\n".join(
        (
            "setoption name NNUEWeight value 100",
            "setoption name NNUEClamp value 0",
            f"setoption name EvalFile value {output_model}",
            "position fen 4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1",
            "eval",
            "setoption name UseBaseline value false",
            "ucinewgame",
            "position fen 4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1",
            "eval",
            f"setoption name BaselineEvalFile value {baseline_model}",
            "setoption name UseBaseline value true",
            "setoption name EvalFile value",
            "eval",
            "quit",
            "",
        )
    )
    completed = subprocess.run(
        [str(engine)], input=commands, check=True, capture_output=True, text=True, timeout=10
    )
    evaluations = re.findall(r"eval score (-?\d+) source (\w+)", completed.stdout)
    if len(evaluations) != 3 or evaluations[0] != evaluations[1]:
        print(completed.stdout, file=sys.stderr)
        return 1
    if any(source != "nnue" for _, source in evaluations):
        print(completed.stdout, file=sys.stderr)
        return 1
    print("PASS: custom EvalFile survived ucinewgame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Candidate:
    name: str
    result_weight: float
    cp_scale: float
    target_cp: float
    ridge: float


def run_cmd(cmd: List[str], timeout: int = 300) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    return p.returncode, p.stdout, p.stderr


def run_uci(engine: Path, commands: List[str], timeout: int = 300) -> str:
    payload = "\n".join(commands + ["quit"]) + "\n"
    p = subprocess.run(
        [str(engine)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Engine command failed: {p.stderr.strip()}")
    return p.stdout


def parse_selfplay_summary(out: str) -> Dict[str, int]:
    m = re.search(
        r"selfplay done games (\d+) whitewins (\d+) blackwins (\d+) draws (\d+) .* written (\d+)",
        out,
    )
    if not m:
        raise RuntimeError("Could not parse selfplay summary from engine output.")
    return {
        "games": int(m.group(1)),
        "whitewins": int(m.group(2)),
        "blackwins": int(m.group(3)),
        "draws": int(m.group(4)),
        "written": int(m.group(5)),
    }


def parse_bench_total(out: str) -> Dict[str, int]:
    m = re.search(r"bench total nodes (\d+) time (\d+) nps (\d+)", out)
    if not m:
        raise RuntimeError("Could not parse bench total line.")
    return {"nodes": int(m.group(1)), "time_ms": int(m.group(2)), "nps": int(m.group(3))}


def parse_last_info_stats(out: str) -> Dict[str, int]:
    matches = re.findall(r"info depth \d+ score cp -?\d+ nodes (\d+) nps (\d+) time (\d+)", out)
    if not matches:
        raise RuntimeError("Could not parse info nodes/nps/time from probe search output.")
    n, nps, t = matches[-1]
    return {"nodes": int(n), "time_ms": int(t), "nps": int(nps)}


def parse_probe_stats(out: str) -> Dict[str, int]:
    m = re.search(r"probe depth (\d+) scorecp (-?\d+) nodes (\d+) time (\d+) nps (\d+)", out)
    if not m:
        raise RuntimeError("Could not parse probe output line.")
    return {
        "depth": int(m.group(1)),
        "score_cp": int(m.group(2)),
        "nodes": int(m.group(3)),
        "time_ms": int(m.group(4)),
        "nps": int(m.group(5)),
    }


def parse_eval_line(out: str) -> Tuple[int, str]:
    m = re.search(r"eval score (-?\d+) source (\w+)", out)
    if not m:
        raise RuntimeError("Could not parse eval output line.")
    return int(m.group(1)), m.group(2)


def train_candidate(
    trainer: Path,
    data_path: Path,
    model_path: Path,
    c: Candidate,
    timeout: int,
) -> Dict[str, str]:
    cmd = [
        sys.executable,
        str(trainer),
        "--data",
        str(data_path),
        "--out",
        str(model_path),
        "--result-weight",
        str(c.result_weight),
        "--cp-scale",
        str(c.cp_scale),
        "--target-cp",
        str(c.target_cp),
        "--ridge",
        str(c.ridge),
    ]
    code, out, err = run_cmd(cmd, timeout=timeout)
    if code != 0:
        raise RuntimeError(f"Trainer failed for {c.name}: {err.strip() or out.strip()}")
    metrics = {"raw_out": out.strip()}
    m_samples = re.search(r"samples (\d+)", out)
    m_rmse = re.search(r"train_rmse ([0-9.]+)", out)
    m_units = re.search(r"trained_units \[([^\]]+)\]", out)
    if m_samples:
        metrics["samples"] = m_samples.group(1)
    if m_rmse:
        metrics["train_rmse"] = m_rmse.group(1)
    if m_units:
        metrics["trained_units"] = "[" + m_units.group(1) + "]"
    return metrics


def parse_candidates(specs: List[str]) -> List[Candidate]:
    out: List[Candidate] = []
    for s in specs:
        # name,result_weight,cp_scale,target_cp,ridge
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Invalid candidate spec '{s}'. Expected 5 comma-separated fields.")
        out.append(
            Candidate(
                name=parts[0],
                result_weight=float(parts[1]),
                cp_scale=float(parts[2]),
                target_cp=float(parts[3]),
                ridge=float(parts[4]),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run reproducible Kepler NNUE datagen->train->benchmark cycles.")
    ap.add_argument("--engine", default="build/kepler", help="Path to kepler engine binary.")
    ap.add_argument("--trainer", default="tools/train_bootstrap_nnue.py", help="Path to bootstrap trainer script.")
    ap.add_argument("--workdir", default="/tmp/kepler_cycle", help="Output directory for datasets/models/results.")
    ap.add_argument("--data", default="", help="Optional existing TSV dataset path. If set, datagen is skipped.")
    ap.add_argument("--bench-depth", type=int, default=6, help="Bench depth for candidate evaluation.")
    ap.add_argument("--probe-movetime", type=int, default=400, help="Fallback probe movetime (ms) if bench times out.")
    ap.add_argument("--uci-timeout", type=int, default=300, help="Default timeout (seconds) for UCI command runs.")
    ap.add_argument("--bench-timeout", type=int, default=600, help="Timeout (seconds) for bench command runs.")
    ap.add_argument("--games", type=int, default=200, help="Selfplay games.")
    ap.add_argument("--movetime", type=int, default=20, help="Selfplay movetime (ms).")
    ap.add_argument("--maxply", type=int, default=260, help="Selfplay max plies.")
    ap.add_argument("--randomplies", type=int, default=8, help="Random opening plies.")
    ap.add_argument("--sampleevery", type=int, default=2, help="Write one sample every N plies.")
    ap.add_argument("--minsampleply", type=int, default=8, help="Do not sample before this ply.")
    ap.add_argument("--seed", type=int, default=42, help="Selfplay RNG seed.")
    ap.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate spec: name,result_weight,cp_scale,target_cp,ridge (repeatable).",
    )
    args = ap.parse_args()

    engine = Path(args.engine).resolve()
    trainer = Path(args.trainer).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not engine.exists():
        raise SystemExit(f"Engine not found: {engine}")
    if not trainer.exists():
        raise SystemExit(f"Trainer not found: {trainer}")

    candidates = parse_candidates(args.candidate)
    data_path = Path(args.data).resolve() if args.data else (workdir / f"data_{run_id}.tsv")
    summary_path = workdir / f"summary_{run_id}.json"
    csv_path = workdir / f"results_{run_id}.csv"

    if args.data:
        if not data_path.exists():
            raise SystemExit(f"Provided dataset not found: {data_path}")
        sp = {"games": 0, "whitewins": 0, "blackwins": 0, "draws": 0, "written": 0}
        print(f"[cycle] using existing dataset -> {data_path}")
    else:
        selfplay_cmd = (
            "selfplay "
            f"games {args.games} movetime {args.movetime} maxply {args.maxply} "
            f"randomplies {args.randomplies} sampleevery {args.sampleevery} "
            f"minsampleply {args.minsampleply} seed {args.seed} outfile {data_path}"
        )
        print(f"[cycle] datagen -> {data_path}")
        out_selfplay = run_uci(engine, ["uci", selfplay_cmd], timeout=max(args.uci_timeout, args.games * args.movetime // 2 + 60))
        sp = parse_selfplay_summary(out_selfplay)
        print(f"[cycle] selfplay done: games={sp['games']} written={sp['written']} draws={sp['draws']}")

    rows: List[Dict[str, object]] = []
    for c in candidates:
        model_path = workdir / f"{run_id}_{c.name}.nnue"
        print(f"[cycle] train {c.name} -> {model_path.name}")
        train_metrics = train_candidate(trainer, data_path, model_path, c, timeout=300)

        bench_mode = "bench"
        try:
            bench_out = run_uci(
                engine,
                ["uci", f"setoption name EvalFile value {model_path}", f"bench {args.bench_depth}"],
                timeout=args.bench_timeout,
            )
            bench = parse_bench_total(bench_out)
        except Exception:
            bench_mode = "probe"
            probe_out = run_uci(
                engine,
                [
                    "uci",
                    f"setoption name EvalFile value {model_path}",
                    "position startpos",
                    f"probe depth {max(1, args.bench_depth)} movetime {max(0, args.probe_movetime)}",
                ],
                timeout=max(args.uci_timeout, 60),
            )
            probe = parse_probe_stats(probe_out)
            bench = {"nodes": probe["nodes"], "time_ms": probe["time_ms"], "nps": probe["nps"]}

        eval_out = run_uci(
            engine,
            [
                "uci",
                f"setoption name EvalFile value {model_path}",
                "position startpos moves e2e4 d7d5 e4d5",
                "eval",
            ],
            timeout=args.uci_timeout,
        )
        eval_score, eval_source = parse_eval_line(eval_out)

        row: Dict[str, object] = {
            "name": c.name,
            "model": str(model_path),
            "result_weight": c.result_weight,
            "cp_scale": c.cp_scale,
            "target_cp": c.target_cp,
            "ridge": c.ridge,
            "bench_nodes": bench["nodes"],
            "bench_time_ms": bench["time_ms"],
            "bench_nps": bench["nps"],
            "bench_mode": bench_mode,
            "eval_probe_score": eval_score,
            "eval_source": eval_source,
            "train_samples": train_metrics.get("samples"),
            "train_rmse": train_metrics.get("train_rmse"),
            "trained_units": train_metrics.get("trained_units"),
        }
        rows.append(row)
        print(
            f"[cycle] {c.name}: {bench_mode}_nps={bench['nps']} eval_probe={eval_score} "
            f"train_rmse={train_metrics.get('train_rmse')}"
        )

    rows_sorted = sorted(rows, key=lambda r: int(r["bench_nps"]), reverse=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    summary = {
        "run_id": run_id,
        "engine": str(engine),
        "trainer": str(trainer),
        "workdir": str(workdir),
        "selfplay": {
            "games": args.games,
            "movetime": args.movetime,
            "maxply": args.maxply,
            "randomplies": args.randomplies,
            "sampleevery": args.sampleevery,
            "minsampleply": args.minsampleply,
            "seed": args.seed,
            "written": sp["written"],
            "data_path": str(data_path),
        },
        "results_csv": str(csv_path),
        "results": rows_sorted,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[cycle] summary -> {summary_path}")
    print(f"[cycle] results -> {csv_path}")
    print("[cycle] top candidates by bench_nps:")
    for r in rows_sorted[:3]:
        print(f"  - {r['name']}: nps={r['bench_nps']} eval_probe={r['eval_probe_score']} rmse={r['train_rmse']}")


if __name__ == "__main__":
    main()

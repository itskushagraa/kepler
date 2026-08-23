#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine


@dataclass
class Candidate:
    name: str
    result_weight: float
    cp_scale: float
    target_cp: float
    ridge: float


OPENINGS: List[List[str]] = [
    [],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["c2c4", "e7e5", "b1c3", "g8f6"],
    ["g1f3", "d7d5", "d2d4", "g8f6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
]


def apply_opening(board: chess.Board, opening: List[str]) -> bool:
    for mv in opening:
        move = chess.Move.from_uci(mv)
        if move not in board.legal_moves:
            return False
        board.push(move)
    return True


def white_result_from_board(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def is_tactical_position(board: chess.Board) -> bool:
    """Identify forcing positions worth sampling more densely.

    Checks, promotions, captures, and available checking moves are included.
    This is intentionally a broad data-generation classifier; Stockfish still
    supplies the target and the trainer applies the tactical sample weight.
    """
    if board.is_check():
        return True
    for move in list(board.legal_moves):
        if move.promotion is not None or board.is_capture(move):
            return True
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        if gives_check:
            return True
    return False


def ply_from_fen(fen: str) -> Optional[int]:
    parts = fen.split()
    if len(parts) < 6:
        return None
    stm = parts[1]
    try:
        fullmove = int(parts[5])
    except ValueError:
        return None
    if fullmove < 1:
        return None
    return (fullmove - 1) * 2 + (0 if stm == "w" else 1)


def score_bucket(score_cp_stm: int) -> str:
    if score_cp_stm <= -300:
        return "le_-300"
    if score_cp_stm <= -151:
        return "-300_to_-151"
    if score_cp_stm <= -51:
        return "-150_to_-51"
    if score_cp_stm <= 50:
        return "-50_to_50"
    if score_cp_stm <= 150:
        return "51_to_150"
    if score_cp_stm <= 300:
        return "151_to_300"
    return "ge_301"


def classify_sample_row(
    fen: str,
    score_cp_stm: int,
    min_ply: int,
    max_ply: int,
    max_abs_score_cp: int,
) -> Tuple[bool, str, Optional[int]]:
    if max_abs_score_cp >= 0 and abs(score_cp_stm) > max_abs_score_cp:
        return False, "score_out_of_range", None
    ply = ply_from_fen(fen)
    if ply is None:
        return False, "bad_fen_ply", None
    if ply < min_ply:
        return False, "ply_too_low", ply
    if max_ply >= min_ply and ply > max_ply:
        return False, "ply_too_high", ply
    return True, "kept", ply


def filter_dataset_file(
    src_path: Path,
    dst_path: Path,
    min_ply: int,
    max_ply: int,
    max_abs_score_cp: int,
) -> Dict[str, int]:
    kept = 0
    total = 0
    filtered_by_reason: Dict[str, int] = {}
    kept_by_score_bucket: Dict[str, int] = {}
    kept_by_opening_id: Dict[str, int] = {}
    kept_by_termination: Dict[str, int] = {}
    kept_by_sample_kind: Dict[str, int] = {}

    def bump(hist: Dict[str, int], key: str) -> None:
        hist[key] = hist.get(key, 0) + 1

    with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
        for line in src:
            row = line.strip()
            if not row:
                continue
            parts = row.split("\t")
            if len(parts) < 3:
                bump(filtered_by_reason, "bad_format")
                continue
            total += 1
            try:
                score_cp_stm = int(parts[1])
            except ValueError:
                bump(filtered_by_reason, "bad_score")
                continue
            fen = parts[2]
            keep, reason, _ = classify_sample_row(fen, score_cp_stm, min_ply, max_ply, max_abs_score_cp)
            if not keep:
                bump(filtered_by_reason, reason)
                continue

            opening_id = parts[5] if len(parts) > 5 and parts[5] else "unknown"
            termination_reason = parts[6] if len(parts) > 6 and parts[6] else "unknown"
            sample_kind = parts[8] if len(parts) > 8 and parts[8] else "regular"
            bump(kept_by_score_bucket, score_bucket(score_cp_stm))
            bump(kept_by_opening_id, opening_id)
            bump(kept_by_termination, termination_reason)
            bump(kept_by_sample_kind, sample_kind)

            dst.write(row + "\n")
            kept += 1

    skipped = sum(filtered_by_reason.values())
    return {
        "total": total,
        "kept": kept,
        "skipped": skipped,
        "filtered_by_reason": filtered_by_reason,
        "kept_by_score_bucket": kept_by_score_bucket,
        "kept_by_opening_id": kept_by_opening_id,
        "kept_by_termination": kept_by_termination,
        "kept_by_sample_kind": kept_by_sample_kind,
    }


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


def generate_teacher_data(
    engine: Path,
    stockfish_path: Path,
    out_path: Path,
    games: int,
    movetime_ms: int,
    maxply: int,
    randomplies: int,
    sampleevery: int,
    tactical_sampleevery: int,
    minsampleply: int,
    seed: int,
    min_ply: int,
    max_ply: int,
    max_abs_score_cp: int,
    play_mode: str,
    play_elo: int,
    label_elo: int,
    analyze_ms: int,
    kepler_hash: int,
    kepler_threads: int,
    sf_hash: int,
    sf_threads: int,
) -> Dict[str, int]:
    play_limit = chess.engine.Limit(time=max(0.001, movetime_ms / 1000.0))
    analyze_limit = chess.engine.Limit(time=max(0.001, analyze_ms / 1000.0))
    rng = random.Random(seed)

    whitewins = 0
    blackwins = 0
    draws = 0
    written = 0
    considered = 0
    filtered_by_reason: Dict[str, int] = {}
    kept_by_score_bucket: Dict[str, int] = {}
    kept_by_opening_id: Dict[str, int] = {}
    kept_by_termination: Dict[str, int] = {}
    kept_by_sample_kind: Dict[str, int] = {}

    def bump(hist: Dict[str, int], key: str) -> None:
        hist[key] = hist.get(key, 0) + 1

    with open(out_path, "w", encoding="utf-8") as out:
        with chess.engine.SimpleEngine.popen_uci(str(engine)) as kepler, chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as sf_play, chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) as sf_label:
            kepler.configure({"Hash": max(1, kepler_hash), "Threads": max(1, kepler_threads)})

            sf_play_cfg: Dict[str, object] = {"Hash": max(1, sf_hash), "Threads": max(1, sf_threads)}
            if play_mode == "limited" and play_elo > 0:
                sf_play_cfg["UCI_LimitStrength"] = True
                sf_play_cfg["UCI_Elo"] = play_elo
            else:
                sf_play_cfg["UCI_LimitStrength"] = False
            sf_play.configure(sf_play_cfg)

            sf_label_cfg: Dict[str, object] = {"Hash": max(1, sf_hash), "Threads": max(1, sf_threads)}
            if label_elo > 0:
                sf_label_cfg["UCI_LimitStrength"] = True
                sf_label_cfg["UCI_Elo"] = label_elo
            else:
                sf_label_cfg["UCI_LimitStrength"] = False
            sf_label.configure(sf_label_cfg)

            for g in range(1, games + 1):
                board = chess.Board()
                opening_id = (g - 1) % len(OPENINGS)
                opening = OPENINGS[opening_id]
                if not apply_opening(board, opening):
                    continue

                for _ in range(randomplies):
                    if board.is_game_over(claim_draw=True):
                        break
                    legal = list(board.legal_moves)
                    if not legal:
                        break
                    board.push(rng.choice(legal))

                kepler_white = ((g - 1) % 2 == 0)
                samples: List[Tuple[str, bool, int, int, str, int, str]] = []

                while len(board.move_stack) < maxply and not board.is_game_over(claim_draw=True):
                    ply = len(board.move_stack)
                    regular_due = ply >= minsampleply and (ply % sampleevery) == 0
                    tactical_due = (
                        ply >= minsampleply
                        and tactical_sampleevery > 0
                        and (ply % tactical_sampleevery) == 0
                    )
                    tactical = is_tactical_position(board) if (regular_due or tactical_due) else False
                    if regular_due or (tactical_due and tactical):
                        info = sf_label.analyse(board, analyze_limit)
                        pov = info.get("score")
                        if pov is not None:
                            cp = pov.pov(board.turn).score(mate_score=20000)
                            if cp is not None:
                                fen = board.fen()
                                cp_stm = int(cp)
                                considered += 1
                                keep, reason, sample_ply = classify_sample_row(
                                    fen,
                                    cp_stm,
                                    min_ply,
                                    max_ply,
                                    max_abs_score_cp,
                                )
                                if keep and sample_ply is not None:
                                    samples.append(
                                        (
                                            fen,
                                            board.turn,
                                            cp_stm,
                                            sample_ply,
                                            score_bucket(cp_stm),
                                            opening_id,
                                            "tactical" if tactical else "regular",
                                        )
                                    )
                                else:
                                    bump(filtered_by_reason, reason)

                    mover = kepler if (board.turn == chess.WHITE) == kepler_white else sf_play
                    play = mover.play(board, play_limit)
                    if play.move is None or play.move not in board.legal_moves:
                        break
                    board.push(play.move)

                white_result = white_result_from_board(board)
                if white_result > 0:
                    whitewins += 1
                elif white_result < 0:
                    blackwins += 1
                else:
                    draws += 1

                outcome = board.outcome(claim_draw=True)
                termination_reason = "unterminated"
                if outcome is not None and outcome.termination is not None:
                    termination_reason = outcome.termination.name.lower()

                game_id = f"teacher-{seed}-{g}"
                for fen, side_is_white, cp_stm, sample_ply, bucket, sample_opening_id, sample_kind in samples:
                    stm_result = white_result if side_is_white else -white_result
                    out.write(
                        f"{stm_result}\t{cp_stm}\t{fen}\t{sample_ply}\t{bucket}\t{sample_opening_id}\t"
                        f"{termination_reason}\t{game_id}\t{sample_kind}\n"
                    )
                    bump(kept_by_score_bucket, bucket)
                    bump(kept_by_opening_id, str(sample_opening_id))
                    bump(kept_by_termination, termination_reason)
                    bump(kept_by_sample_kind, sample_kind)
                    written += 1

                print(
                    f"[cycle] teacher game {g}/{games} result "
                    f"{'1-0' if white_result > 0 else ('0-1' if white_result < 0 else '1/2-1/2')} "
                    f"samples={len(samples)}",
                    flush=True,
                )

    return {
        "games": games,
        "whitewins": whitewins,
        "blackwins": blackwins,
        "draws": draws,
        "written": written,
        "samples_considered": considered,
        "samples_filtered": sum(filtered_by_reason.values()),
        "filtered_by_reason": filtered_by_reason,
        "kept_by_score_bucket": kept_by_score_bucket,
        "kept_by_opening_id": kept_by_opening_id,
        "kept_by_termination": kept_by_termination,
        "kept_by_sample_kind": kept_by_sample_kind,
    }


def train_candidate(
    trainer: Path,
    data_path: Path,
    model_path: Path,
    c: Candidate,
    arch: str,
    hidden_size: int,
    epochs: int,
    lr: float,
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
        "--arch",
        str(arch),
        "--hidden-size",
        str(max(24, hidden_size)),
        "--epochs",
        str(max(1, epochs)),
        "--lr",
        str(max(1e-4, lr)),
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
    ap.add_argument("--stockfish", default="stockfish", help="Path to Stockfish binary for teacher mode.")
    ap.add_argument("--datagen-mode", choices=["selfplay", "teacher"], default="teacher", help="Dataset generation mode.")
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
    ap.add_argument(
        "--tactical-sampleevery",
        type=int,
        default=0,
        help="Also sample forcing positions every N plies; 0 disables tactical oversampling.",
    )
    ap.add_argument("--minsampleply", type=int, default=8, help="Do not sample before this ply.")
    ap.add_argument("--filter-min-ply", type=int, default=20, help="Keep samples with ply >= this value.")
    ap.add_argument("--filter-max-ply", type=int, default=60, help="Keep samples with ply <= this value.")
    ap.add_argument("--filter-max-abs-score-cp", type=int, default=200, help="Keep samples where |cp| <= this value. Set <0 to disable.")
    ap.add_argument("--teacher-play-mode", choices=["limited", "full"], default="limited", help="Teacher game mode; limited intentionally weakens Stockfish, full disables UCI_Elo limiting.")
    ap.add_argument("--teacher-play-elo", type=int, default=3000, help="Stockfish play Elo in limited teacher mode. Ignored in full mode.")
    ap.add_argument("--teacher-label-elo", type=int, default=0, help="Stockfish label Elo in teacher mode. Use 0 for max strength.")
    ap.add_argument("--teacher-analyze-ms", type=int, default=30, help="Label analysis time per sampled position (ms).")
    ap.add_argument("--kepler-hash", type=int, default=64, help="Kepler hash in teacher mode.")
    ap.add_argument("--kepler-threads", type=int, default=1, help="Kepler threads in teacher mode.")
    ap.add_argument("--teacher-hash", type=int, default=64, help="Stockfish hash in teacher mode.")
    ap.add_argument("--teacher-threads", type=int, default=1, help="Stockfish threads in teacher mode.")
    ap.add_argument("--trainer-arch", choices=["psqt", "halfkp"], default="halfkp", help="Trainer architecture.")
    ap.add_argument("--trainer-hidden-size", type=int, default=1536, help="Trainer hidden size.")
    ap.add_argument("--trainer-epochs", type=int, default=80, help="Trainer epochs.")
    ap.add_argument("--trainer-lr", type=float, default=0.08, help="Trainer learning rate.")
    ap.add_argument("--seed", type=int, default=42, help="Selfplay RNG seed.")
    ap.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate/filter data and its summary without invoking a trainer.",
    )
    ap.add_argument(
        "--candidate",
        action="append",
        required=False,
        help="Candidate spec: name,result_weight,cp_scale,target_cp,ridge (repeatable).",
    )
    args = ap.parse_args()

    engine = Path(args.engine).resolve()
    trainer = Path(args.trainer).resolve()
    stockfish_bin = shutil.which(args.stockfish)
    stockfish = Path(stockfish_bin).resolve() if stockfish_bin else Path(args.stockfish).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not engine.exists():
        raise SystemExit(f"Engine not found: {engine}")
    if not args.generate_only and not trainer.exists():
        raise SystemExit(f"Trainer not found: {trainer}")
    if not args.data and args.datagen_mode == "teacher" and not stockfish.exists():
        raise SystemExit(f"Stockfish not found: {args.stockfish}")

    if not args.generate_only and not args.candidate:
        raise SystemExit("At least one --candidate is required unless --generate-only is used.")
    candidates = parse_candidates(args.candidate or [])
    data_path = Path(args.data).resolve() if args.data else (workdir / f"data_{run_id}.tsv")
    summary_path = workdir / f"summary_{run_id}.json"
    csv_path = workdir / f"results_{run_id}.csv"

    if args.data:
        if not data_path.exists():
            raise SystemExit(f"Provided dataset not found: {data_path}")
        sp = {
            "games": 0,
            "whitewins": 0,
            "blackwins": 0,
            "draws": 0,
            "written": 0,
            "samples_considered": 0,
            "samples_filtered": 0,
            "filtered_by_reason": {},
            "kept_by_score_bucket": {},
            "kept_by_opening_id": {},
            "kept_by_termination": {},
        }
        print(f"[cycle] using existing dataset -> {data_path}")
    else:
        print(f"[cycle] datagen ({args.datagen_mode}) -> {data_path}")
        if args.datagen_mode == "teacher":
            sp = generate_teacher_data(
                engine=engine,
                stockfish_path=stockfish,
                out_path=data_path,
                games=max(1, args.games),
                movetime_ms=max(1, args.movetime),
                maxply=max(20, args.maxply),
                randomplies=max(0, args.randomplies),
                sampleevery=max(1, args.sampleevery),
                tactical_sampleevery=max(0, args.tactical_sampleevery),
                minsampleply=max(0, args.minsampleply),
                seed=args.seed,
                min_ply=max(0, args.filter_min_ply),
                max_ply=args.filter_max_ply,
                max_abs_score_cp=args.filter_max_abs_score_cp,
                play_mode=args.teacher_play_mode,
                play_elo=args.teacher_play_elo,
                label_elo=args.teacher_label_elo,
                analyze_ms=max(1, args.teacher_analyze_ms),
                kepler_hash=max(1, args.kepler_hash),
                kepler_threads=max(1, args.kepler_threads),
                sf_hash=max(1, args.teacher_hash),
                sf_threads=max(1, args.teacher_threads),
            )
            print(
                f"[cycle] teacher done: games={sp['games']} written={sp['written']} "
                f"draws={sp['draws']} filtered={sp['samples_filtered']}"
            )
        else:
            selfplay_cmd = (
                "selfplay "
                f"games {args.games} movetime {args.movetime} maxply {args.maxply} "
                f"randomplies {args.randomplies} sampleevery {args.sampleevery} "
                f"minsampleply {args.minsampleply} seed {args.seed} outfile {data_path}"
            )
            out_selfplay = run_uci(engine, ["uci", selfplay_cmd], timeout=max(args.uci_timeout, args.games * args.movetime // 2 + 60))
            sp = parse_selfplay_summary(out_selfplay)
            filter_stats = filter_dataset_file(
                src_path=data_path,
                dst_path=data_path.with_suffix(".filtered.tsv"),
                min_ply=max(0, args.filter_min_ply),
                max_ply=args.filter_max_ply,
                max_abs_score_cp=args.filter_max_abs_score_cp,
            )
            os.replace(data_path.with_suffix(".filtered.tsv"), data_path)
            sp["samples_considered"] = filter_stats["total"]
            sp["samples_filtered"] = filter_stats["skipped"]
            sp["written"] = filter_stats["kept"]
            sp["filtered_by_reason"] = filter_stats.get("filtered_by_reason", {})
            sp["kept_by_score_bucket"] = filter_stats.get("kept_by_score_bucket", {})
            sp["kept_by_opening_id"] = filter_stats.get("kept_by_opening_id", {})
            sp["kept_by_termination"] = filter_stats.get("kept_by_termination", {})
            print(
                f"[cycle] selfplay done: games={sp['games']} written={sp['written']} "
                f"draws={sp['draws']} filtered={sp['samples_filtered']}"
            )

    if not data_path.exists() or data_path.stat().st_size == 0:
        raise SystemExit(
            "Dataset is empty after filtering. Increase games or relax "
            "--filter-min-ply/--filter-max-ply/--filter-max-abs-score-cp."
        )

    if args.generate_only:
        generation_summary = {
            "run_id": run_id,
            "engine": str(engine),
            "stockfish": str(stockfish),
            "data_path": str(data_path),
            "games": sp.get("games", 0),
            "written": sp.get("written", 0),
            "samples_considered": sp.get("samples_considered", 0),
            "samples_filtered": sp.get("samples_filtered", 0),
            "kept_by_score_bucket": sp.get("kept_by_score_bucket", {}),
            "kept_by_sample_kind": sp.get("kept_by_sample_kind", {}),
            "seed": args.seed,
        }
        summary_path.write_text(json.dumps(generation_summary, indent=2) + "\n", encoding="utf-8")
        print(f"[cycle] generation summary -> {summary_path}")
        return

    rows: List[Dict[str, object]] = []
    for c in candidates:
        model_path = workdir / f"{run_id}_{c.name}.nnue"
        print(f"[cycle] train {c.name} -> {model_path.name}")
        train_metrics = train_candidate(
            trainer=trainer,
            data_path=data_path,
            model_path=model_path,
            c=c,
            arch=args.trainer_arch,
            hidden_size=args.trainer_hidden_size,
            epochs=args.trainer_epochs,
            lr=args.trainer_lr,
            timeout=300,
        )

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
            "trainer_arch": args.trainer_arch,
            "trainer_hidden_size": args.trainer_hidden_size,
            "trainer_epochs": args.trainer_epochs,
            "trainer_lr": args.trainer_lr,
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
        "stockfish": str(stockfish),
        "workdir": str(workdir),
        "trainer_config": {
            "arch": args.trainer_arch,
            "hidden_size": args.trainer_hidden_size,
            "epochs": args.trainer_epochs,
            "lr": args.trainer_lr,
        },
        "selfplay": {
            "datagen_mode": args.datagen_mode,
            "games": args.games,
            "movetime": args.movetime,
            "maxply": args.maxply,
            "randomplies": args.randomplies,
            "sampleevery": args.sampleevery,
            "tactical_sampleevery": args.tactical_sampleevery,
            "minsampleply": args.minsampleply,
            "filter_min_ply": args.filter_min_ply,
            "filter_max_ply": args.filter_max_ply,
            "filter_max_abs_score_cp": args.filter_max_abs_score_cp,
            "teacher_play_mode": args.teacher_play_mode,
            "teacher_play_elo": args.teacher_play_elo,
            "teacher_label_elo": args.teacher_label_elo,
            "teacher_analyze_ms": args.teacher_analyze_ms,
            "kepler_hash": args.kepler_hash,
            "kepler_threads": args.kepler_threads,
            "teacher_hash": args.teacher_hash,
            "teacher_threads": args.teacher_threads,
            "seed": args.seed,
            "written": sp["written"],
            "samples_considered": sp.get("samples_considered", 0),
            "samples_filtered": sp.get("samples_filtered", 0),
            "filtered_by_reason": sp.get("filtered_by_reason", {}),
            "kept_by_score_bucket": sp.get("kept_by_score_bucket", {}),
            "kept_by_opening_id": sp.get("kept_by_opening_id", {}),
            "kept_by_termination": sp.get("kept_by_termination", {}),
            "kept_by_sample_kind": sp.get("kept_by_sample_kind", {}),
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

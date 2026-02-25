# Kepler: Chess Engine

<p align="left">
  <img alt="Language" src="https://img.shields.io/badge/C%2B%2B%20%2F%20Python-Engine%20%2B%20ML-informational?style=for-the-badge">
  <img alt="Domain" src="https://img.shields.io/badge/Domain-Chess%20AI-blue?style=for-the-badge">
  <img alt="Eval" src="https://img.shields.io/badge/Eval-NNUE-orange?style=for-the-badge">
  <img alt="Search" src="https://img.shields.io/badge/Search-AlphaBeta%20%2B%20Heuristics-8A2BE2?style=for-the-badge">
</p>

> Kepler is a chess engine I’m building because I like chess and I love ML + algorithms.  
> It’s part learning project, part “how strong can I make this thing?”

---

## Why Kepler?

I wanted something I could understand end‑to‑end. So I built it from scratch, one subsystem at a time.  
The goal is to keep it **measurable**, **hackable**, and easy to iterate on.

---

## Highlights

- NNUE evaluation (HalfKP) with fast incremental updates
- Search improvements (move ordering, LMR/QS tweaks, heuristics)
- Teacher‑data workflow with filtering + curation
- Gauntlet harness for controlled Elo estimation

---

## Quick Start

### Build (CMake)

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
```

### Run (UCI)

```bash
./build-release/kepler
```

Common UCI commands:

```text
uci
isready
ucinewgame
setoption name EvalFile value models/kepler_baseline_pst_v1.nnue
position startpos
go movetime 1000
```

---

## NNUE Workflow (End-to-End)

### 1) Generate Teacher Data

```bash
python3 tools/nnue_cycle.py \
  --engine build-release/kepler \
  --stockfish stockfish \
  --games 200 \
  --movetime 20 \
  --datagen-mode teacher \
  --teacher-play-elo 3000 \
  --teacher-label-elo 0 \
  --filter-min-ply 20 \
  --filter-max-ply 60 \
  --filter-max-abs-score-cp 200
```

### 2) Train NNUE

```bash
python3 tools/train_bootstrap_nnue.py \
  --data /tmp/kepler_data.tsv \
  --out /tmp/kepler_bootstrap.nnue \
  --arch halfkp \
  --hidden-size 1536 \
  --result-weight 0.4 \
  --cp-scale 300 \
  --target-cp 100 \
  --ridge 0.5 \
  --epochs 80
```

### 3) Evaluate in Gauntlet

```bash
python3 tools/rated_gauntlet.py \
  --engine build-release/kepler \
  --stockfish stockfish \
  --baseline-model models/kepler_baseline_pst_v1.nnue \
  --eval-model /tmp/kepler_bootstrap.nnue \
  --max-plies 0 \
  --levels 2000,2200,2400 \
  --games-per-level 100
```

---

## Dataset Format

TSV rows are:

```
result<TAB>score<TAB>fen<TAB>...metadata
```

Metadata columns (when enabled) include ply, score bucket, opening ID, and termination reason.  
Training ignores extra columns after FEN.

---

---

## Project Structure

```
src/        Engine core (search, eval, UCI, NNUE runtime)
include/    Headers
tools/      Data generation, training, curation, gauntlets
models/     Baseline NNUE files
tests/      Perft + engine validation
```

---

## Docs & Usage

Key scripts:
- `tools/nnue_cycle.py` - teacher data generation + training loop
- `tools/train_bootstrap_nnue.py` - NNUE trainer
- `tools/rated_gauntlet.py` - Elo benchmarking harness
- `tools/curate_dataset.py` - dataset curation + split generation
- `tools/train_nnue_pipeline.py` - training with validation tracking

---

## Motivation

I like playing chess, and I love ML + algorithms, so I decided to build a chess engine from scratch and see how far I could push it.

## Current Strength (WIP)

Right now Kepler sits around the **~2000 Elo** range in my internal gauntlets (e.g., estimated ~2038 in controlled Stockfish‑based testing).  
It’s always evolving as I train new nets and tweak search, but I’m limited by **one GPU** so training throughput isn’t infinite.

---

## Disclaimer

This is a personal research project, so expect rough edges. I’m iterating fast and breaking things often.

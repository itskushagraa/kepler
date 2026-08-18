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

I wanted something I could understand end‑to‑end. So I built it from scratch.
The goal is an estimated playing strength of 3000+ elo, although I'm currently limited to 1 GPU, which restricts the scale of training, dataset size, and architectural experiments I can run.

---

## Quick Start

### Build (CMake)

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
```

Run the move-generation regression suite:

```bash
./build-release/perft
```

It contains the six standard ChessProgramming.org positions and prints depths
1–5 for each. The target is standalone and does not depend on search or UCI.

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

Syzygy probing is optional. Place Fathom’s `tbprobe.h` and `tbprobe.c` under
`external/fathom/`, then configure with `-DKEPLER_SYZYGY=ON`.

---

## NNUE Workflow

### 1) Generate Teacher Data

```bash
python3 tools/nnue_cycle.py \
  --engine build-release/kepler \
  --stockfish stockfish \
  --games 200 \
  --movetime 20 \
  --datagen-mode teacher \
  --teacher-play-elo 3190 \
  --teacher-label-elo 0 \
  --filter-min-ply 20 \
  --filter-max-ply 60 \
  --filter-max-abs-score-cp 200
```

### 2) Train NNUE (this will take a LONG time depending on what params you set for the step above)

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

### 3) Evaluate in Gauntlet (against stockfish)

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

## Project Structure

```
src/        Engine core and headers (move generation, search, eval, UCI, NNUE)
tools/      Data generation, training, curation, gauntlets
models/     Baseline NNUE files
tests/      Move-generation/perft regression suite
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

I like playing chess; I love Machine Learning and algorithm analysis/design, so I decided to build a chess engine from scratch and see how far I could push it.

## Development Status

Kepler is usable for UCI analysis and self-play, with legal move generation,
alpha-beta search, classical evaluation, optional NNUE evaluation, and a NNUE
data/training workflow. Historical internal gauntlets reported roughly
1800–2000 Elo; that is not a current benchmark and should be refreshed.

The active development areas are search strength, evaluation tuning,
training-data quality, and optional Syzygy support. Perft is the primary
correctness guardrail for position and move-generation changes.

---

## Disclaimer

This is a personal research project. Keep generated builds and training output
outside version control, and run perft after position or move-generation edits.
Using this on sites like lichess and chess.com can result in a ban and is
discouraged.

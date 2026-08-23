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

Run the regression checks:

```bash
ctest --test-dir build-release --output-on-failure
```

This covers the perft smoke suite and verifies that fixed node budgets remain
global and exact with 1, 2, 4, and 8 search threads.

Run the fast move-generation smoke suite:

```bash
./build-release/perft
```

The default runs all six standard ChessProgramming.org positions through depth
3, validates every node count, prints progress immediately, and returns a
nonzero status on failure. Run the complete depth 1–5 suite with:

```bash
./build-release/perft --full
```

Other useful forms are `./build-release/perft --depth 4` and
`./build-release/perft --depth 4 --threads 4`. The positional form
`perft 5 4` remains supported for older local workflows. The target is
standalone and does not depend on search or UCI.

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
  --generate-only \
  --engine build-release/kepler \
  --stockfish stockfish \
  --workdir /tmp/kepler_cycle \
  --games 300 \
  --movetime 12 \
  --datagen-mode teacher \
  --sampleevery 3 \
  --tactical-sampleevery 2 \
  --teacher-play-mode limited \
  --teacher-play-elo 2850 \
  --teacher-label-elo 0 \
  --teacher-analyze-ms 15 \
  --filter-min-ply 12 \
  --filter-max-ply 120 \
  --filter-max-abs-score-cp 800
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

The checked-in `models/kepler_baseline_pst_v1.nnue` is a small bootstrap PSQT
model. It is useful as a stable fallback, but it is not a full-strength
nonlinear HalfKP network. The legacy `train_bootstrap_nnue.py --arch halfkp`
path is retained for reproducibility, but it is a sparse linear bootstrap and
should not be confused with the nonlinear trainer below.

For the actual runtime-compatible nonlinear HalfKP candidate, train with
NumPy:

```bash
python3 tools/train_halfkp_nnue.py \
  --data /tmp/kepler_cycle/data_RUN_ID.tsv \
  --out /tmp/kepler_halfkp_candidate.nnue \
  --hidden-size 128 \
  --epochs 80 \
  --batch-size 128 \
  --result-weight 0.1 \
  --ridge 0.0003 \
  --validation-split 0.2 \
  --tactical-weight 1.25 \
  --patience 10
```

Teacher rows include a game ID and regular/tactical sample kind. The trainer
holds out complete games, stops when validation no longer improves, restores
the best epoch, and reports both floating-point and quantized RMSE in
`/tmp/kepler_halfkp_candidate.metrics.json`. Its two perspective accumulators
and ReLU placement match `src/nnue.cpp`. Validate the incremental path before
playing games:

```bash
./build-release/eval_consistency /tmp/kepler_halfkp_candidate.nnue
```

For comparison against the current baseline, use the paired sequential test.
It plays each opening twice with colors swapped and exits `0` only when the
candidate clears the configured improvement boundary, `1` when it is rejected,
and `2` when the maximum game budget is inconclusive:

```bash
python3 tools/nnue_ab_sprt.py \
  --engine build-release/kepler \
  --model-a models/kepler_baseline_pst_v1.nnue \
  --model-b /tmp/kepler_halfkp_candidate.nnue \
  --games 64 \
  --nodes 100000 \
  --threads 1 \
  --hash 128 \
  --elo0 0 \
  --elo1 30 \
  --draw-rate 0.35 \
  --pgn-out /tmp/kepler-halfkp-ab.pgn \
  --json-out /tmp/kepler-halfkp-ab.json
```

The SPRT result is a promotion gate, not a claim of universal Elo. A model
that is rejected or inconclusive remains an experiment; the checked-in
baseline is not replaced automatically. For an experimental dense PSQT
candidate using the same runtime file format, train with NumPy:

```bash
python3 tools/train_dense_nnue.py \
  --data /tmp/kepler_data.tsv \
  --out /tmp/kepler_dense_psqt.nnue \
  --hidden-size 64 \
  --epochs 80
```

Candidates must pass the paired A/B test before replacing the baseline.

The same paired runner can compare search binaries by adding `--engine-b` and
using the same model for A and B. For fixed-position diagnostics, use
`tools/search_quality.py` against a PGN before committing to a game match.

### 3) Benchmark against full-strength Stockfish

```bash
python3 tools/rated_gauntlet.py \
  --engine build-release/kepler \
  --stockfish stockfish \
  --baseline-model models/kepler_baseline_pst_v1.nnue \
  --eval-model /tmp/kepler_bootstrap.nnue \
  --nodes 100000 \
  --max-plies 400 \
  --games-per-opponent 100 \
  --audit-samples 32
```

This is a two-pass benchmark. The games first run against full-strength
references using fixed node budgets and paired openings/colors. After the PGN
is written, the optional quality pass replays the games and analyzes Kepler’s
moves with a separate full-strength Stockfish process. It does not influence
gameplay or the rating. Use `--audit-all` instead of `--audit-samples 32` when
you want every Kepler move analyzed; this is substantially slower. The JSON
report includes centipawn-loss, best/near-best move, blunder, mate-loss, WDL
loss, phase, and opponent breakdowns.

Full-strength mode never enables Stockfish’s `UCI_LimitStrength` or
`UCI_Elo` options. Without an explicit trusted rating anchor, it reports a
reproducible score but deliberately does not claim an absolute Elo. For a
calibrated rating, use the built-in reference-ladder builder. It runs a
full-strength round-robin, fits all engines jointly, and emits both the target
rating and a reference manifest. The example manifest is at
`references/engines.example.json`; replace the paths with the binaries you
actually intend to benchmark.

For a direct Kepler measurement, include Kepler in that manifest and run:

```bash
python3 tools/reference_ladder.py \
  --engines-config references/engines.json \
  --target-name Kepler \
  --anchor-name "Stockfish 18" \
  --anchor-rating 3651 \
  --games-per-pair 40 \
  --nodes 1000000 \
  --max-plies 400 \
  --hash 256 \
  --threads 1 \
  --out-json /tmp/kepler-reference-ladder.json \
  --out-pgn /tmp/kepler-reference-ladder.pgn \
  --out-reference-config /tmp/kepler-references.json
```

`3651` is an example external Stockfish 18 anchor on the CCRL 40/15 scale; it
is a scale anchor, not a claim that a fixed-node game has the same playing
conditions as CCRL. Keep the anchor, engine binaries, node budget, thread
counts, opening seed, and game count fixed. The result is usable only when
the JSON reports `calibration.status == "converged"`, the Kepler entry is not
`boundary`, and its confidence interval is acceptably narrow. Forty games per
pair is the production default; two games per pair is only a smoke test.

The generated JSON is the primary rating result. The generated reference
manifest can also be used for a separate Kepler-only run and move-quality
audit:

```json
[
  {"name": "Stockfish full", "path": "/path/to/stockfish", "rating": 3400},
  {"name": "Reference engine", "path": "/path/to/reference", "rating": 2500}
]
```

Run it with `--reference-config references.json`. The rating is fitted across
all games and includes a 95% confidence interval. The 3400 value above is an
example only; it must be replaced with the rating and protocol you actually
trust. Keep the binary, model hashes, node budget, thread counts, opening seed,
and reference ratings fixed when comparing runs.

For exploratory weakened-Stockfish matches, use the mode explicitly:

```bash
python3 tools/rated_gauntlet.py \
  --mode limited --levels 2000,2200,2400 \
  --games-per-opponent 100
```

Limited mode is labeled non-authoritative in the output.

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
- `tools/train_halfkp_nnue.py` - nonlinear HalfKP trainer
- `tools/nnue_ab_sprt.py` - paired A/B candidate promotion test
- `tools/search_quality.py` - fixed-node PGN move-quality audit
- `tools/rated_gauntlet.py` - full-strength/reference benchmark and exploratory gauntlet
- `tools/reference_ladder.py` - joint full-strength reference calibration
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

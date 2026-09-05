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

`NNUEWeight` controls the neural share from 0–100 (default 25).
`NNUEClamp` limits how far NNUE may differ from classical evaluation (default
300 centipawns); set it to 0 to disable the clamp. Loading a non-empty
`EvalFile` selects that model persistently across `ucinewgame`.

`Hash` is a strict memory ceiling: the power-of-two table is rounded down and
will never allocate more than the requested number of MiB. Search uses one
Lazy-SMP worker per configured thread and a shared lock-free transposition
table. Sliding rook, bishop, and queen attacks use magic-bitboard lookup tables
in both move generation and evaluation.

Clock searches distinguish an optimum and a hard budget. Kepler adjusts the
optimum after each completed iteration using best-move and score stability,
the root score gap, and aspiration re-searches. All Lazy-SMP workers share one
clock and are cancelled when the main worker is ready, so joining helpers
cannot delay `bestmove`. Sudden-death allocation keeps a conservative
50-move horizon and charges `MoveOverhead` per anticipated move; explicit
`go movetime`, fixed depth, and fixed node searches retain their fixed-limit
behavior.

The built-in deterministic-position benchmark makes NPS and thread scaling
easy to inspect without playing games:

```text
setoption name Threads value 1
bench 6
setoption name Threads value 2
bench 6
setoption name Threads value 4
bench 6
```

Repeat each configuration several times and compare the `bench total` lines;
wall-clock NPS is machine- and load-dependent. `bench 6 diff` also compares a
run with the preceding benchmark in the same engine session.

The current baseline remains at NNUE weight 25/clamp 300 until a replacement
network passes the documented static calibration and paired A/B gate. A
validated full-NNUE configuration at weight 100/clamp 0 now bypasses the
classical evaluator entirely, so that experiment measures both strength and
the available NPS gain instead of paying for both evaluation paths.

Syzygy probing is optional. Place Fathom’s `tbprobe.h` and `tbprobe.c` under
`external/fathom/`, then configure with `-DKEPLER_SYZYGY=ON`.

### Lichess BOT deployment

Kepler supports the standard-chess UCI flow used by the official
[`lichess-bot`](https://github.com/lichess-bot-devs/lichess-bot) bridge. The
deployment wrapper pins the tested bridge revision under ignored `.local/`
state and uses an exact dependency lock, keeps the OAuth token out of files
and command-line arguments, builds the Release engine, and runs an offline
bridge/clock smoke test:

```bash
python3 tools/lichess_bot.py setup
python3 tools/lichess_bot.py status
python3 tools/lichess_bot.py smoke
```

The committed live policy accepts one standard, casual, human real-time game
at a time, from 1+0 bullet through classical controls. The setup applies a
small tracked patch to the pinned bridge so its special first move scales down
to 600 ms in 1+0 instead of always consuming ten seconds. It disables
correspondence, pondering, variants, bot opponents, rated games, automatic
matchmaking, online move sources, automatic draws, and resigning. Review
`deploy/lichess/config.yml` before widening that policy.

The Lichess account must have never played a game. While signed into that
account, create a personal OAuth token with only the `bot:play` scope using
the [official token
instructions](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token).
Load it without writing it into shell history or the repository:

```bash
read -s LICHESS_BOT_TOKEN
export LICHESS_BOT_TOKEN
python3 tools/lichess_bot.py account
```

The read-only `account` command validates the `bot:play` scope and prints the
selected username, title, game count, and upgrade state without exposing the
token. Do not continue unless it reports the intended username and either
`eligible_zero_games` or `already_bot`.

Converting an account to BOT status is irreversible. After the account check
confirms zero games, run the one-time upgrade command:

```bash
python3 tools/lichess_bot.py upgrade --confirm-irreversible --verbose
```

The upgrade command starts the bot after conversion. Later sessions use
`python3 tools/lichess_bot.py run`; prefix it with `caffeinate -i` on macOS if
the machine might otherwise sleep. Press Ctrl-C once to stop accepting games
and finish active games. PGNs are written under `deploy/lichess/games/` and
bridge logs remain under ignored `.local/` state. Revoke the token from
Lichess immediately if it is ever exposed.

---

## Strength Development Workflow

The trusted production workflow is documented in
[`docs/strength-development.md`](docs/strength-development.md). It covers the
current measured diagnosis, resumable million-row teacher generation,
game-disjoint curation gates, matched H128/H256 training in centipawn units,
static blend calibration, mandatory A/A sanity, resumable paired
pentanomial A/B promotion, search-quality diagnosis, and final reference
rating.

The shortest production sequence is:

```bash
python3 tools/generate_teacher_shards.py \
  --out-dir /private/tmp/kepler-strength-data-v1
python3 tools/train_strength_models.py \
  --dataset-dir /private/tmp/kepler-strength-data-v1/curated \
  --out-dir /private/tmp/kepler-strength-models-v1
python3 tools/run_nnue_experiment.py \
  --baseline models/kepler_baseline_pst_v1.nnue \
  --candidate /private/tmp/kepler-strength-models-v1/halfkp_h128.nnue \
  --validation-data /private/tmp/kepler-strength-data-v1/curated/validation.tsv \
  --out-dir /private/tmp/kepler-strength-models-v1/h128-experiment
```

Use `caffeinate -i` for the long generation/training/playing stages on macOS.
Do not use smoke-test bypasses when producing a model for promotion.

## Low-scale NNUE Workflow

The manual commands below are useful for experiments and debugging. They do
not satisfy the production data-size and diversity gates by themselves.

### 1) Generate Teacher Data

```bash
python3 tools/nnue_cycle.py \
  --generate-only \
  --engine build-release/kepler \
  --stockfish stockfish \
  --workdir /tmp/kepler_cycle \
  --games 2000 \
  --movetime 8 \
  --datagen-mode teacher \
  --sampleevery 2 \
  --tactical-sampleevery 1 \
  --teacher-play-mode limited \
  --teacher-play-elo 2600 \
  --teacher-label-elo 0 \
  --teacher-analyze-nodes 10000 \
  --teacher-kepler-game-fraction 0.5 \
  --filter-min-ply 12 \
  --filter-max-ply 120 \
  --filter-max-abs-score-cp 800
```

`teacher-analyze-nodes` makes labels reproducible across machines. Half of the
example games contain Kepler and half are Stockfish self-play, preventing the
corpus from consisting almost entirely of already-lost Kepler positions. Use
different seeds and output directories to create independent shards.

### 2) Curate and split the data

```bash
python3 tools/curate_dataset.py \
  --input /tmp/kepler_cycle_a/data_RUN_A.tsv \
  --input /tmp/kepler_cycle_b/data_RUN_B.tsv \
  --out-dir /tmp/kepler_curated \
  --max-rows 100000 \
  --tactical-fraction 0.4 \
  --val-fraction 0.2 \
  --seed 42
```

The curator namespaces explicit game IDs across shards, removes duplicate
positions while retaining side/castling/en-passant state, balances phase,
material class, score band, and regular/tactical samples, and splits only on
complete games.
Inspect `curation_manifest.json`; `game_leakage` must be empty.

### 3) Train nonlinear HalfKP candidates

```bash
python3 tools/train_halfkp_nnue.py \
  --data /tmp/kepler_curated/train.tsv \
  --validation-data /tmp/kepler_curated/validation.tsv \
  --out /tmp/kepler_halfkp_h128.nnue \
  --hidden-size 128 \
  --epochs 80 \
  --batch-size 256 \
  --target-mode centipawn \
  --target-clip 2000 \
  --wdl-cp 600 \
  --result-weight 0.1 \
  --ridge 0.0003 \
  --tactical-weight 1.0 \
  --score-balance-power 0.5 \
  --patience 10
```

Repeat with `--hidden-size 256` for the larger candidate. The checked-in
`models/kepler_baseline_pst_v1.nnue` is a stable bootstrap PSQT fallback, not
a full-strength nonlinear HalfKP network. The trainer rejects overlapping
train/validation game IDs, stops when validation no longer improves, restores
the best epoch, and reports floating-point and quantized RMSE in the adjacent
metrics JSON. Its two perspective accumulators
and ReLU placement match `src/nnue.cpp`. Validate the incremental path before
playing games:

```bash
./build-release/eval_consistency /tmp/kepler_halfkp_h128.nnue
```

### 4) Select the model and blend by paired games

For comparison against the current baseline, use the paired sequential test.
It plays each opening twice with colors swapped and exits `0` only when the
candidate clears the configured improvement boundary, `1` when it is rejected,
and `2` when the maximum game budget is inconclusive:

```bash
python3 tools/nnue_ab_sprt.py \
  --engine build-release/kepler \
  --model-a models/kepler_baseline_pst_v1.nnue \
  --model-b /tmp/kepler_halfkp_candidate.nnue \
  --games 256 \
  --nodes 100000 \
  --threads 1 \
  --hash 128 \
  --elo0 0 \
  --elo1 20 \
  --draw-rate 0.8 \
  --pgn-out /tmp/kepler-halfkp-ab.pgn \
  --json-out /tmp/kepler-halfkp-ab.json
```

After selecting H128 or H256, sweep the blend with the same model on both
sides. This tests 50% NNUE against 25%, with the classical-anchor clamp
disabled for both:

```bash
python3 tools/nnue_ab_sprt.py \
  --engine build-release/kepler \
  --model-a /tmp/kepler_halfkp_h128.nnue \
  --model-b /tmp/kepler_halfkp_h128.nnue \
  --nnue-weight-a 25 --nnue-weight-b 50 \
  --nnue-clamp-a 0 --nnue-clamp-b 0 \
  --games 256 --nodes 100000 --draw-rate 0.8 \
  --elo0 0 --elo1 20 \
  --json-out /tmp/kepler-blend-25-vs-50.json
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

### 5) Benchmark against full-strength Stockfish

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

The current columns are result, Stockfish score from the side to move, FEN,
ply, score bucket, opening ID, termination, game ID, sample kind, phase, and
material class. Legacy trainers ignore metadata after FEN; the nonlinear
HalfKP trainer uses all available grouping and diagnostic metadata.

---

## Project Structure

```
src/        Engine core and headers (move generation, search, eval, UCI, NNUE)
tools/      Data generation, training, curation, gauntlets
models/     Baseline NNUE files
tests/      Move-generation/perft regression suite
deploy/     Secret-free deployment configuration (including Lichess BOT)
```

---

## Docs & Usage

Key scripts:
- `tools/generate_teacher_shards.py` - resumable parallel production data generation
- `tools/train_strength_models.py` - matched strength-gated H128/H256 training
- `tools/eval_quality.py` - held-out static evaluation diagnostics
- `tools/tune_eval_blend.py` - held-out NNUE weight/clamp calibration
- `tools/run_nnue_experiment.py` - static audit + A/A + paired A/B orchestration
- `tools/nnue_cycle.py` - teacher data generation + training loop
- `tools/train_bootstrap_nnue.py` - NNUE trainer
- `tools/train_halfkp_nnue.py` - nonlinear HalfKP trainer
- `tools/nnue_ab_sprt.py` - paired A/B candidate promotion test
- `tools/search_quality.py` - fixed-node PGN move-quality audit
- `tools/rated_gauntlet.py` - full-strength/reference benchmark and exploratory gauntlet
- `tools/reference_ladder.py` - joint full-strength reference calibration
- `tools/curate_dataset.py` - dataset curation + split generation
- `tools/train_nnue_pipeline.py` - training with validation tracking
- `tools/lichess_bot.py` - pinned Lichess bridge setup, smoke, upgrade, and run wrapper

---

## Motivation

I like playing chess; I love Machine Learning and algorithm analysis/design, so I decided to build a chess engine from scratch and see how far I could push it.

## Development Status

Kepler is usable for UCI analysis, Lichess BOT play, and self-play, with legal
move generation, alpha-beta search, classical evaluation, optional NNUE
evaluation, and a strength-gated NNUE workflow. Historical internal gauntlets
reported roughly 1800–2000 Elo; that is not a current benchmark and should not
be cited as the engine's present rating.

The August 2026 correctness pass repaired castling make/unmake corruption,
full-game threefold detection, insufficient-material adjudication, mate-score
UCI reporting, first-move over-pruning, deep-search stack exhaustion, and
multithread node/result ownership. The old H256 result predates those repairs
and is not promotion evidence.

The active development areas are search strength, evaluation tuning,
training-data quality, and optional Syzygy support. Perft is the primary
correctness guardrail for position and move-generation changes.

---

## Disclaimer

This is a personal research project. Keep generated builds and training output
outside version control, and run perft after position or move-generation edits.
Never run Kepler on a normal human account or through browser automation.
Lichess use is supported only through an explicitly converted BOT account and
the official Bot API deployment above; other sites require their own written
bot policy and authorization.

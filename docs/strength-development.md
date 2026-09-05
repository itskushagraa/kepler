# Kepler strength-development protocol

This is the authoritative path for improving Kepler's playing strength. A
static-evaluation score is a diagnostic, a paired A/B result is a promotion
decision, and a reference-ladder result is a rating estimate. They answer
different questions and must not be substituted for one another.

## Current diagnosis (2026-08-22)

The live false-mate score was a real search-state bug, not merely a weak
evaluation. Legal move generation discarded the `isCastle` flag. Search made
the rook move, then failed to undo it, corrupting later root branches. The
exact Lichess position now returns an ordinary finite score with pruning on or
off and with one or four threads.

The corrected baseline remains weak enough that evaluation is a major
bottleneck. On 2,000 held-out teacher positions:

| configuration | MAE | RMSE | sign accuracy | correlation |
| --- | ---: | ---: | ---: | ---: |
| checked-in baseline, weight 25/clamp 300 | 267.9 cp | 355.9 cp | 65.5% | 0.568 |
| old H256, weight 75/clamp 300 | 231.3 cp | 307.0 cp | 69.6% | 0.573 |

The old H256 blend grid reached about 274.8 cp RMSE at weight 50 with no
clamp. That is useful evidence that NNUE can help, but it is not a promotion
result. The old network used only 100,000 rows from 1,989 games, used the old
compressed target scale, and its previous game test ran through the castling
corruption. Treat every old A/B result as invalid.

A corrected 32-position move-quality sample at 100,000 Kepler nodes and
300,000 Stockfish nodes found a 50% exact-best-move rate, 34.9 cp average
loss, 11.5 cp median loss, and 105.9 cp 90th-percentile loss. The sample was
much weaker with queens present (54.5 cp average loss) than without queens
(6.3 cp in rook endings). This is a small diagnostic sample, not an Elo
estimate, but it points to middlegame/opening search and evaluation as the
largest current weakness.

## NixOS development environments

The repository pins Nixpkgs in `flake.lock` and provides two shells:

```bash
nix develop          # C++ toolchain, Python tooling, and Stockfish 18
nix develop .#cuda   # the same tools plus sm_86 CUDA, nvcc, and CUDA PyTorch
```

The CUDA shell targets the RTX 3070's compute capability 8.6 without enabling
CUDA globally for every Nixpkgs package. On a multi-user Nix installation,
configure `https://cache.nixos-cuda.org` and its public key in the system Nix
settings before realizing the shell. Otherwise Nix may attempt a very large
local PyTorch/CUDA build.

Verify actual GPU execution, not merely the presence of `nvcc`:

```bash
nix develop .#cuda
nvidia-smi
nvcc --version
python3 - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 6)
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), y.norm().item())
PY
```

## Performance foundation (2026-09-05)

Before adding or retuning more pruning heuristics, the engine's main known hot
paths were simplified:

- The shared four-way clustered TT publishes packed entries through lock-free
  64-bit atomics with XOR verification. Probes and stores no longer acquire a
  striped mutex, and `Hash` allocations round down to the requested ceiling.
- Slider move generation, attack detection, SEE, and classical mobility now
  share precomputed magic-bitboard attack tables instead of scanning rays.
- Production search is explicitly Lazy SMP; the unreachable root-split branch
  inside single-worker search was removed.
- Clock allocation uses explicit moves-to-go when supplied and otherwise
  preserves a conservative 50-move sudden-death horizon, including endgames.
  It reserves `MoveOverhead` for every anticipated move. Completed iterations
  adjust the soft deadline using PV/score stability, the top-two root score
  gap, and aspiration failure count while retaining a 2x hard bound shared by
  every worker. The main worker cancels helpers before joining them.
- Weight 100/clamp 0 uses NNUE directly when a model is loaded. Other blends
  intentionally retain the classical evaluation, and the production default
  is unchanged pending a promoted network.

Use repeated `bench 6` runs at Threads 1, 2, and 4 to assess NPS scaling, then
use the existing paired A/B protocol to decide whether a performance or timing
change improves Elo. NPS alone is not a promotion result.

## 0. Establish a clean tested binary

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
ctest --test-dir build-release --output-on-failure
./build-release/perft --full
python3 -m unittest discover -s tests -p 'test_*.py'
```

Do not compare results from different engine hashes. The experiment reports
engine and model SHA-256 hashes for this reason.

## 1. Generate a promotion-scale teacher corpus

```bash
systemd-inhibit --what=sleep --mode=block \
  --why="Kepler teacher dataset generation" \
  python3 tools/generate_teacher_shards.py \
  --engine build-release/kepler \
  --stockfish "$(command -v stockfish)" \
  --out-dir "$HOME/kepler-data/kepler-strength-data-v1" \
  --games 24000 \
  --games-per-shard 500 \
  --workers BENCHMARKED_WORKER_COUNT \
  --label-nodes 20000 \
  --curated-rows 1000000
```

The command runs independent shards in parallel, writes each shard log, and
resumes completed shards. It automatically curates game-disjoint train and
validation sets. It fails unless the corpus has at least one million rows,
20,000 games, 32 distinct randomized-opening fingerprints, a controlled
extreme-score fraction, and no single-game concentration. On machines where
Stockfish is on `PATH`, `--stockfish stockfish` is sufficient.

Do not use `--skip-strength-gate` for a real candidate. That option exists
only for smoke tests.

## 2. Train matched H128 and H256 candidates

Start with FP32. Enable `--amp` only after the FP32 CUDA smoke and parity tests
pass. The feature table is a sparse PyTorch embedding, so each batch gathers
and updates only active HalfKP rows. The NumPy backend remains available with
`--backend numpy --device cpu` and is the default when no backend is supplied.

```bash
systemd-inhibit --what=sleep --mode=block \
  --why="Kepler CUDA NNUE training" \
  python3 tools/train_strength_models.py \
  --dataset-dir "$HOME/kepler-data/kepler-strength-data-v1/curated" \
  --out-dir "$HOME/kepler-data/kepler-strength-models-v1" \
  --backend torch \
  --device cuda \
  --epochs 80 \
  --batch-size BENCHMARKED_GPU_BATCH \
  --result-weight 0.15 \
  --target-clip 2000 \
  --wdl-cp 600 \
  --checkpoint-dir "$HOME/kepler-data/kepler-strength-checkpoints-v1"
```

Each completed epoch is written atomically. To resume both model sizes, point
the wrapper at the checkpoint root; it selects `h128/latest.pt` and
`h256/latest.pt` independently:

```bash
python3 tools/train_strength_models.py \
  --dataset-dir "$HOME/kepler-data/kepler-strength-data-v1/curated" \
  --out-dir "$HOME/kepler-data/kepler-strength-models-v1" \
  --backend torch --device cuda --epochs 80 \
  --checkpoint-dir "$HOME/kepler-data/kepler-strength-checkpoints-v1" \
  --resume-checkpoint "$HOME/kepler-data/kepler-strength-checkpoints-v1"
```

Metrics and checkpoints record dataset hashes, Git and flake revisions,
backend, device, GPU, CUDA/PyTorch versions, seed, optimizers, batch size,
best epoch, and validation history.

The trainer keeps teacher values in centipawns, blends in game outcome on the
same scale, restores the best held-out epoch, reports errors by phase,
material and result, and rejects undersized or game-leaking data. Both model
files must load and pass incremental-accumulator validation:

```bash
./build-release/eval_consistency "$HOME/kepler-data/kepler-strength-models-v1/halfkp_h128.nnue"
./build-release/eval_consistency "$HOME/kepler-data/kepler-strength-models-v1/halfkp_h256.nnue"
```

## 3. Calibrate each candidate statically

```bash
python3 tools/tune_eval_blend.py \
  --engine build-release/kepler \
  --model "$HOME/kepler-data/kepler-strength-models-v1/halfkp_h128.nnue" \
  --data "$HOME/kepler-data/kepler-strength-data-v1/curated/validation.tsv" \
  --positions 2000 \
  --weights 25,50,75,100 \
  --clamps 0,150,300,600 \
  --json-out "$HOME/kepler-data/kepler-strength-models-v1/h128-blend-grid.json"
```

Repeat for H256. Use the best held-out RMSE setting as that model's A/B
configuration. The grid is a cheap calibration filter; it cannot promote a
model because lower teacher error does not guarantee stronger play.

## 4. Run A/A sanity and paired A/B

For each candidate, run the complete resumable experiment. Substitute the
candidate weight and clamp selected above:

```bash
systemd-inhibit --what=sleep --mode=block \
  --why="Kepler NNUE A/B experiment" \
  python3 tools/run_nnue_experiment.py \
  --engine build-release/kepler \
  --baseline models/kepler_baseline_pst_v1.nnue \
  --candidate "$HOME/kepler-data/kepler-strength-models-v1/halfkp_h128.nnue" \
  --validation-data "$HOME/kepler-data/kepler-strength-data-v1/curated/validation.tsv" \
  --out-dir "$HOME/kepler-data/kepler-strength-models-v1/h128-experiment" \
  --aa-games 100 \
  --games 2000 \
  --min-games 100 \
  --nodes 100000 \
  --threads 1 \
  --hash 128 \
  --baseline-weight 25 \
  --baseline-clamp 300 \
  --candidate-weight 50 \
  --candidate-clamp 0 \
  --elo0 0 \
  --elo1 10 \
  --draw-rate 0.70
```

The A/A run uses identical files and settings to detect color/opening or
runner bias. The A/B run cannot start unless that matching report passes.
Games are paired by opening with colors swapped and evaluated as
pentanomial pair outcomes. JSON and PGN checkpoints are atomically rewritten
after every completed pair, and rerunning the same command resumes them.

Only `accept_candidate` is a promotion. `reject_candidate` means keep the
baseline. `inconclusive` means the run reached its game cap without enough
evidence; it is not a weak acceptance. The wrapper returns exit code 0, 1, or
2 for those three outcomes respectively.

## 5. Diagnose the promoted configuration's search

```bash
python3 tools/search_quality.py \
  --engine build-release/kepler \
  --stockfish "$(command -v stockfish)" \
  --pgn 'deploy/lichess/games/kepler_bot games.pgn' \
  --model /path/to/promoted.nnue \
  --engine-name kepler_bot \
  --positions 128 \
  --nodes 100000 \
  --sf-nodes 1000000 \
  --threads 1 \
  --nnue-weight SELECTED_WEIGHT \
  --nnue-clamp SELECTED_CLAMP \
  --json-out "$HOME/kepler-data/kepler-search-quality.json"
```

Use the phase/material breakdown and worst positions to choose one search
change at a time. Search changes use the same A/A plus paired A/B path with
`--engine-b`; do not mix several unmeasured heuristics into one binary.

## 6. Measure rating only after promotion

Candidate promotion measures relative improvement. A reference ladder then
measures the resulting engine on a declared external scale. Keep all hashes,
node limits, openings and anchor values fixed, and reject a ladder result
whose optimizer did not converge, whose Kepler estimate is on a boundary, or
whose confidence interval is too wide. See the reference-ladder section in
the main README for the command.

## Interpretation rules

- Perft and state round trips establish chess-state correctness, not strength.
- Static teacher error isolates evaluation quality, not playing Elo.
- Search move loss isolates decisions at one node budget, not game strength.
- Paired A/B determines whether a candidate improves this baseline/protocol.
- The reference ladder estimates rating on the explicitly chosen anchor
  scale; it is never universal or hardware-independent Elo.

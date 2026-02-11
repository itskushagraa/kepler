# Kepler Engine - Data/NNUE Bootstrap Workflow

## Locked Baseline

Current locked baseline NNUE file is:

- `models/kepler_baseline_pst_v1.nnue`

By default, UCI `UseBaseline` is enabled and the engine will use this file.
You can re-load it manually at any time with:

```text
usebaseline
```

Generate self-play training data:

```bash
./build/kepler
```

Then in UCI:

```text
selfplay games 20 movetime 30 maxply 300 randomplies 8 sampleevery 2 minsampleply 8 seed 42 outfile /tmp/kepler_data.tsv
```

Train a bootstrap NNUE file from that dataset:

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

Dataset rows use `result<TAB>score<TAB>fen` and may include extra metadata columns after FEN (trainer ignores extras).

Load the produced NNUE:

```text
setoption name EvalFile value /tmp/kepler_bootstrap.nnue
eval
```

Reproducible datagen+train+compare cycle:

```bash
python3 tools/nnue_cycle.py \
  --engine build/kepler \
  --stockfish stockfish \
  --games 200 \
  --movetime 20 \
  --datagen-mode teacher \
  --teacher-play-elo 3000 \
  --teacher-label-elo 0 \
  --filter-min-ply 20 \
  --filter-max-ply 60 \
  --filter-max-abs-score-cp 200 \
  --trainer-arch halfkp \
  --trainer-hidden-size 1536 \
  --trainer-epochs 80 \
  --candidate base,0.4,300,100,0.5 \
  --candidate score_heavy,0.35,280,100,0.5
```

Outputs:
- `summary_<timestamp>.json` with full config + parsed metrics
- `results_<timestamp>.csv` sorted by bench NPS

Run a gauntlet with a candidate model loaded via `EvalFile`:

```bash
python3 tools/rated_gauntlet.py \
  --engine build/kepler \
  --stockfish stockfish \
  --baseline-model models/kepler_baseline_pst_v1.nnue \
  --eval-model /tmp/kepler_bootstrap.nnue \
  --max-plies 0 \
  --levels 2000,2200,2400,2500 \
  --games-per-level 20
```

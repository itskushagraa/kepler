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
  --result-weight 0.8 \
  --cp-scale 400 \
  --target-cp 100
```

Load the produced NNUE:

```text
setoption name EvalFile value /tmp/kepler_bootstrap.nnue
eval
```

Reproducible datagen+train+compare cycle:

```bash
python3 tools/nnue_cycle.py \
  --engine build/kepler \
  --games 200 \
  --movetime 20 \
  --candidate base,0.8,400,100,1.0 \
  --candidate score_heavy,0.6,300,100,1.0
```

Outputs:
- `summary_<timestamp>.json` with full config + parsed metrics
- `results_<timestamp>.csv` sorted by bench NPS

# Installation (One-Time)

This repo assumes you run commands via `.venv/bin/python` (no activation required) and log runs via:

- `--report` → writes `run_summary.json`, `train.log`, `metrics.jsonl`, and `checkpoint.pt` into `reports/...`
- `--wandb` → streams metrics to Weights & Biases (W&B)

Training entrypoint rules:
- `amg.py` requires a config path (`<config.yaml>` or `--config <path>`)
- `amg.py` requires `--run-note "..."` for every run
- optional `--run-postfix <tag>` appends a suffix to run names
- existing run folders are protected: duplicate run names raise an error

## 1) Create a virtualenv and install dependencies

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 2) (Optional) CarRacing / Box2D troubleshooting

If `gymnasium.make("CarRacing-v3")` fails on your machine:

```bash
.venv/bin/python -m pip install swig
.venv/bin/python -m pip install "gymnasium[box2d]"
```

## 3) (Optional) Plot/video extras

If plotting or gameplay recording fails due to missing packages:

```bash
.venv/bin/python -m pip install matplotlib moviepy
```

## 4) Configure W&B (recommended)

Export your API key (and optional entity) in your shell, or put them in a local `.env` and `source .env`:

```bash
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export WANDB_ENTITY=your-team   # optional
```

## 5) Verify the install

```bash
.venv/bin/python -m pytest -q tests/test_components.py tests/test_algorithms.py tests/test_reporting.py
```

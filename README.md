# DSTNet: Dynamic Spatio-Temporal Network for Motion Forecasting

> A research-grade PyTorch implementation of **DSTNet** for trajectory prediction on the **Argoverse 1 Motion Forecasting Dataset**.

---

# Overview

This repository provides a modular and reproducible implementation of **DSTNet**, a transformer-based trajectory prediction network proposed for autonomous driving.

The implementation follows the methodology presented in the original paper while adopting modern software engineering practices suitable for academic research and future extensions.

The repository is being developed as the baseline implementation for an M.Tech thesis. After reproducing the original model, additional research contributions will be built on top of this codebase.

---

# Features

- Complete PyTorch implementation
- Argoverse 1 Motion Forecasting support
- Offline preprocessing and dataset caching
- Modular transformer architecture
- Configurable experiments
- Mixed precision training support
- Automatic checkpointing
- TensorBoard logging
- Resume training
- Evaluation metrics
- Visualization utilities
- Reproducible experiments

---

# Repository Structure

```
DSTNet/
│
├── configs/
│
├── data/
│   └── argoverse1/
│       ├── train/
│       ├── val/
│       ├── test/
│       └── cache/
│
├── datasets/
│
├── models/
│   ├── encoder/
│   ├── attention/
│   ├── decoder/
│   ├── refinement/
│   └── layers/
│
├── losses/
│
├── engine/
│
├── evaluation/
│
├── utils/
│
├── scripts/
│
├── tests/
│
├── checkpoints/
│
├── outputs/
│
├── logs/
│
├── requirements.txt
│
└── README.md
```

---

# Dataset

The implementation expects the official Argoverse 1 Motion Forecasting dataset.

```
data/
└── argoverse1/
    ├── train/
    │   ├── xxxx.csv
    │   ├── xxxx.csv
    │   └── ...
    │
    ├── val/
    │   ├── xxxx.csv
    │   └── ...
    │
    ├── test/
    │   ├── xxxx.csv
    │   └── ...
    │
    └── cache/
        ├── train/
        ├── val/
        └── test/
```

Each CSV file corresponds to a single traffic scene.

---

# Development Workflow

The repository is developed in the following stages.

| Version | Description           |
| ------- | --------------------- |
| v0.1    | Repository Foundation |
| v0.2    | Dataset Pipeline      |
| v0.3    | Scene Representation  |
| v0.4    | Tri-ATM               |
| v0.5    | Decoder               |
| v0.6    | Adaptive Refinement   |
| v0.7    | Training Pipeline     |
| v0.8    | Evaluation            |
| v0.9    | Optimization          |
| v1.0    | Paper Reproduction    |

---

# Project Goals

The primary goals are

- faithfully reproduce DSTNet
- maintain clean modular architecture
- provide reproducible experiments
- simplify future research extensions
- support Argoverse 1
- support CPU and GPU execution

---

# Installation

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Preprocessing

Before training, preprocess the dataset.

```bash
python scripts/preprocess.py
```

This creates versioned cached `SceneData` pickle files under

```
data/argoverse1/cache_v2/
```

which significantly reduces loading time during training.

---

# Training

Debug mode

```bash
python scripts/train.py --config configs/debug.yaml
```

Local CPU

```bash
python scripts/train.py --config configs/local_cpu.yaml
```

Paper configuration

```bash
python scripts/train.py --config configs/paper.yaml
```

---

# Evaluation

```bash
python scripts/evaluate.py
```

---

# Inference

```bash
python scripts/infer.py
```

---

# Logging

Training logs are stored inside

```
logs/
```

TensorBoard files

```
outputs/tensorboard/
```

Model checkpoints

```
checkpoints/
```

---

# Coding Standards

The repository follows

- Python 3.10+
- PyTorch 2.x
- PEP-8
- Type hints
- Google style docstrings
- Modular design
- Object-oriented implementation
- Minimal hardcoded constants
- Reproducible experiments

---

# Kaggle Training and Cache Preparation

The Kaggle training entry point is `scripts/train_kaggle.py`. It expects the
official Argoverse CSV splits for scene indexing and three preprocessed cache
datasets. The Argoverse 1 HD maps are needed when building caches.

## Build the cache datasets

The corrected preprocessor aligns every actor to the focal agent's timestamps.
Existing caches were built with the old row-order alignment and must be
regenerated. The cache builder writes the exact `SceneData` pickle format used
by the trainer, a `VERSION` file, and a completeness manifest.

Run one command per Kaggle notebook session so each output stays below the
15 GB `/kaggle/working` limit:

```bash
python scripts/prepare_kaggle_cache.py --split train-1
python scripts/prepare_kaggle_cache.py --split train-2
python scripts/prepare_kaggle_cache.py --split val
```

Each command creates an output folder under `/kaggle/working` containing
`cache/` and `manifest.json`. Publish each output folder as a separate Kaggle
dataset. The two training commands use a deterministic, size-balanced split of
the complete training CSV directory; the validation command caches the full
validation split. If Kaggle mounts your Argoverse or map files at different
paths, pass `--train-root`, `--val-root`, or `--map-root`, or set the matching
`DSTNET_TRAIN_ROOT`, `DSTNET_VAL_ROOT`, or `DSTNET_MAP_ROOT` environment
variable. `--output-root` and `--max-cache-gb` can also be overridden.

## Run training

Attach all three published cache datasets and the Argoverse CSV datasets to
the training notebook, then run:

```bash
python scripts/train_kaggle.py
```

For the full two-stage model, first train the backbone with the default
settings and publish `/kaggle/working/checkpoints/best_model.pth` as a Kaggle
dataset. In a new run, attach that checkpoint dataset and start the refinement
stage by setting these variables before launching the script:

```python
import os
os.environ["DSTNET_REFINEMENT_ENABLED"] = "1"
os.environ["DSTNET_INIT_CHECKPOINT"] = "/kaggle/input/<checkpoint-dataset>/best_model.pth"
os.environ["DSTNET_CHECKPOINT_ROOT"] = "/kaggle/working/refinement-checkpoints"
os.environ["DSTNET_LOG_ROOT"] = "/kaggle/working/refinement-logs"
os.environ["DSTNET_BATCH_SIZE"] = "4"
```

The refinement stage initializes all compatible backbone weights and starts a
fresh optimizer and learning-rate schedule for its new parameters. Lower the
batch size further if the selected Kaggle GPU runs out of memory.

By default the trainer expects these input paths:

```text
/kaggle/input/datasets/kedaradhikari/dstnet-training-cache-part-1/cache
/kaggle/input/datasets/kedaradhikari/dstnet-training-cache-part-2/cache
/kaggle/input/datasets/kedaradhikari/dstnet-validation-cache/cache
```

Override the paths with `DSTNET_TRAIN_CACHE_A`, `DSTNET_TRAIN_CACHE_B`, and
`DSTNET_VAL_CACHE` if the published dataset slugs differ. Before training,
the script checks cache versions, manifests, missing or extra sequence IDs,
and overlap between training shards. Set `DSTNET_VALIDATE_FINITE=1` to turn
on the detailed per-layer numerical checks; they are off by default in the
Kaggle entry point to avoid synchronizing the GPU after every tensor check.

Useful run-time overrides include `DSTNET_BATCH_SIZE`, `DSTNET_NUM_WORKERS`,
`DSTNET_EPOCHS`, `DSTNET_LEARNING_RATE`, and `DSTNET_VALIDATE_EVERY`.

---

# Future Extensions

Once the baseline reproduction is complete, the repository will be extended with

- improved attention mechanisms
- enhanced multimodal prediction
- uncertainty estimation
- Argoverse 2 support
- Waymo Open Motion Dataset support
- distributed training

---

# Citation

If you use this implementation in academic work, please cite the original DSTNet paper.

---

# License

This repository is intended for research and educational purposes.

The implementation follows the methodology described in the original publication.

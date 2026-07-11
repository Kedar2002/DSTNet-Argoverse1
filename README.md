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

This creates cached tensor files under

```
data/argoverse1/cache/
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

# Neural Networks on Riemannian Geometry

Final project for the Information Geometry course exploring geometric deep learning on Riemannian manifolds.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Models and Manifolds](#models-and-manifolds)
- [Datasets](#datasets)
- [Usage](#usage)
- [Results](#results)
- [Notebooks](#notebooks)
- [License](#license)

## Overview

This project implements and compares various geometric deep learning approaches on Riemannian manifolds, with a focus on:

- **SPD Manifold Operations**: Symmetric Positive Definite matrices with the SPDNet implementation
- **Grassmann Manifold Operations**: Operations on Grassmann manifolds wwith the GrNet implementation

The project explores how geometric structure can be leveraged in deep learning, particularly for skeleton-based action recognition using the HDM05 motion capture dataset.

## Project Structure

```
riemannian_geometry/
├── README.md                    # This file
├── LICENSE                      # Project license
├── requirements.txt             # Python dependencies
├── accuracy.pdf                 # Model accuracy comparison results
├── loss.pdf                     # Training loss curves
├── skeleton_rotation.gif        # Visualization of skeleton data
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── data_spd.ipynb          # SPD representation exploration
│   ├── eda.ipynb               # Exploratory data analysis
│   ├── toy_example.ipynb       # Simple examples and demonstrations
│   └── images/                 # Generated plots and figures
│       ├── accuracy_comparison.png
│       └── dl_geometric_comparison.png
└── src/                        # Source code
    ├── config/                 # Configuration files
    │   └── paths.py           # Path configurations
    ├── data/                   # Data loading and preprocessing
    │   ├── data_loader.py     # Main data loading utilities
    │   ├── datasets.py        # PyTorch dataset classes
    │   ├── datasets_p.py      # Alternative dataset implementations
    │   ├── hdm05_loader.py    # HDM05 dataset specific loader
    │   ├── spd_repr.py        # SPD matrix representations
    │   ├── grassman_repr.py   # Grassmann manifold representations
    │   ├── preprocessing.py   # Data preprocessing utilities
    │   └── windowing.py       # Time series windowing
    ├── manifolds/              # Manifold operations
    │   ├── ops.py             # General manifold operations
    │   ├── spd_ops.py         # SPD manifold specific operations
    │   └── grassmann_ops.py   # Grassmann manifold operations
    ├── models/                 # Neural network architectures
    │   ├── baselines.py       # Baseline models
    │   ├── spdnet.py          # SPDNet implementation
    │   ├── grnet.py           # Grassmann Network
    │   └── grgcn.py           # Grassmann Graph Convolutional Network
    ├── training/               # Training scripts and utilities
    │   ├── train_baseline.py  # Baseline model training
    │   ├── train_spdnet.py    # SPDNet training
    │   ├── train_grnet.py     # GRNet training
    │   ├── train_grgcn.py     # GRGCN training
    │   ├── grassmann_training.py  # Grassmann-specific training utilities
    │   ├── losses.py          # Loss functions
    │   ├── eval.py            # Evaluation utilities
    │   └── utils.py           # Training utilities
    ├── evaluate/               # Evaluation scripts
    │   ├── evaluate_baseline.py   # Baseline evaluation
    │   ├── evaluate_spdnet.py     # SPDNet evaluation
    │   └── analyze_spdnet.py      # SPDNet analysis tools
    ├── download_data.py        # Script to download datasets
    └── run_pipeline.py         # Main pipeline runner
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy, SciPy
- Additional dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rdgzmanuel/riemannian_geometry.git
cd riemannian_geometry
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python src/download_data.py
```

4. Running the full pipeline to preprocess the data:
```bash
python src/run_pipeline.py
```

## Models and Manifolds

### Symmetric Positive Definite (SPD) Manifold

The SPD manifold consists of symmetric matrices with positive eigenvalues. This space has a natural Riemannian structure and is particularly useful for covariance-based representations.

**Implemented operations** (in `src/manifolds/spd_ops.py`):
- Riemannian metric
- Logarithmic and exponential maps
- Geodesic computations

**SPDNet** (in `src/models/spdnet.py`):
A neural network that operates directly on the SPD manifold, respecting its geometric structure.

### Grassmann Manifold

The Grassmann manifold represents the space of k-dimensional linear subspaces of ℝⁿ. It's fundamental in many applications including computer vision and signal processing.

**Implemented operations** (in `src/manifolds/grassmann_ops.py`):
- Grassmann metric
- Projection operations
- Geodesic calculations

**Grassmann Networks**:
- **GRNet** (in `src/models/grnet.py`): Neural network operating on Grassmann manifold

## Dataset

### HDM05 Motion Capture Dataset

The project uses the HDM05 dataset containing motion capture data of human skeletal movements. This dataset is particularly well-suited for geometric deep learning approaches.

**Data representations**:
- **SPD representation**: Covariance matrices of skeletal joint positions
- **Grassmann representation**: Subspace representations of motion patterns

**Data pipeline**:
1. Raw motion capture data loading (`hdm05_loader.py`)
2. Preprocessing and normalization (`preprocessing.py`)
3. Time series windowing (`windowing.py`)
4. Manifold representation conversion (`spd_repr.py`, `grassman_repr.py`)

## Usage

### Training Models

Train baseline models:
```bash
python src/training/train_baseline.py
```

Train SPDNet:
```bash
python src/training/train_spdnet.py
```

Train Grassmann networks:
```bash
python src/training/train_grnet.py
```

### Evaluation

Evaluate trained models:
```bash
python src/evaluate/evaluate_baseline.py
python src/evaluate/evaluate_spdnet.py
```

Analyze SPDNet results:
```bash
python src/evaluate/analyze_spdnet.py
```

## Results

The project includes comprehensive results comparing different geometric approaches:

- **accuracy.pdf**: Comparison of model accuracies across different architectures
- **loss.pdf**: Training and validation loss curves
- **skeleton_rotation.gif**: Visualization of the motion capture data

Key findings and performance comparisons can be found in the notebooks and result PDFs.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for exploration and analysis:

- **toy_example.ipynb**: Simple demonstrations of manifold operations and concepts
- **eda.ipynb**: Exploratory data analysis of the HDM05 dataset
- **data_spd.ipynb**: Exploration of SPD matrix representations and properties

These notebooks provide interactive visualizations and step-by-step explanations of the geometric concepts implemented in the project.

## License

This project is licensed under the terms specified in the LICENSE file.

---

**Course**: Information Geometry  
**Authors**: Natalia Leyenda, Joaquín Mir, Sofía Pedrós, Manuel Rodriguez ([@rdgzmanuel](https://github.com/rdgzmanuel))  
**Year**: 2025
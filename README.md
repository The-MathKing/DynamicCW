# DynamicCW: Topologically-Motivated Graph Representation Learning

This repository contains the official codebase for **DynamicCW**, an architectural framework that topologically lifts standard graphs into 2-dimensional regular cell complexes. By elevating 2-cells (faces/cycles) to first-class, learning-active entities with continuous embeddings, the model breaks past the strict 1-dimensional Weisfeiler-Lehman (1-WL) expressivity limit.

To ensure strict permutation equivariance while maintaining its topological capabilities, DynamicCW aggregates cross-dimensional messages via the **absolute boundary incidence matrices** ($|B_1|$ and $|B_2|$). This provides the network with a powerful structural inductive bias that naturally counts face and edge participations, resulting in highly stable cross-domain transfer and topological awareness.

## Installation

We recommend using a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Benchmarks

This repository contains multiple suites for researchers to validate the model's topological representation power and zero-shot transfer capabilities.

### 1. Zero-Shot Cross-Domain Transfer
Evaluates the model's ability to generalize by training exclusively on the macro-structural `PROTEINS` dataset and testing zero-shot on `NCI1` molecular graphs.
```bash
python run_all_benchmarks_v2.py
```
*(You can also use `python run_full_benchmarks.py` to run the legacy baseline suite).*

### 2. $H_1$ Homology Ablation Test
Validates the network's topological awareness by comparing the global Euclidean representation shift when deleting a non-cycle (terminal) edge versus a cycle-forming topological edge.
```bash
python run_betti_ablation.py
```

### 3. Light / Dry Run Tests
For rapid local verification and prototyping without downloading massive datasets.
```bash
python run_light_tests.py
```

## Core Codebase Structure
- **`model.py`**: The heart of the architecture. Contains the `CellularMessagePassingLayer` and `CurvatureMPSN` which process multi-dimensional tensors dynamically.
- **`data_processing.py`**: Contains the topological logic for lifting standard PyTorch Geometric 1D graphs into 2D simplicial (clique) complexes.
- **`train.py`**: The training loops and evaluation criteria.
- **`adversarial_utils.py`**: Tools for targeted robustness and adversarial topological perturbations.

# DynamicCW: Discrete Ricci Flow and CW Complexes for Graph Neural Networks

This repository contains the official testing suite and implementation for evaluating Graph Neural Networks (GNNs) equipped with Discrete Forman-Ricci Flow and higher-dimensional Cellular (CW) Complex message passing.

## Repository Structure

The codebase has been streamlined for reproducibility and testing. The core components are:

*   **`run_full_benchmarks.py`**: The primary orchestrator script. Run this file to execute the full $N=10$ cross-validation benchmarking suite. It evaluates manifold diversity, adversarial robustness, architectural ablations (gating and 3-cells), and scalability profiling. Results are aggregated into a final JSON file.
*   **`model.py`**: Contains the novel `CurvatureWeightedSimplicialConv` layer and the `CurvatureMPSN` architecture. This implements the geometry-guided message filtering.
*   **`model_baselines.py`**: Contains the standard Graph Convolutional Network (GCN) baseline used for benchmarking performance gains.
*   **`data_processing.py`**: Handles the topological lifting of standard PyTorch Geometric graphs into TopoNetX Simplicial Complexes (up to 3-cells/tetrahedrons) and computes the discrete Forman-Ricci Curvature for 1-cells (edges).
*   **`adversarial_utils.py`**: Utilities for injecting structural noise (e.g., random edge deletion) and breaking cycles to test model robustness.
*   **`train.py`**: Helper module containing the training and evaluation loops, as well as the logic to extract boundary incidence matrices ($B_1$, $B_2$) from the complexes.

## Environment Setup

To run the benchmarking suite, ensure your Python environment is set up with PyTorch, PyTorch Geometric, and TopoNetX.

Additionally, to run the scalability profiling on the multi-million node graphs, you must install the Open Graph Benchmark (OGB):

```bash
pip install ogb
```

## Running the Benchmarks

To execute the entire statistical validation suite:

```bash
python run_full_benchmarks.py
```

*Note: The orchestrator script processes thousands of graphs multiple times ($N=10$ independent trials) across different architectures. This is computationally intensive and can take several hours depending on hardware acceleration.*

# DynamicCW: When Does Curvature Gating Help Cellular Message Passing, and When Can't It?

Official code and reproducibility suite for the theoretical analysis and empirical benchmarking of **Curvature-Gated Cellular Message Passing**.

**Authors**: Aryan Padarthi, Raghav Srinivasan, Olivia Kim, Ethan Ye  
*Allen High School, Allen, TX, USA*

---

## Abstract & Research Questions

Higher-order graph neural networks operating on 2-dimensional CW complexes lift graphs to vertices (0-cells), edges (1-cells), and rings (2-cells) to surpass the 1-dimensional Weisfeiler-Lehman (1-WL) limit. Simultaneously, discrete Forman-Ricci curvature has been widely proposed as a geometric inductive bias to mitigate over-squashing across topological bottlenecks.

This repository provides the formal derivations, unit tests, synthetic experiments, and standard benchmarks investigating:
1. **The Expressivity Ceiling Theorem:** Proving that combinatorial Augmented Forman-Ricci Curvature ($\mathrm{AF}_3$) on a 2-cell complex is completely determined by local 1-step cellular Weisfeiler-Lehman (1-CWL) colorings, adding zero strictly new distinguishing expressivity beyond standard Cellular Message Passing on the same complex.
2. **Symmetry & Homology Limits:** Proving that on co-spectral strongly regular graphs (SRG-16-6-2-2), $\mathrm{AF}_3 \equiv -2$ is uniformly invariant, and local cellular message passing cannot detect non-local 1-homology without chordless cycle lifting.
3. **Disentangling Curvature Inductive Biases:** Parameter-matched 10-seed ablation grid comparing $\mathrm{AF}_3$ against degree-only ($\kappa_{\text{deg}} = 4 - d_u - d_v$), cycle-aware Forman, shuffled $\kappa$, un-gated, dynamic vs static faces, and sum vs mean readouts.
4. **Dirichlet Energy & Over-Smoothing:** Measuring normalized Hodge 0- and 1-Dirichlet energy across cellular layers to analyze the role of residuals and normalization in preventing collapse.

---

## Directory Structure

```
code/
├── data_processing.py              # Canonical, permutation-equivariant chordless cycle lifting & curvature variants
├── model.py                        # CurvatureWeightedCellularConv & DynamicCWNet with LayerNorm and residuals
├── train.py                        # Training loops, TopoNetX incidence matrices, and data processing
├── experiments_synthetic.py        # SRG expressivity, cycle counting regression, & bottleneck transfer
├── run_comprehensive_benchmarks.py # Multi-seed ablation grid (10 seeds, paired t-tests, parameter matching)
├── run_dirichlet_energy.py         # Hodge 0- and 1-Dirichlet energy tracking across cellular depths
├── generate_paper_figures.py       # Script generating publication vector figures with error bars from JSON logs
├── tests/
│   └── test_invariants.py          # Unit tests for Betti numbers, permutation equivariance, & orientation invariance
└── requirements.txt                # Python dependencies
```

---

## Installation & Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

---

## Running Unit Tests & Invariant Verification

Run the automated test suite verifying exact SRG Betti numbers, permutation equivariance under node permutations, and orientation invariance:
```bash
python code/tests/test_invariants.py
```

---

## Running Controlled Synthetic & Expressivity Experiments

```bash
python code/experiments_synthetic.py
```
Outputs:
- SRG-16-6-2-2 2-clique complex distance ($L_2 = 0$) vs Chordless cycle lifting distance ($L_2 > 0$).
- Substructure cycle counting regression across curvature types.
- Bottleneck over-squashing graph transfer accuracy.

---

## Running Comprehensive Multi-Seed Ablations

```bash
python code/run_comprehensive_benchmarks.py
```
Evaluates all models across 10 random seeds on ZINC-12k under matched parameter budgets (~100k params), computing mean, standard deviation, 95% confidence intervals, and paired two-tailed $t$-tests with $p$-values.

---

## Generating Figures & Visualizations

```bash
python code/generate_paper_figures.py
```
Generates vector graphics in `figures/` directly from JSON logs:
- `figures/fig2_ablation_grid.pdf` / `.png`
- `figures/fig3_dirichlet_energy.pdf` / `.png`

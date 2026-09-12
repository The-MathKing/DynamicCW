# DynamicCW: When Does Forman-Ricci Curvature Gating Help Cellular Message Passing?

Official code and reproducibility suite for the theoretical analysis and empirical benchmarking of **Curvature-Gated Cellular Message Passing**.

**Authors**: Aryan Padarthi, Raghav Srinivasan, Olivia Kim, Ethan Ye  
*Allen High School, Allen, TX, USA*

---

## Abstract & Research Questions

Higher-order graph neural networks operating on 2-dimensional CW complexes lift graphs to vertices (0-cells), edges (1-cells), and rings (2-cells) to surpass the 1-dimensional Weisfeiler-Lehman (1-WL) limit. Simultaneously, discrete Forman-Ricci curvature has been widely proposed as a geometric inductive bias to mitigate over-squashing across topological bottlenecks.

This repository provides the formal derivations, unit tests, synthetic experiments, and standard benchmarks investigating:
1. **The Expressivity Ceiling Theorem:** Proving that combinatorial Augmented Forman-Ricci Curvature ($\mathrm{AF}_3$) on a 2-cell complex is completely determined by local 1-step cellular Weisfeiler-Lehman (1-CWL) colorings, adding zero strictly new distinguishing expressivity beyond standard Cellular Message Passing on the same complex.
2. **Symmetry & Homology Limits:** On the 3-WL-indistinguishable Rook's 4x4 / Shrikhande strongly regular graph pair (SRG-16-6-2-2), $\mathrm{AF}_3 \equiv -2$ is uniformly invariant on the 2-truncated clique complex (the full clique complex already separates them trivially via Rook's 4-cliques, per Bodnar et al. 2021); chordless cycle lifting separates them deterministically via differing face types (Shrikhande alone has pentagons).
3. **Disentangling Curvature Inductive Biases:** Parameter-matched 20-seed ablation grid on a ZINC-12k subsample comparing $\mathrm{AF}_3$ against degree-only ($\kappa_{\text{deg}} = 4 - d_u - d_v$), cycle-aware Forman, shuffled $\kappa$, un-gated, dynamic vs static faces, and sum vs mean readouts, with TOST equivalence testing (not just non-significance) and Bonferroni correction.
4. **Dirichlet Energy & Over-Smoothing:** Measuring normalized 0- and 1-Dirichlet energy (the latter via an orientation-free unsigned edge-variation Laplacian, not the signed Hodge Laplacian) across cellular layers to analyze the role of residuals and normalization in preventing collapse.

---

## Directory Structure

```
code/
├── data_processing.py              # Canonical, permutation-equivariant chordless cycle lifting & curvature variants
├── model.py                        # CurvatureWeightedCellularConv & DynamicCWNet with LayerNorm and residuals
├── train.py                        # Training loops, TopoNetX incidence matrices, and data processing
├── verify_srg_separation.py        # Prop. 1: exact Betti/face-count verification + SRG separation test (disclosed init, no positional encoding)
├── experiments_synthetic.py        # Cycle counting regression (incl. 1-WL GIN baseline) & bottleneck transfer
├── run_synthetic_multiseed.py      # Multi-seed (25) wrapper for the above, with TOST equivalence testing
├── run_comprehensive_benchmarks.py # Multi-seed ablation grid (20 seeds, paired t-tests + TOST, Bonferroni correction)
├── run_dirichlet_energy.py         # 0- and 1-Dirichlet energy tracking across cellular depths
├── generate_paper_figures.py       # Script generating publication vector figures with error bars from JSON logs
├── tests/
│   └── test_invariants.py          # Unit tests for Betti numbers, permutation equivariance, & orientation invariance
└── requirements.txt                # Python dependencies

Note: run_all_benchmarks_v2.py, run_light_tests.py, run_multi_seed.py, run_betti_ablation.py,
run_full_benchmarks.py, run_mutag_benchmark.py, run_scaling_benchmark.py, run_zinc_benchmark.py,
model_baselines.py, adversarial_utils.py, check_datasets.py, and dry_run_test.py are exploratory
scripts from an earlier phase of this project and are **not** used to produce any number reported
in the current paper. See the warning docstring at the top of run_all_benchmarks_v2.py.
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

## Reproducing the SRG Separation Test (Proposition 1)

```bash
python code/verify_srg_separation.py
```
Recomputes every topological quantity in Proposition 1 from scratch in under a minute:
exact Betti vectors ((1,9,8) vs (1,2,1)), chordless face counts (164 vs 204), and the
$L_2$ embedding distance under triangle lifting ($\approx 0$) vs chordless lifting ($>0$),
using disclosed small-scale random initialization and no positional encoding.

---

## Running Controlled Synthetic & Expressivity Experiments

```bash
python code/experiments_synthetic.py       # single-seed smoke test
python code/run_synthetic_multiseed.py     # 25-seed version with TOST equivalence testing (paper numbers)
```
Outputs:
- Substructure cycle counting regression, including a parameter-matched 1-WL GIN baseline.
- Bottleneck over-squashing graph transfer convergence rate.

---

## Running Comprehensive Multi-Seed Ablations

```bash
python code/run_comprehensive_benchmarks.py
```
Evaluates all models across 20 random seeds on a 700-molecule ZINC-12k subsample under matched
parameter budgets (~100k params), computing mean, standard deviation, paired two-tailed $t$-tests,
TOST equivalence tests (margin 0.02 MAE), and a Bonferroni correction across comparisons. Set
`DYNAMICCW_N_WORKERS` to control parallelism (default 4; uses `multiprocessing` with the `spawn`
context, since `fork` deadlocks with pre-initialized PyTorch/BLAS thread pools on macOS).

---

## Generating Figures & Visualizations

```bash
python code/generate_paper_figures.py
```
Generates vector graphics in `figures/` directly from JSON logs:
- `figures/fig2_ablation_grid.pdf` / `.png`
- `figures/fig3_dirichlet_energy.pdf` / `.png`

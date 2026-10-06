# Distributed Nash Equilibrium Seeking under Joint Exogenous and Strategic Wasserstein Ambiguity

MATLAB implementations and numerical reproducibility package for the paper:

**Distributed Nash Equilibrium Seeking under Joint Exogenous and Strategic Wasserstein Ambiguity**  
Sichen Qian, Hongzhe Liu, and Wenwu Yu.

**Paper status:** Prepared for submission to *Automatica*.

## Numerical examples

| Script | Example | Figures in the manuscript |
| --- | --- | --- |
| `example1_theory_verification.m` | Four-player edge-service game: distributed convergence, auxiliary errors, 100 feasible initializations, and the original-game Nash gap | Figs. 1-2 |
| `example2_statistical_robustness.m` | Eight-player edge-service game: finite-sample coverage, held-out SNE gaps, and sensitivity to the two robustness radii | Figs. 3-4 |

Both scripts are self-contained. Game parameters, communication graphs, samples,
random seeds, solver settings, and local helper functions are defined in each
script. The distributed dynamics in Example 1 seek the fixed-dual proxy Nash
equilibrium. Reference equilibria of the original double-budget robust game are
computed offline from the exact semidual formulation.

Example 2 uses 5,000 Monte Carlo replications to calibrate each exogenous
Wasserstein radius and 3,000 independent replications to validate joint coverage.
For each sample size, 100 independent datasets are used to compute proxy
equilibria, each evaluated with 20,000 held-out realizations. The calibrated
radii describe empirical coverage of the known simulated uniform laws.

## Requirements

- MATLAB; tested with **R2024a**.
- **Optimization Toolbox**, used for the offline equilibrium computations
  (`fmincon` and, in Example 1, `lsqnonlin`).
- **Parallel Computing Toolbox** is optional. Example 2 detects it automatically
  and otherwise runs serially. A newly created pool uses at most 10 workers.

No external input files or LaTeX installation are required. Random seeds are
fixed; Example 2 uses local random streams for its statistical replications.

## Usage

Open either script in MATLAB and click **Run**, or execute the following commands
from the folder containing the scripts:

```matlab
example1_theory_verification
example2_statistical_robustness
```

The scripts can also be run from another folder using MATLAB's `run` function:

```matlab
run(fullfile('path', 'to', 'repository', 'example1_theory_verification.m'))
run(fullfile('path', 'to', 'repository', 'example2_statistical_robustness.m'))
```

Each script creates its output folder relative to its own location. No output
paths need to be edited:

```text
results/
  example1/
  example2/
```

Running a script again overwrites its corresponding output files. The examples
save their results in separate folders. The `results/` directory is excluded
from version control by `.gitignore` and is generated when the scripts run.

## Outputs

Each script exports standalone figures in `.fig`, `.png` (600 dpi), and `.eps`
formats, and saves a MAT file containing the parameters and numerical results:

- `results/example1/example1_results.mat`: private samples, analytical constants,
  theorem checks, gains, proxy and reference equilibria, the distributed
  trajectory, results from 100 initializations, and approximation bounds.
- `results/example2/example2_results.mat`: calibration and coverage samples,
  random seeds, held-out equilibrium gaps, reference equilibria, radius sweeps,
  dependence diagnostics, and fixed-multiplier sensitivity results.

Example 1 produces four figure stems; Example 2 produces seven. Figures are
saved as separate panels for assembly in the manuscript.

## Figure correspondence

| Figure | Output folder | File stem | Content |
| --- | --- | --- | --- |
| Fig. 1(a) | `results/example1/` | `example1_fig1a` | Strategy convergence to the proxy equilibrium, with the original-game DRNE as a reference |
| Fig. 1(b) | `results/example1/` | `example1_fig1b` | Strategy, consensus, and inner-response tracking errors |
| Fig. 2(a) | `results/example1/` | `example1_fig1c` | Normalized composite errors over 100 feasible initializations |
| Fig. 2(b) | `results/example1/` | `example1_fig1d` | Original-game eta-DRNE violation along the distributed trajectory |
| Fig. 3(a) | `results/example2/` | `example2_fig3a` | Empirical Wasserstein distance and calibrated radius for player 5 |
| Fig. 3(b) | `results/example2/` | `example2_fig3b` | Independently validated joint coverage with Wilson intervals |
| Fig. 3(c) | `results/example2/` | `example2_fig3c` | Held-out SNE gap as the sample size increases |
| Fig. 4(a) | `results/example2/` | `example2_fig4a` | Aggregate-service sensitivity to the strategic and exogenous radii |
| Fig. 4(b) | `results/example2/` | `example2_fig4c` | Original-game eta-DRNE violation at the fixed-dual proxy equilibrium |

The manuscript's Fig. 4(b) uses the file stem `example2_fig4c`. Two additional
panels are exported by Example 2:

- `example2_fig4b`: dependence sensitivity of the baseline proxy equilibrium
  under fixed marginal distributions.
- `example2_fig4d`: two-budget map of the original-game Nash gap at the fixed
  proxy equilibrium.

## Numerical checks

The scripts check the applicable gain and time-scale inequalities, equilibrium
residuals, and consistency of the reference computations. They stop with an
error when a required check fails. Example 1 also checks convergence from the
100 feasible initializations.

The Example 1 gain selection uses the analytical strong-monotonicity lower
bound of approximately 3.618. A sampled symmetric-Jacobian check is saved
separately as supplementary numerical evidence. Time-dependent plots use the
slow time `s = alpha*t`; the original time is retained in the MAT file.

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@unpublished{qian_joint_wasserstein_games,
  author = {Qian, Sichen and Liu, Hongzhe and Yu, Wenwu},
  title  = {Distributed Nash Equilibrium Seeking under Joint Exogenous and Strategic {Wasserstein} Ambiguity},
  year   = {2026},
  note   = {Prepared for submission to Automatica}
}
```

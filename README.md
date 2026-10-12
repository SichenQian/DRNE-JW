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
save their results in separate folders. This repository includes the archived
reference `results/`. Use the verification entry below to recompute in a
separate `reproduced_results/` folder and retain that reference.

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

## Full-precision verification and fixed version

The authoritative repository is
[`SichenQian/DRNE-JW`](https://github.com/SichenQian/DRNE-JW). The annotated tag
[`Automatica-reproducibility-v1`](https://github.com/SichenQian/DRNE-JW/releases/tag/Automatica-reproducibility-v1)
fixes the source, reference results, and verification evidence:

```sh
git checkout Automatica-reproducibility-v1
git rev-parse HEAD
```

From MATLAB in that checkout, run:

```matlab
summary = verify_reproduction;
figures = verify_figure_data;
```

The entry runs both complete examples, captures a fresh log, and compares
their numeric/logical MAT fields with the archived reference. Fresh outputs
go to `reproduced_results/`, preserving `results/`. An alternative folder can
be supplied to `verify_reproduction(outputFolder)`. The scripts also respect
`DRNE_RESULTS_DIR` when set; verification restores its previous value on exit.
Both primary scripts remain self-contained. The verification entry additionally
uses `compare_numeric_records.m` and `write_full_precision.m`.

| Verification item | Published record in `verification/` |
| --- | --- |
| Actual command, MATLAB environment, completion | `reproduction_log.txt`, `verification_summary.json` |
| Example 1 constants, chosen gains, thresholds and margins | `example1_certificate.txt`, `example1_theorem_audit.csv` |
| Example 1 realized samples, complete main initial state and 100 additional seeds | `example1_initialization.json`, `example1_certificate.txt` |
| Example 2 conditions for all nine sample sizes | `example2_certificate.txt`, `example2_theorem_audit.csv` |
| Monte Carlo counts, independent seed families, coverage and intervals | `statistical_protocol.json` |
| Fresh outputs versus archived numerical arrays | `reference_comparison.json` |
| All 11 generated figure panels versus archived figures | `figure_reference_comparison.json` |
| Manuscript parameters and all nine manuscript panels | `paper_correspondence.json`, `paper_correspondence.md` |

Certificates use 17 significant digits; full arrays remain in the MAT files.
Any large arrays summarized in a readable certificate are explicitly marked.
Example 1 records actual `ode15s` options, physical time, state order and the
integrated initial state. Its private-sample seed is `260826` (`twister`), its
main initial-state seed is `101`, and its additional seeds are `1001:1100`.
The distributed flow seeks the fixed-dual proxy equilibrium. The offline
semidual reference is checked by an original-game best-response residual.

Example 2 records calibration and independent validation seeds separately
from equilibrium-training and held-out evaluation seeds. Wilson intervals
use `z = 1.96`. Serial and optional parallel computations use local random
streams; actual worker counts are recorded. Worker counts and runtime are
excluded from array comparisons, with all exclusions listed. Comparison
tolerances and exact-equality results are both reported.

Success ends with `VERIFICATION_PASS`; failed conditions or reference
comparisons terminate execution. `verification_manifest.json` records the
inspected manuscript TEX/PDF and figure-source SHA-256 hashes and released
code/evidence Git blob hashes. The manuscript itself is not an input dependency.

### Interpretation of the certificates

Analytical monotonicity and gain inequalities support the specified proxy-game
instance. Sampled Jacobian eigenvalues provide supplementary checks. Empirical
Wasserstein-radius calibration uses the known generating uniform law; measured
coverage and Wilson intervals are Monte Carlo evidence, not a substitute for
the theoretical finite-sample assumptions. Strategic-radius sweep points below
the coverage threshold are sensitivity experiments.

The original robust-game gap at the fixed proxy generally remains positive.
The record retains the certified approximation bound and its ratio to the
measured gap, including when the bound is too conservative to be informative.
The unit-prefactor theoretical slope reference in Fig. 2(a) is distinguished
from the rigorous envelope with prefactor `c1`. Figure-data comparisons are
independent of fonts, colors and export metadata.

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

# Manuscript-to-code numerical correspondence

Audited on 2026-10-13 for `Automatica-reproducibility-v1`. The inspected manuscript is `autosam.tex` / `autosam.pdf`; the exact source hashes are in `paper_correspondence.json` and the root manifest.

The four-player game coefficients, realized radii, fixed multipliers, graph, selected gains and rounded equilibrium profiles match the numerical section. The eight-player experiment uses the specified nine sample sizes, 5000 calibration replications, 3000 independent coverage replications, 100 datasets per K and 20000 held-out draws. The four-player theory audit passes 19 conditions; the nine eight-player audits pass 135 conditions in total. Observed joint coverage ranges from 0.9440 to 0.95966666666666667, with the Wilson intervals retained. The maximum offline reference-DRNE residual is approximately 1.257e-13. Fixed-dual proxy equilibria and original-game DRNE references remain distinct. Empirical coverage uses the known simulated generating law.

The manuscript-side MAT records were compared against the archived repository records, and a fresh complete run was then compared against those same records. All compared numerical arrays are exactly identical in MATLAB R2024a; elapsed runtime and worker-count metadata are explicitly excluded. The fresh run ends with `VERIFICATION_PASS`.

| Manuscript panel | Active TEX figure source | Repository native figure | Numerical correspondence |
| --- | --- | --- | --- |
| Fig. 1(a) | `numerical_simulation/example1_fig1a.eps` | `results/example1/example1_fig1a.fig` | Exact scientific data |
| Fig. 1(b) | `numerical_simulation/example1_fig1b.eps` | `results/example1/example1_fig1b.fig` | Exact scientific data |
| Fig. 2(a) | `numerical_simulation/example1_fig1c.eps` | `results/example1/example1_fig1c.fig` | Exact scientific data |
| Fig. 2(b) | `numerical_simulation/example1_fig1d.eps` | `results/example1/example1_fig1d.fig` | Exact scientific data |
| Fig. 3(a) | `numerical_simulation/example2_fig3a.eps` | `results/example2/example2_fig3a.fig` | Exact scientific data |
| Fig. 3(b) | `numerical_simulation/example2_fig3b.eps` | `results/example2/example2_fig3b.fig` | Exact scientific data |
| Fig. 3(c) | `numerical_simulation/example2_fig3c.eps` | `results/example2/example2_fig3c.fig` | Exact scientific data |
| Fig. 4(a) | `numerical_simulation/example2_fig4a.eps` | `results/example2/example2_fig4a.fig` | Exact scientific data |
| Fig. 4(b) | `numerical_simulation/example2_fig4c.eps` | `results/example2/example2_fig4c.fig` | Exact scientific data |

The original Fig. 3(b) FIG contains an additional errorbar helper at K=-250, outside the manuscript plotting range. It is recorded in the JSON rather than counted as a statistical observation. The nine coverage estimates, both Wilson error arrays, and target line match exactly.

Fresh figure data were independently compared by `verify_figure_data`: 11 generated figures passed, including all 9 manuscript panels. Native line/patch/errorbar data are checked; fonts, colors, axes layout and export metadata are presentation differences. The inspected manuscript EPS/PDF panels were visually checked. This release establishes data reproduction, not byte-identical EPS output.

The manuscript LaTeX source and PDF were left unchanged. This numerical check does not re-prove the manuscript's theorems; the explicit assumptions and claim boundaries continue to apply.

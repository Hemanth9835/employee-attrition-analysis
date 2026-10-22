# Execution and methodology check

**Result: passed.** All four notebooks completed in fresh processes using IPython. All 96 code cells have sequential execution counts and no remaining errors.

| Issue found | Correction | Effect on modelling |
|---|---|---|
| Windows temporarily denied rewriting an unchanged CSV catalog | Skip CSV writes when the bytes already match | No data or preprocessing change |
| Fastest-fit ranking changed the secondary economical LR between runs | Keep the originally reviewed configuration, provided it still meets the 0.005 CV AP tolerance | Reproducible comparisons; primary winners and final policies unchanged |

730 numeric comparisons reproduce the original tables within saved display precision. All four reviewed tuned candidates and both NN baseline cases reproduce their original validation probabilities within floating-point precision. The freshly refitted final LR and 8-neuron NN also reproduce the original test probabilities. Runtime/latency differences are expected and are not treated as predictive changes.

All 33 processed CSVs across 11 variants pass row alignment, feature-order, finite-value and transform-reproduction checks. Stratified train/validation/test sizes remain 3,000/1,000/1,000. Tree variants remain unscaled. Original notebook and frozen model/score artifacts are unchanged.

| Frozen final candidate | Test AP | Test ROC-AUC |
|---|---:|---:|
| Logistic regression | 0.2695 | 0.7078 |
| Small NN (8 neurons) | 0.2746 | 0.7100 |

The recommendation remains LR as the simpler default candidate, with the small NN as a challenger. The operating recall/alert budget is a business choice among the already evaluated validation-defined policies; no test-driven threshold changes were made.

The analytical project can be concluded. Future deployment or additional tuning should be treated as separate work. This verification used IPython directly in clean Python processes, not a separately launched Jupyter UI kernel. The compact evidence is included in [artifacts/verification.json](artifacts/verification.json); detailed audit caches stay local and are excluded from Git.

## GitHub packaging check: 23 October 2026

A separate directory containing only the curated repository files was tested with the original Kaggle download and the existing pinned Python 3.11.4 environment.

- Notebook 1's setup recreated the English CSV byte-for-byte, preserving its recorded SHA-256.
- All 17 code cells in Notebook 2 ran and regenerated the 33 processed CSVs without relying on pre-existing generated folders.
- Notebook 3's seven startup cells loaded the regenerated inputs. Model searches were not repeated.
- All nine code cells in Notebook 4 ran, including an enabled frozen test-prediction check. Final ranking and decision tables matched the published results.
- All declared package pins matched the installed environment. Notebook syntax/schema and links to repository documents passed inspection.

The packaging changes add the existing translation mapping, portable repository discovery and output-directory creation. They do not change features, model parameters, splits or frozen thresholds. A new package installation on a different machine was not tested.

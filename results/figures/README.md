# Project diagrams

Three figures cover the four requested topics without repeating the same information.

| Figure | Purpose |
|---|---|
| 01_architecture_and_steps | Analytical architecture, data paths, processing sequence and guardrails |
| 02_statistical_findings | Exploratory statistical evidence and its implications |
| 03_experiments_and_model_selection | Baselines, experiments, focused tuning and the final LR/NN comparison |

Each figure is supplied as PNG for README display and SVG for resizing. The workflow and model-selection diagrams are embedded in the project README; the statistical diagram is linked there to keep the page concise.

All values come from saved notebook results. No model was retrained to create these figures. AP means average precision. Observations are exploratory; no causal or measured energy-saving claims are made.

Sources: the original AAB notebook, particularly cells 27, 99–106, 114–119, 131, 138–150 and 153–158; final result CSVs under results/tables. The 40%/60% labels are frozen validation recall targets, not equal realized test recall.

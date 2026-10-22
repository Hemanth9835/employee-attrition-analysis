# Processed data catalog

Predictor CSVs only; match each file to data/splits/y_<split>.csv by row_id.

| Variant | Inputs | Intended experiment |
|---|---:|---|
| scaled_full_onehot | 43 | Initial logistic baseline/tuning |
| unscaled_full_onehot | 43 | Initial decision-tree and random-forest baselines/tuning |
| scaled_reference | 39 | Reference LR and original-feature NN; final candidates |
| unscaled_reference | 39 | Later random-forest feature importance |
| interactions_scaled | 41 | LR dissatisfaction interactions |
| logs_scaled | 39 | LR log salary/overtime replacement |
| relative_added_scaled | 40 | LR relative salary added; original predictors retained |
| relative_no_job_scaled | 35 | LR relative salary with job level removed |
| relative_replace_both_scaled | 34 | LR relative salary with salary/job level removed |
| combined_engineered_scaled | 42 | Second NN case: originals + relative salary + two interactions |
| reduced_three_scaled | 35 | Experimental LR removing gender, work_life_balance and num_companies |

Each folder contains train.csv (3,000 rows), validation.csv (1,000) and test.csv (1,000). Each includes an extra row_id index column; input counts exclude that identifier.

For CV/search, use raw X_train with fresh fold-fitted preprocessing in the model pipeline. These training-wide transformed CSVs are suitable for fixed-split work only. Whole-feature ablations and optional subsets remain defined in Notebook 3 rather than exporting every temporary removal as another dataset.

The test files belong to the already completed frozen assessment. Do not tune or choose thresholds against them. Final model pipelines accept raw inputs rather than these transformed files.

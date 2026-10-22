# Download and prepare the data

Dataset: [Employee Attrition Early Warning Dataset on Kaggle](https://www.kaggle.com/datasets/yangjiu/employee-attrition-early-warning-dataset/data), by yangjiu, version 1.

1. Download and unzip the dataset from Kaggle. Sign in if Kaggle requests it.
2. Create `data/raw/` inside the cloned repository and place `employee_attrition_data.csv` there. Keep the original row order and values.
3. Open `01_EDA.ipynb` from the repository root. Its setup cell translates the six categorical columns and saves `data/raw/employee_attrition_data_english.csv`. Missing values remain missing.
4. Run `02_Feature_Engineering_and_Preprocessing.ipynb` to generate the fixed splits, all 11 preprocessing variants and their metadata.
5. Run Notebook 4 to reproduce the frozen final comparison. Notebook 3 is optional when reviewing final results and repeats training/searches when run.

The downloaded data, English copy, split CSVs and processed CSVs stay local and are ignored by Git. Their absence from GitHub is intentional. The source dataset retains its original terms and attribution.

The original download used for this study has SHA-256:

```text
a0e0a16e734c724eee9832e4cb35027973b46a7b0afbdcf4acdcff38821b7323
```

The translated English CSV has SHA-256:

```text
5570f114b577c6defac13e010bc8ca1b152f286322c1d9b1fa6f413c10114d90
```

Notebook 1 verifies the translated hash before saving. A mismatch means the download, translation, row order or CSV serialization differs from this study. Use the recorded dataset version and pinned environment rather than changing the frozen models or thresholds to accommodate a mismatch.

Notebook 2 saves unscaled tree inputs separately from scaled LR/NN inputs. See the [processed-data catalog](preprocessed/README.md). CV in Notebook 3 still learns preprocessing inside each training fold.

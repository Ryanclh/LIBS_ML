# LIBS-ML — Embedded Machine Learning for On-Device LIBS-Based Soil Analysis

An end-to-end machine learning pipeline for **Laser-Induced Breakdown Spectroscopy (LIBS)** soil analysis — from raw spectra all the way to **MCU-deployable C code**. Built as a Final Year Project: *Embedded Machine Learning for On-Device LIBS-Based Soil Nutrient Estimation*.

The pipeline handles two classification tasks:

- **Task A — NPK Nutrient Classification:** classify Nitrogen (N), Phosphorus (P), and Potassium (K) levels into qualitative categories (**Low / Medium / High**). Labels are generated semi-supervised via k-Means clustering on weighted, multi-line NIST atomic-emission peak areas.
- **Task B — Contamination Detection:** binary classification of soil samples (**Clean / Contaminated** — TPH, arsenic, heavy metals, etc.) using ground-truth labels.

The final phase exports trained LightGBM models to **plain C source files** (via [m2cgen](https://github.com/BayesWitnesses/m2cgen)) suitable for deployment on ARM Cortex-M microcontrollers, and validates the full methodology against an external dataset with laboratory-measured P/K concentrations.

## Pipeline Overview

```
Raw LIBS spectra (320 samples × 12,981 wavelengths, 197–974 nm)
        │
        ▼
┌──────────────────────┐  1. Polynomial (deg-5) baseline correction — removes
│  P1: Preprocessing   │     Bremsstrahlung continuum background
│                      │  2. Savitzky-Golay filter (w=15, o=3) — smooths noise,
│                      │     preserves peak shapes
│                      │  3. L2 normalization — compensates shot-to-shot
│                      │     laser energy fluctuations
│                      │  4. Label generation: k-Means on NPK peak areas (Task A)
│                      │     + ground-truth CSV labels (Task B)
└─────────┬────────────┘
          ▼
┌──────────────────────┐  Compare feature selection methods:
│  P2: Feature         │  • ANOVA F-test (filter-based)
│      Selection       │  • Recursive Feature Elimination (wrapper)
│                      │  • Random Forest Gini importance (embedded) → "Top-87"
│                      │  • NIST atomic-line domain-knowledge masks
│                      │  12,981 wavelengths → 87 features
└─────────┬────────────┘
          ▼
┌──────────────────────┐  Train and benchmark classifiers (5-fold OOF CV):
│  P3: Model           │  • Baseline: PLS-DA (Al-Juboori et al. replication, 73.01%)
│      Development     │  • Prep + RF / XGBoost / LightGBM (Gini top-87 features)
│                      │  • Autoencoder (AE) latent + XGBoost
│                      │  • Inception-style 1D CNN (TensorFlow)
│                      │  • Mamba SSM (PyTorch, optional GPU)
└─────────┬────────────┘
          ▼
┌──────────────────────┐  • Export trained models (joblib / .keras / .pth)
│  P4/P5: Deployment   │  • m2cgen: LightGBM → plain C code
│  & Benchmarking      │     (Prep_LGBM_N.c / _P.c / _K.c / _B.c)
│                      │  • Test-set evaluation (accuracy, macro F1)
│                      │  • External dataset validation with true P/K
│                      │     concentrations (monotonicity + Kruskal-Wallis)
│                      │  • Efficiency radar charts (Params, MACs, RAM, Flash)
└──────────────────────┘
```

## Models Compared

| Model | Input | Notes |
|---|---|---|
| PLS-DA | Full spectrum | Literature baseline (Al-Juboori et al., 73.01%) |
| Prep + Random Forest | Gini top-87 features | Classic ensemble baseline |
| Prep + XGBoost | Gini top-87 features | Gradient boosting |
| Prep + LightGBM | Gini top-87 features | **Selected for embedded export** (compact C output) |
| AE + XGBoost | Autoencoder latent | Deep feature extractor + boosting head |
| Inception 1D CNN | Full L2-normalized spectrum | Multi-scale convolutions (kernel 1/3/5 + max-pool) |
| Mamba SSM | Full spectrum | Selective state-space model (optional; requires GPU + `mamba-ssm`) |

## Validation Methodology

- **Preprocessing ablation:** raw spectra vs. sub-optimal pipeline (airPLS + Min-Max) vs. champion pipeline (Poly-5 + SG + L2), benchmarked with Random Forest against the 73.01% PLS-DA baseline.
- **5-fold out-of-fold cross-validation** with confusion matrices (Tasks A & B) and ROC/AUC curves (Task B).
- **External dataset validation (P & K):** predictions from models trained on the external LIBS dataset are checked for **monotonicity** (mean true concentration should increase Low → Medium → High) and statistical significance via the **Kruskal-Wallis test** against laboratory-measured extractable P/K values (mg/kg).
- **Autoencoder fidelity:** reconstruction-vs-original spectrum plots with NIST emission lines (N, P, K) highlighted, plus MSE quantification.
- **TinyML efficiency profiling:** radar charts comparing accuracy, parameter count, MACs, RAM, and flash footprint per model — the basis for choosing LightGBM for MCU export.

## Repository Structure

```
LIBS_ML/
├── P1_Preprocessing/            # Phase 1: spectral preprocessing & label generation
│   ├── libs_preprocessing.py    #   AsLS baseline correction, SG filter, L2 normalization
│   ├── generate_labels.py       #   NPK label generation via k-Means clustering
│   ├── baseline_comparison.py   #   Baseline correction method comparison
│   ├── normalization_comparison.py
│   ├── filter_comparison.py     #   Smoothing filter comparison (SG vs. alternatives)
│   ├── irish_soil_labels.csv    #   Ground-truth contamination labels
│   ├── irish_soil_preprocessed.npz   # Phase 1 output (consumed by Phase 2)
│   ├── Baseline Selection/      #   Comparison plots & recommendations
│   ├── Normalization Selection/ #   Comparison plots & recommendations
│   ├── Filter Selection/        #   Comparison plots & recommendations
│   └── test_preprocessing.py    #   Unit tests for preprocessing functions
│
├── P2_Feature Selection/        # Phase 2: feature selection under TinyML constraints
│   └── feature_selection.py     #   ANOVA vs. RFE vs. RF Gini comparison,
│                                 #   NIST-line validation, size/latency estimation
│
├── P3_Model Development/        # Phase 3: classifier training & benchmarking
│   ├── model_development.py     #   PLS-DA / RF / SVM training & evaluation
│   ├── model_dev2.py            #   Extended experiments
│   ├── trained_models/          #   Serialized best models (.pkl)
│   └── phase3_model_results/    #   Trained model artifacts (.joblib)
│
├── P4P5.ipynb                   # Phases 4 & 5: final implementation notebook
│                                 #   (model zoo training, 5-fold OOF CV, autoencoder,
│                                 #   external validation, m2cgen C export, radar charts)
│
├── libs_ml_pipeline.py          # Full end-to-end dual-task pipeline (Tasks A & B)
├── test.py / Test.ipynb         # Experiments and prototyping
└── Embeded ML Context Feeder.txt  # Research context: objectives, gaps, methodology
```

## Datasets

| Dataset | Contents | Used for |
|---|---|---|
| [Irish Soil Dataset](https://www.kaggle.com/datasets/ryancclh/irish-soil-dataset) | 320 samples × 12,981 wavelengths (197–974 nm), incl. Clay, Sandy, and contaminated variants (TPH, arsenic, heavy metals) | Tasks A & B (internal) |
| [LIBS Extra Dataset](https://www.kaggle.com/datasets/ryancclh/libs-extra-dataset) | External LIBS spectra + soil fertility data with lab-measured extractable P/K (mg/kg) | External validity check (P & K) |

> **Note:** The raw spectral CSV is **not included** in this repository. The `P4P5.ipynb` notebook is designed to run on [Kaggle](https://www.kaggle.com/) (GPU recommended for CNN / Mamba) and pulls the datasets from the Kaggle inputs above. Update the `filepath` arguments in the P1–P3 scripts for local runs.

## Requirements

- Python 3.9+
- Data / ML: `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`
- Boosting: `xgboost`, `lightgbm`
- Deep learning: `tensorflow` (Inception CNN, autoencoder)
- Optional: `torch` + `mamba-ssm` (Mamba SSM model; CUDA required)
- Embedded export: `m2cgen`

```bash
pip install numpy pandas scipy scikit-learn joblib xgboost lightgbm tensorflow m2cgen
```

## Usage

### Run the final notebook (recommended)

Open `P4P5.ipynb` in Kaggle (or Jupyter with the datasets downloaded locally) and run top-to-bottom:

1. **Training section** — preprocessing ablation, Task A & B model training (RF / XGB / LGBM / AE+XGB / CNN / Mamba), 5-fold OOF cross-validation, autoencoder training, m2cgen C export of the LightGBM models.
2. **Testing section** — test-set evaluation of all cached models, external P/K concentration validation, and efficiency radar charts.

Outputs: exported model artifacts (`FYP_Exported_Models_TaskA/`, `TaskB/`, `TaskA_Ext/` zipped for download) and portable C files (`Prep_LGBM_N.c`, `Prep_LGBM_P.c`, `Prep_LGBM_K.c`, `Prep_LGBM_B.c`) ready to be compiled into MCU firmware.

### Run phase by phase (local)

```bash
# Phase 1: preprocess raw spectra and generate labels
python "P1_Preprocessing/libs_preprocessing.py"

# Phase 2: compare feature selection methods
python "P2_Feature Selection/feature_selection.py"

# Phase 3: train and benchmark classifiers
python "P3_Model Development/model_development.py"

# Full dual-task pipeline (Tasks A & B) with LOSO diagnostics
python libs_ml_pipeline.py
```

## Methodology Notes

- **Champion preprocessing:** Polynomial (deg-5) baseline correction → Savitzky-Golay (w=15, o=3) → L2 normalization. This combination was empirically validated against raw and sub-optimal alternatives (e.g., airPLS + Min-Max) in the preprocessing ablation study.
- **Feature selection:** Random Forest Gini importance ("Top-87" wavelengths) tends to outperform univariate statistical tests on spectral data because it captures multivariate peak interactions; ANOVA serves as an efficient first-pass filter. Selected features are cross-checked against NIST atomic emission lines (N, P, K, Fe, Ca, Mg, Pb, Cd, Cr) for physical interpretability.
- **Pseudo-labels for NPK:** k-Means (k=3) on weighted sums of multiple NIST emission lines per element, with cluster centers sorted into Low / Medium / High.
- **Embedded export:** LightGBM was selected for MCU deployment because it offers the best trade-off between accuracy and flash/RAM footprint; m2cgen transpiles the trained model to dependency-free C.
- **Matrix-effect diagnostics:** Leave-One-Soil-Type-Out (LOSO) cross-validation in `libs_ml_pipeline.py` checks whether models learn spectroscopic matrix effects rather than memorizing soil types.

See `Embeded ML Context Feeder.txt` for the full research context, literature justification, and SMART objectives.

## Project Status

- [x] Phase 1 — Preprocessing (baseline, smoothing, normalization) with method comparisons
- [x] Phase 2 — Feature selection comparison (ANOVA / RFE / RF Gini → Top-87)
- [x] Phase 3 — Model development (PLS-DA baseline vs. RF / XGB / LGBM / AE+XGB / CNN / Mamba)
- [x] Phase 4 — Embedded export via m2cgen (LightGBM → C code)
- [x] Phase 5 — Validation & benchmarking (5-fold OOF CV, external P/K dataset, efficiency radar charts)

## Acknowledgements

- Irish Soil LIBS Dataset provided in collaboration with SETU (South East Technological University), Ireland.
- Baseline methodology informed by Al-Juboori et al.'s LIBS soil classification work.

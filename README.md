# Early Disease Risk Prediction Using PhysioNet Sepsis Data

A machine-learning project for **early sepsis risk prediction** using the **PhysioNet/Computing in Cardiology Challenge 2019 dataset**.

The project processes hourly patient clinical data, performs preprocessing and feature engineering, generates qSOFA and NEWS2-style clinical scores, trains multiple machine-learning models, evaluates their performance, and provides an interactive Streamlit dashboard.

The official PhysioNet Challenge dataset contains one pipe-delimited `.psv` file per patient, with each row representing approximately one hour of clinical data. The dataset contains 40,336 training subjects across training sets A and B.

---

## Project Overview

The objective of this project is to build an **early-warning system for sepsis risk**.

Instead of only identifying sepsis after it has occurred, the project uses previous patient measurements to estimate whether sepsis onset is likely within the next **6 hours**.

### Main stages

```text
PhysioNet Patient Data
        ↓
Data Loading
        ↓
Data Cleaning & Imputation
        ↓
Time-Series Alignment
        ↓
Rolling-Window Feature Engineering
        ↓
qSOFA / NEWS2-Style Scores
        ↓
Sepsis Risk Label Creation
        ↓
Patient-Level Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Explainability
        ↓
Streamlit Dashboard
```

---

## Dataset

This project uses the:

**PhysioNet/Computing in Cardiology Challenge 2019 — Early Prediction of Sepsis from Clinical Data**

The official dataset contains hourly clinical measurements and the `SepsisLabel` target. The challenge was designed around early prediction of sepsis from current and past clinical information.

### Dataset structure

The downloaded dataset is organized into two training sets:

```text
data/raw/physionet/
├── training_set A/
│   ├── patient_file_1.psv
│   ├── patient_file_2.psv
│   └── ...
│
└── training_set B/
    ├── patient_file_1.psv
    ├── patient_file_2.psv
    └── ...
```

Each `.psv` file contains hourly measurements for one patient.

Important fields include:

* `HR` — Heart Rate
* `O2Sat` — Oxygen Saturation
* `Temp` — Temperature
* `SBP` — Systolic Blood Pressure
* `MAP` — Mean Arterial Pressure
* `DBP` — Diastolic Blood Pressure
* `Resp` — Respiratory Rate
* `EtCO2`
* `BaseExcess`
* `HCO3`
* `FiO2`
* `pH`
* `PaCO2`
* `SaO2`
* `AST`
* `BUN`
* `Alkalinephos`
* `Calcium`
* `Chloride`
* `Creatinine`
* `Bilirubin_direct`
* `Glucose`
* `Lactate`
* `Magnesium`
* `Phosphate`
* `Potassium`
* `Bilirubin_total`
* `TroponinI`
* `Hct`
* `Hgb`
* `PTT`
* `WBC`
* `Fibrinogen`
* `Platelets`
* `Age`
* `Gender`
* `Unit1`
* `Unit2`
* `HospAdmTime`
* `ICULOS`
* `SepsisLabel`

The official dataset documentation states that each row represents a collection of measurements at the same hourly time point and that `SepsisLabel` indicates sepsis according to the challenge definition.

---

## Important Dataset Note

The PhysioNet dataset is **not included in this GitHub repository**.

The dataset is large and contains the original patient `.psv` files. These files are therefore excluded using `.gitignore`.

You should download the dataset separately from PhysioNet and place it inside:

```text
data/raw/physionet/
```

The project automatically searches recursively for `.psv` files, so `training_set A` and `training_set B` do not need to be manually merged.

---

# Project Structure

```text
disease_risk_prediction/
│
├── data/
│   ├── raw/
│   │   └── physionet/
│   │       ├── training_set A/
│   │       └── training_set B/
│   │
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluate.py
│   └── explainability.py
│
├── dashboard/
│   └── app.py
│
├── models/
│
├── outputs/
│
├── train.py
├── train_lstm.py
├── make_report_plots.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Matplotlib
* SHAP
* Streamlit
* PyTorch (for the optional LSTM model)

---

# Feature Engineering

The project converts the hourly clinical time-series data into machine-learning features.

For the selected physiological and laboratory variables, the following features are calculated:

* Mean
* Standard deviation
* Last observed value
* Trend / slope

The project also generates:

* qSOFA-style score
* NEWS2-style score
* ICU time information
* Hours to sepsis onset
* Binary prediction label

### Main model features

The current feature dataset contains **58 model features**, including rolling statistics and clinical scores.

The rolling-window approach allows the model to capture both:

1. The patient's current physiological state
2. The recent trend of the patient's condition

---

# Prediction Target

The first positive `SepsisLabel` for each patient is treated as the patient's sepsis onset.

The project creates a prediction window based on whether sepsis onset occurs within the following **6 hours**.

Conceptually:

```text
Patient clinical timeline

Hour 1 ───── Hour 10 ───── Hour 20 ───── Sepsis onset
                                      ↑
                              Prediction window
```

The purpose is to provide an early warning rather than simply detecting sepsis after the event.

The original PhysioNet Challenge itself was designed around early sepsis prediction using current and past data.

---

# Machine Learning Models

The project supports three main machine-learning models:

### 1. Logistic Regression

Provides a simple and interpretable baseline classification model.

### 2. Random Forest

Captures nonlinear relationships between physiological measurements and sepsis risk.

### 3. XGBoost

A gradient-boosted tree model used for more advanced nonlinear prediction.

---

# Patient-Level Train/Test Split

A **patient-level split** is used instead of randomly splitting individual rows.

This is important because the dataset contains multiple time points for the same patient.

If rows from the same patient were placed in both training and testing datasets, information from that patient could leak into the test set and produce overly optimistic results.

The project therefore separates patients before model training.

---

# Model Evaluation

Because sepsis prediction is a highly imbalanced classification problem, accuracy alone is not sufficient.

The project evaluates models using:

* **AUROC**
* **AUPRC**
* **Sensitivity / Recall**
* **Specificity**
* **True Positives**
* **False Positives**
* **False Negatives**
* **True Negatives**
* **Mean Lead Time**
* **Median Lead Time**

### Lead Time

Lead time measures how early a correct positive prediction occurs before the patient's sepsis onset.

This is particularly relevant for an early-warning system.

---

# Clinical Scores

## qSOFA-style Score

The project calculates a simplified qSOFA-style score using available measurements such as:

* Respiratory rate
* Systolic blood pressure

Mental-status information is not directly available in the selected feature set, so this should be described as a **qSOFA-style/simplified score**, not a complete clinical qSOFA implementation.

## NEWS2-style Score

A NEWS2-style score is calculated from available physiological measurements.

It is used as a clinical reference/baseline within the project.

The project should describe this as a **NEWS2-style implementation**, rather than claiming it reproduces every component of the official NEWS2 scoring system.

---

# Installation

Clone the repository:

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```powershell
cd disease_risk_prediction
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

# Add the PhysioNet Dataset

After downloading and extracting the dataset, place the two training folders inside:

```text
data/raw/physionet/
```

Expected structure:

```text
data/raw/physionet/
│
├── training_set A/
│   ├── *.psv
│   └── ...
│
└── training_set B/
    ├── *.psv
    └── ...
```

No manual merging of the `.psv` files is required.

---

# Run the Project

## Step 1 — Preprocess the dataset

From the project root:

```powershell
python -m src.preprocessing
```

This loads the `.psv` files, cleans the clinical data, handles missing values, aligns the patient timelines, and creates:

```text
data/processed/vitals_clean.csv
```

---

## Step 2 — Create machine-learning features

Run:

```powershell
python -m src.feature_engineering
```

This creates the rolling-window features and saves:

```text
data/processed/features.csv
```

---

## Step 3 — Train the models

Run:

```powershell
python train.py
```

The training pipeline:

1. Loads the engineered features
2. Splits patients into training and testing groups
3. Prepares model features
4. Scales the required features
5. Trains Logistic Regression
6. Trains Random Forest
7. Trains XGBoost when available
8. Evaluates model performance
9. Compares against the NEWS2-style baseline
10. Runs model explainability
11. Saves model artifacts
12. Saves test-window data for the dashboard

---

# Run the Dashboard

After training has completed:

```powershell
streamlit run dashboard/app.py
```

The Streamlit dashboard provides:

* Patient selection
* Current sepsis-risk prediction
* Risk percentage
* Risk trend
* qSOFA-style score
* NEWS2-style score
* ICU-hour timeline
* Feature information
* Patient timeline
* Cohort overview
* Alert threshold control

---

# Optional LSTM Model

An additional sequence-model implementation is available in:

```text
train_lstm.py
```

It can be run using:

```powershell
python train_lstm.py
```

The LSTM component is optional and is not required for the main machine-learning pipeline.

---

# Generate Report Plots

The project also includes:

```text
make_report_plots.py
```

Run:

```powershell
python make_report_plots.py
```

This can be used to generate visualizations for project documentation and reports.

---

# Generated Files

The following files/directories are generated during processing and training:

```text
data/processed/
outputs/
models/*.joblib
models/*.pkl
models/*.pickle
```

These files are excluded from GitHub where appropriate using `.gitignore`.

The raw PhysioNet data is also excluded:

```text
data/raw/
*.psv
```

This keeps the GitHub repository focused on the source code rather than large dataset and model files.

---

# GitHub Repository

The repository is intended to contain:

```text
Source Code
Documentation
Requirements
Project Configuration
```

It should **not** contain:

```text
PhysioNet .psv files
Large processed CSV files
Virtual environment
Generated model binaries
Generated outputs
```

---

# SRS Functionality Mapping

| SRS Functionality          | Implementation                            |
| -------------------------- | ----------------------------------------- |
| 1. Data Ingestion          | `src/preprocessing.py`                    |
| 2. Data Preprocessing      | `src/preprocessing.py`                    |
| 3. Feature Engineering     | `src/feature_engineering.py`              |
| 4. Risk Prediction         | `src/models.py`, `train.py`               |
| 5. Explainability          | `src/explainability.py`                   |
| 6. Dashboard Visualization | `dashboard/app.py`                        |
| 7. Alerting                | `dashboard/app.py`                        |
| 8. Model Evaluation        | `src/evaluate.py`, `make_report_plots.py` |

---

# Current Dataset Processing

The current implementation processes approximately:

```text
Patients:        40,336
Feature windows: 1,350,530
Positive windows: 12,505
Positive rate:   ~0.93%
Model features:  58
```

These figures correspond to the current processed dataset used during development and may change if the preprocessing or feature-engineering pipeline is modified.

---

# Important Project Considerations

## Class Imbalance

Sepsis-positive windows represent a small fraction of all prediction windows.

Therefore, metrics such as AUROC, AUPRC, sensitivity, and specificity are more informative than simply reporting accuracy.

## Missing Clinical Measurements

Clinical time-series data contains many missing observations.

The preprocessing pipeline handles missing values using patient-level forward/backward filling and fallback imputation.

## Data Leakage

Patient-level splitting is deliberately used to prevent the same patient's observations from appearing in both training and testing datasets.

## Early Warning

The purpose of the project is early risk identification. A prediction should use information available up to the current time rather than future patient information.

---

# Limitations

This project is an academic implementation and has several limitations:

* The model is not clinically validated.
* The qSOFA implementation is simplified.
* The NEWS2 implementation is NEWS2-style rather than a complete clinical implementation.
* Imputation can influence model predictions.
* Model performance may vary with preprocessing and feature-engineering choices.
* The project has not been validated prospectively in a hospital environment.
* High model performance on this dataset should not be interpreted as evidence of clinical readiness.

---

# Clinical Disclaimer

**This project is for academic and research purposes only.**

It is **not a medical diagnostic system** and should not be used to make real-world clinical decisions.

The predictions generated by this project should not replace qualified medical professionals, clinical assessment, laboratory testing, or established hospital protocols.

---

# PhysioNet Dataset Reference

The dataset is from the **PhysioNet/Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data**. The official resource describes the dataset, challenge objective, hourly clinical records, and `SepsisLabel` definition.

Official resource:

[PhysioNet 2019 Sepsis Challenge Dataset](https://physionet.org/content/challenge-2019/1.0.0/?utm_source=chatgpt.com)

When using the dataset, refer to the original PhysioNet Challenge publication and dataset citation provided by PhysioNet.

---

# Author

**Vibhanshu Solanki And Piyush Solanki**

Data Science / Machine Learning Project

---

## License

This project is intended for academic and educational use.

The PhysioNet dataset is subject to its own license and access terms. Refer to the official PhysioNet resource for the dataset's licensing information.

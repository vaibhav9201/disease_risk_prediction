# Early Disease Risk Prediction Using Wearable and EHR Data

Final-year project implementation, built to match:
- **Phase-1 Project Proposal** (Motivation, Conceptual Design, Innovativeness, Utility, Ideation)
- **Phase-2 SRS** (all 8 functionalities in section 2.1, the use-case model in 2.2, and the evaluation metrics in section 2.4)

## What this does

Predicts whether a patient will deteriorate into a critical condition
(e.g., sepsis-like decline) within the next few days, using a rolling
window of wearable-style vitals (heart rate, SpO2, respiratory rate, BP,
temperature) plus EHR-style fields (medication adherence, symptom
severity). Outputs a risk score, an explanation of what's driving it, and
raises an alert when risk crosses a threshold — shown on a dashboard.

## Project structure

```
disease_risk_prediction/
├── data/
│   ├── generate_synthetic_data.py   # creates the sample dataset (see note below)
│   ├── raw/                         # raw input CSV lives here
│   └── processed/                   # cleaned data + engineered features
├── src/
│   ├── preprocessing.py             # Functionality 2: cleaning, missing values, alignment
│   ├── feature_engineering.py       # Functionality 3: windows, stats, qSOFA/NEWS2
│   ├── models.py                    # Functionality 4: LogReg / RandomForest / XGBoost
│   ├── evaluate.py                  # Functionality 8: AUROC, AUPRC, sensitivity, lead-time
│   └── explainability.py            # Functionality 5: SHAP (+ fallback)
├── dashboard/
│   └── app.py                       # Functionality 6 & 7: dashboard + alerting (Streamlit)
├── train.py                         # runs the full pipeline end-to-end
├── train_lstm.py                    # optional advanced sequence model (PyTorch)
├── make_report_plots.py             # ROC / PR / feature-importance charts for your report
├── requirements.txt
└── models/, outputs/                # generated after running train.py
```

## How to run

```bash
pip install -r requirements.txt

# 1. Train everything (generates sample data automatically on first run)
python train.py

# 2. (Optional) generate report figures
python make_report_plots.py

# 3. (Optional) train the advanced LSTM model
python train_lstm.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

`train.py` will automatically:
1. Generate a synthetic sample dataset (see note below) if no raw data is present
2. Clean and align it into a fixed daily grid
3. Build sliding-window features + qSOFA/NEWS2 clinical scores
4. Train Logistic Regression and Random Forest (always), and XGBoost (if installed)
5. Evaluate all models — AUROC, AUPRC, sensitivity, specificity, and **mean lead-time**
6. Benchmark against a NEWS2-only clinical baseline (this is your "Innovativeness" evidence)
7. Run SHAP explainability (or a permutation-importance fallback if `shap` isn't installed)
8. Save the best model + a `test_windows.csv` for the dashboard

Everything runs with just `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`
— no extra installs needed for the core pipeline. `xgboost`, `shap`, `torch`,
and `streamlit` are optional upgrades (the code detects and uses them if present).

## ⚠️ Important: using a REAL dataset for your submission

**`generate_synthetic_data.py` creates fabricated sample data** so the whole
pipeline runs immediately without needing dataset access approval. **This is
for development/demo only — do not present synthetic data as your real
result to your supervisor.** Before your final submission:

1. Get real data from one of the sources named in your SRS (section 1.2):
   - **MIMIC-III / MIMIC-IV** (physionet.org) — requires completing the free
     CITI "Data or Specimens Only Research" training, then requesting
     credentialed access (takes ~1–2 weeks approval).
   - **PhysioNet 2019 Sepsis Challenge dataset** — openly downloadable, no
     credentialing needed, and already structured for this exact task.
   - A Kaggle vitals dataset with a similar schema (patient_id, day, vitals,
     outcome label) also works with minimal changes to `preprocessing.py`.
2. Place the file at `data/raw/patient_vitals.csv` with the same column
   names used here (see the docstring in `generate_synthetic_data.py`), or
   adjust `VITAL_COLS` / column names in `src/preprocessing.py` and
   `src/feature_engineering.py` to match your real file's schema.
3. Delete `data/processed/*.csv` and re-run `python train.py`.

## Mapping to your SRS functionalities

| SRS Functionality | Implemented in |
|---|---|
| 1. Data Ingestion | `data/generate_synthetic_data.py`, `src/preprocessing.load_raw` |
| 2. Data Preprocessing | `src/preprocessing.py` |
| 3. Feature Engineering | `src/feature_engineering.py` |
| 4. Risk Prediction | `src/models.py`, `train_lstm.py` |
| 5. Explainability | `src/explainability.py` |
| 6. Dashboard Visualization | `dashboard/app.py` |
| 7. Alerting | `dashboard/app.py` (threshold slider) |
| 8. Model Evaluation | `src/evaluate.py`, `make_report_plots.py` |

## Notes for your viva / defense

- The **patient-level train/test split** (`patient_level_split` in
  `src/models.py`) is deliberate and important to mention: splitting by row
  instead of by patient would leak information and inflate your metrics.
- **Lead-time** (`src/evaluate.py`) is the clinically meaningful metric —
  be ready to explain it: "how many days before the actual event did we
  correctly flag the patient?"
- The **NEWS2-only baseline comparison** is your strongest "innovativeness"
  talking point: it directly shows the ML model beating the standard
  clinical scoring system your committee will already be familiar with.
- Be prepared to explain **why results look very strong on the synthetic
  data** (the simulated deterioration pattern is easier to learn than real
  noisy clinical data) — set that expectation now so it isn't a surprise
  when real data gives lower (more realistic) numbers.

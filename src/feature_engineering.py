"""
feature_engineering.py
----------------------
Optimized feature engineering for PhysioNet Sepsis Dataset.
"""

import os
import numpy as np
import pandas as pd

VITAL_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "EtCO2", "Glucose", "Lactate", "WBC", "Hgb",
    "Platelets", "Creatinine"
]

WINDOW_SIZE = 6
HORIZON = 6


def add_scores(df):
    """Add qSOFA and NEWS2-style scores."""

    df["qsofa"] = (
        (df["Resp"] >= 22).astype(int) +
        (df["SBP"] <= 100).astype(int)
    )

    hr = df["HR"]
    rr = df["Resp"]
    spo2 = df["O2Sat"]
    temp = df["Temp"]
    sbp = df["SBP"]

    score = np.zeros(len(df), dtype=np.int8)

    score += np.select(
        [hr <= 40, hr >= 131, (hr >= 111) & (hr <= 130),
         (hr >= 41) & (hr <= 50), (hr >= 91) & (hr <= 110)],
        [3, 3, 2, 1, 1],
        default=0
    )

    score += np.select(
        [rr <= 8, rr >= 25, (rr >= 21) & (rr <= 24),
         (rr >= 9) & (rr <= 11)],
        [3, 3, 2, 1],
        default=0
    )

    score += np.select(
        [spo2 <= 91, (spo2 >= 92) & (spo2 <= 93),
         (spo2 >= 94) & (spo2 <= 95)],
        [3, 2, 1],
        default=0
    )

    score += np.select(
        [temp <= 35, temp >= 39.1,
         (temp > 35) & (temp <= 36),
         (temp >= 38.1) & (temp <= 39)],
        [3, 3, 1, 1],
        default=0
    )

    score += np.select(
        [sbp <= 90, (sbp >= 91) & (sbp <= 100),
         (sbp >= 101) & (sbp <= 110)],
        [3, 2, 1],
        default=0
    )

    df["news2"] = score

    return df


def build_features(df):

    print("Preparing data...")

    df = df.sort_values(
        ["patient_id", "ICULOS"]
    ).reset_index(drop=True)

    df = add_scores(df)

    # --------------------------------------------------
    # Rolling features
    # --------------------------------------------------

    print("Creating rolling features...")

    grouped = df.groupby("patient_id", sort=False)

    result = df[["patient_id", "ICULOS"]].copy()

    for col in VITAL_COLS:

        if col not in df.columns:
            continue

        print(f"Processing {col}...")

        g = grouped[col]

        result[f"{col}_mean"] = (
            g.rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE)
             .mean()
             .reset_index(level=0, drop=True)
             .values
        )

        result[f"{col}_std"] = (
            g.rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE)
             .std()
             .reset_index(level=0, drop=True)
             .values
        )

        result[f"{col}_last"] = df[col].values

        # Approximate slope using first and last value
        first = (
            g.shift(WINDOW_SIZE - 1)
             .values
        )

        last = df[col].values

        result[f"{col}_slope"] = (
            (last - first) / (WINDOW_SIZE - 1)
        )

    # Scores
    result["qsofa"] = df["qsofa"].values
    result["news2"] = df["news2"].values

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    print("Creating sepsis prediction labels...")

    # First sepsis hour for every patient
    onset = (
        df.loc[df["SepsisLabel"] == 1]
        .groupby("patient_id")["ICULOS"]
        .min()
        .rename("onset_hour")
    )

    result = result.join(onset, on="patient_id")

    # Prediction target:
    # Sepsis occurs within next 6 hours
    time_to_onset = result["onset_hour"] - result["ICULOS"]

    result["label"] = (
        time_to_onset.between(1, HORIZON)
    ).astype(int)

    result["hours_to_onset"] = np.where(
        result["label"] == 1,
        time_to_onset,
        np.nan
    )

    # Remove incomplete first 5 hours
    result = result.dropna(
        subset=[f"{VITAL_COLS[0]}_mean"]
    )

    result = result.drop(columns=["onset_hour"])

    return result.reset_index(drop=True)


def run_feature_engineering(
    in_path="data/processed/vitals_clean.csv",
    out_path="data/processed/features.csv"
):

    print("\nLoading preprocessed data...")

    df = pd.read_csv(in_path)

    print(
        f"Loaded {len(df):,} rows "
        f"and {df['patient_id'].nunique():,} patients."
    )

    features = build_features(df)

    if features.empty:
        raise ValueError(
            "No features were created. Check the preprocessing output."
        )

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True
    )

    features.to_csv(
        out_path,
        index=False
    )

    print("\nFeature engineering completed.")
    print(f"Feature rows: {len(features):,}")
    print(f"Patients: {features['patient_id'].nunique():,}")
    print(f"Positive windows: {features['label'].sum():,}")
    print(f"Positive rate: {features['label'].mean():.4f}")
    print(f"Saved → {out_path}")

    return features


if __name__ == "__main__":
    run_feature_engineering()
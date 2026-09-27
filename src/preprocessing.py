"""
preprocessing.py
----------------
Preprocessing pipeline for PhysioNet Sepsis Dataset.

Steps:
1. Load all .psv patient files
2. Add patient_id
3. Clean invalid values
4. Handle missing values
5. Keep ICU-hour information
6. Save a combined processed CSV
"""

import os
import glob
import numpy as np
import pandas as pd


# PhysioNet features used by the model
VITAL_COLS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "Glucose",
    "Lactate",
    "WBC",
    "Hgb",
    "Platelets",
    "Creatinine",
]

# Columns that must be preserved
ID_COLS = [
    "patient_id",
    "ICULOS",
    "Age",
    "Gender",
    "HospAdmTime",
    "SepsisLabel",
]


def load_raw(
    data_dir="data/raw/physionet"
):
    """
    Load all PhysioNet .psv files and combine them into one DataFrame.
    """

    files = glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True)

    if not files:
        raise FileNotFoundError(
            f"No .psv files found in: {data_dir}\n"
            "Put the PhysioNet patient files inside this folder."
        )

    print(f"Found {len(files)} PhysioNet patient files.")

    frames = []

    for i, file_path in enumerate(files, start=1):

        try:
            df = pd.read_csv(file_path, sep="|")

            # Patient ID from filename
            patient_id = os.path.splitext(
                os.path.basename(file_path)
            )[0]

            df["patient_id"] = patient_id

            frames.append(df)

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

        if i % 1000 == 0:
            print(f"Loaded {i}/{len(files)} files...")

    if not frames:
        raise ValueError("No valid PhysioNet files could be loaded.")

    data = pd.concat(frames, ignore_index=True)

    print(
        f"Combined dataset: {len(data):,} rows, "
        f"{data['patient_id'].nunique():,} patients"
    )

    return data


def clean_and_impute(df):
    """
    Clean PhysioNet data and handle missing values.

    Missing medical measurements are first forward-filled
    within each patient and then backward-filled.
    Remaining missing values are filled using training-data-style
    population medians.
    """

    df = df.copy()

    # Sort chronologically for every patient
    df = df.sort_values(
        ["patient_id", "ICULOS"]
    ).reset_index(drop=True)

    # Convert numerical columns to numeric
    numeric_cols = [
        col for col in df.columns
        if col not in ["patient_id"]
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Physiologically reasonable ranges
    clip_ranges = {
        "HR": (20, 250),
        "O2Sat": (50, 100),
        "Temp": (30, 45),
        "SBP": (40, 250),
        "MAP": (30, 200),
        "DBP": (20, 150),
        "Resp": (4, 70),
        "EtCO2": (10, 100),
        "Glucose": (20, 1000),
        "Lactate": (0, 30),
        "WBC": (0.1, 100),
        "Hgb": (2, 25),
        "Platelets": (1, 1000),
        "Creatinine": (0, 20),
    }

    for col, (low, high) in clip_ranges.items():

        if col in df.columns:
            df[col] = df[col].clip(
                lower=low,
                upper=high
            )

    # Forward-fill and backward-fill within patient
    existing_vitals = [
        col for col in VITAL_COLS
        if col in df.columns
    ]

    if existing_vitals:

        df[existing_vitals] = (
            df.groupby("patient_id")[existing_vitals]
            .transform(
                lambda x: x.ffill().bfill()
            )
        )

        # Remaining missing values → population median
        for col in existing_vitals:

            median_value = df[col].median()

            if pd.isna(median_value):
                median_value = 0

            df[col] = df[col].fillna(median_value)

    # Age and Gender
    if "Age" in df.columns:
        df["Age"] = (
            df.groupby("patient_id")["Age"]
            .transform(lambda x: x.ffill().bfill())
        )

        df["Age"] = df["Age"].fillna(
            df["Age"].median()
        )

    if "Gender" in df.columns:
        df["Gender"] = (
            df.groupby("patient_id")["Gender"]
            .transform(lambda x: x.ffill().bfill())
        )

        df["Gender"] = df["Gender"].fillna(
            df["Gender"].mode()[0]
            if not df["Gender"].mode().empty
            else 0
        )

    # Make sure SepsisLabel exists
    if "SepsisLabel" not in df.columns:
        raise ValueError(
            "SepsisLabel column was not found in the PhysioNet dataset."
        )

    df["SepsisLabel"] = (
        df["SepsisLabel"]
        .fillna(0)
        .astype(int)
    )

    return df


def align_to_fixed_grid(df):
    """
    PhysioNet data is already hourly through ICULOS.

    Instead of creating artificial days, keep the original ICU-hour
    timeline and ensure each patient's records are sorted correctly.
    """

    df = df.copy()

    df = df.sort_values(
        ["patient_id", "ICULOS"]
    ).reset_index(drop=True)

    # Remove duplicate patient/hour combinations
    df = df.drop_duplicates(
        subset=["patient_id", "ICULOS"],
        keep="first"
    )

    return df


def run_preprocessing(
    raw_dir="data/raw/physionet",
    out_path="data/processed/vitals_clean.csv"
):
    """
    Complete preprocessing pipeline.
    """

    print("\nLoading PhysioNet dataset...")

    df = load_raw(raw_dir)

    print("\nCleaning and imputing data...")

    df = clean_and_impute(df)

    print("\nAligning ICU time grid...")

    df = align_to_fixed_grid(df)

    # Create output directory
    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True
    )

    # Save processed data
    df.to_csv(
        out_path,
        index=False
    )

    remaining_missing = (
        df[VITAL_COLS]
        .isna()
        .sum()
        .sum()
    )

    print("\nPreprocessing completed.")
    print(
        f"Rows: {len(df):,}"
    )
    print(
        f"Patients: {df['patient_id'].nunique():,}"
    )
    print(
        f"Remaining vital missing values: "
        f"{remaining_missing:,}"
    )
    print(
        f"Sepsis-positive rows: "
        f"{df['SepsisLabel'].sum():,}"
    )
    print(
        f"Saved → {out_path}"
    )

    return df


if __name__ == "__main__":
    run_preprocessing()
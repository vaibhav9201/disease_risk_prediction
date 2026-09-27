"""
models.py
----------
Implements SRS Functionality 4 (Risk Prediction):
"The system shall predict a disease risk probability within a defined
prediction horizon using trained ML/DL models."

Staged approach (per the project proposal's Conceptual Design):
  1. Baseline: Logistic Regression (always available, sklearn)
  2. Stronger baseline: Random Forest (always available, sklearn)
  3. Advanced: XGBoost (optional — used if installed)

Class imbalance (per SRS 2.3 General Constraints) is handled via
class_weight="balanced" / scale_pos_weight, since disease-onset windows are
a small minority of all windows.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS_SUFFIXES = ("_mean", "_std", "_last", "_slope")


def get_feature_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.endswith(FEATURE_COLS_SUFFIXES)]
    cols += ["qsofa", "news2"]
    return cols


def patient_level_split(df: pd.DataFrame, test_size=0.25, random_state=42):
    """Split patients without allowing patient leakage."""

    # Convert to normal NumPy array
    patient_ids = df["patient_id"].astype(str).unique().tolist()

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state
    )

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    train_df = df[
        df["patient_id"].astype(str).isin(train_ids)
    ].copy()

    test_df = df[
        df["patient_id"].astype(str).isin(test_ids)
    ].copy()

    return train_df, test_df


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Optional advanced model. Returns None if xgboost isn't installed so
    the rest of the pipeline still runs (pip install xgboost to enable)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[models.py] xgboost not installed — skipping. "
              "Run: pip install xgboost")
        return None

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def prepare_xy(df: pd.DataFrame, feature_cols, scaler: StandardScaler = None, fit_scaler=False):
    X = df[feature_cols].values
    y = df["label"].values
    if fit_scaler:
        scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, y, scaler

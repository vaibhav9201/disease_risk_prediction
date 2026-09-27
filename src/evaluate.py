"""
evaluate.py
-----------
Evaluation functions for the PhysioNet Sepsis Prediction project.

Metrics:
- AUROC
- AUPRC
- Sensitivity / Recall
- Specificity
- Confusion matrix
- Mean lead-time in hours
- NEWS2-style clinical baseline
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    test_df: pd.DataFrame,
    threshold=0.5
):
    """
    Evaluate a trained sepsis prediction model.
    """

    # Probability of sepsis
    y_proba = model.predict_proba(X_test)[:, 1]

    # Convert probability to prediction
    y_pred = (y_proba >= threshold).astype(int)

    # --------------------------------------------------
    # AUROC
    # --------------------------------------------------

    if len(np.unique(y_test)) > 1:
        auroc = roc_auc_score(
            y_test,
            y_proba
        )
    else:
        auroc = float("nan")

    # --------------------------------------------------
    # AUPRC
    # --------------------------------------------------

    if len(np.unique(y_test)) > 1:
        auprc = average_precision_score(
            y_test,
            y_proba
        )
    else:
        auprc = float("nan")

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # --------------------------------------------------
    # Sensitivity
    # --------------------------------------------------

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )

    # --------------------------------------------------
    # Specificity
    # --------------------------------------------------

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    # --------------------------------------------------
    # Lead time
    # --------------------------------------------------
    # For correctly predicted positive windows,
    # calculate how many hours before sepsis onset
    # the model raised the alert.

    tp_mask = (
        (y_test == 1) &
        (y_pred == 1)
    )

    if "hours_to_onset" in test_df.columns:

        lead_times = (
            test_df.loc[
                tp_mask,
                "hours_to_onset"
            ]
            .dropna()
            .values
        )

    else:
        lead_times = np.array([])

    if len(lead_times) > 0:

        mean_lead_time = float(
            np.mean(lead_times)
        )

        median_lead_time = float(
            np.median(lead_times)
        )

    else:

        mean_lead_time = float("nan")
        median_lead_time = float("nan")

    # --------------------------------------------------
    # Return results
    # --------------------------------------------------

    return {

        "AUROC": (
            round(auroc, 3)
            if not np.isnan(auroc)
            else "N/A"
        ),

        "AUPRC": (
            round(auprc, 3)
            if not np.isnan(auprc)
            else "N/A"
        ),

        "Sensitivity (Recall)": (
            round(sensitivity, 3)
            if not np.isnan(sensitivity)
            else "N/A"
        ),

        "Specificity": (
            round(specificity, 3)
            if not np.isnan(specificity)
            else "N/A"
        ),

        "True Positives": int(tp),

        "False Positives": int(fp),

        "False Negatives": int(fn),

        "True Negatives": int(tn),

        "Mean Lead-Time (hours)": (
            round(mean_lead_time, 2)
            if not np.isnan(mean_lead_time)
            else "N/A"
        ),

        "Median Lead-Time (hours)": (
            round(median_lead_time, 2)
            if not np.isnan(median_lead_time)
            else "N/A"
        ),

        "n_true_positive_windows_with_leadtime": int(
            len(lead_times)
        ),
    }


def compare_to_news2_baseline(
    test_df: pd.DataFrame,
    news2_threshold=5
):
    """
    Compare the ML model with NEWS2-style score alone.
    """

    y_true = test_df["label"].values

    y_pred = (
        test_df["news2"].values >= news2_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    return {

        "NEWS2-only Sensitivity": (
            round(sensitivity, 3)
            if not np.isnan(sensitivity)
            else "N/A"
        ),

        "NEWS2-only Specificity": (
            round(specificity, 3)
            if not np.isnan(specificity)
            else "N/A"
        ),

        "NEWS2 threshold used": news2_threshold,

        "NEWS2 True Positives": int(tp),

        "NEWS2 False Positives": int(fp),

        "NEWS2 False Negatives": int(fn),

        "NEWS2 True Negatives": int(tn),
    }


def print_report(
    results: dict,
    title: str = "Evaluation Results"
):
    """
    Print evaluation results in readable format.
    """

    print(
        f"\n===== {title} ====="
    )

    for key, value in results.items():

        print(
            f"{key}: {value}"
        )
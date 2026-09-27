"""
make_report_plots.py
----------------------
Generates the figures typically needed for the project report / viva:
ROC curve, Precision-Recall curve, and top-feature importance chart.

Run AFTER train.py: python make_report_plots.py
"""

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.models import get_feature_columns


def main():
    with open("models/best_model_name.json") as f:
        meta = json.load(f)
    model = joblib.load(f"models/{meta['model']}.joblib")
    scaler = joblib.load("models/scaler.joblib")
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    test_df = pd.read_csv("outputs/test_windows.csv")
    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df["label"].values
    y_proba = model.predict_proba(X_test)[:, 1]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curve — {meta['model']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {meta['model']}")
    plt.tight_layout()
    plt.savefig("outputs/pr_curve.png", dpi=150)
    plt.close()

    # Top feature importance
    top_features = pd.read_csv("outputs/top_features.csv")
    value_col = "mean_abs_shap" if "mean_abs_shap" in top_features.columns else "importance_mean"
    plt.figure(figsize=(7, 5))
    plt.barh(top_features["feature"][::-1], top_features[value_col][::-1])
    plt.xlabel(value_col)
    plt.title("Top Features Driving Risk Predictions")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()

    print("Saved plots -> outputs/roc_curve.png, outputs/pr_curve.png, outputs/feature_importance.png")


if __name__ == "__main__":
    main()

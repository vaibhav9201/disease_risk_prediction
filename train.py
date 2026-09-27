"""
train.py
--------
Train and evaluate PhysioNet Sepsis prediction models.

Run:
    python train.py
"""

import os
import json
import joblib

from src.feature_engineering import run_feature_engineering
from src.models import (
    get_feature_columns,
    patient_level_split,
    prepare_xy,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.evaluate import (
    evaluate_model,
    compare_to_news2_baseline,
    print_report,
)
from src.explainability import (
    explain_with_shap,
    global_feature_importance,
    permutation_importance_fallback,
)

ALERT_THRESHOLD = 0.5


def main():

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # --------------------------------------------------
    # Step 1: Load existing features
    # --------------------------------------------------

    print("=" * 60)
    print("PHYSIONET SEPSIS RISK PREDICTION")
    print("=" * 60)

    print("\nStep 1/5: Loading feature dataset...")

    features_path = "data/processed/features.csv"

    if not os.path.exists(features_path):
        print("features.csv not found.")
        print("Running feature engineering...")
        features = run_feature_engineering()
    else:
        import pandas as pd

        features = pd.read_csv(features_path)

    print(f"Feature rows: {len(features):,}")
    print(f"Patients: {features['patient_id'].nunique():,}")
    print(f"Positive windows: {features['label'].sum():,}")
    print(f"Positive rate: {features['label'].mean():.4f}")

    # --------------------------------------------------
    # Step 2: Patient-level split
    # --------------------------------------------------

    print("\nStep 2/5: Splitting data by patient...")

    train_df, test_df = patient_level_split(
        features,
        test_size=0.25,
        random_state=42
    )

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")

    print(
        f"Training patients: {train_df['patient_id'].nunique():,}"
    )
    print(
        f"Testing patients:  {test_df['patient_id'].nunique():,}"
    )

    # --------------------------------------------------
    # Step 3: Prepare features
    # --------------------------------------------------

    print("\nStep 3/5: Preparing model features...")

    feature_cols = get_feature_columns(features)

    print(f"Number of features: {len(feature_cols)}")

    X_train, y_train, scaler = prepare_xy(
        train_df,
        feature_cols,
        fit_scaler=True
    )

    X_test, y_test, _ = prepare_xy(
        test_df,
        feature_cols,
        scaler=scaler
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    results_summary = {}

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("Training Logistic Regression...")
    print("-" * 60)

    log_reg = train_logistic_regression(
        X_train,
        y_train
    )

    res_lr = evaluate_model(
        log_reg,
        X_test,
        y_test,
        test_df,
        threshold=ALERT_THRESHOLD
    )

    print_report(
        res_lr,
        "Logistic Regression"
    )

    results_summary["LogisticRegression"] = res_lr

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("Training Random Forest...")
    print("-" * 60)

    rf = train_random_forest(
        X_train,
        y_train
    )

    res_rf = evaluate_model(
        rf,
        X_test,
        y_test,
        test_df,
        threshold=ALERT_THRESHOLD
    )

    print_report(
        res_rf,
        "Random Forest"
    )

    results_summary["RandomForest"] = res_rf

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("Training XGBoost...")
    print("-" * 60)

    xgb_model = train_xgboost(
        X_train,
        y_train
    )

    best_model = rf
    best_name = "RandomForest"

    if xgb_model is not None:

        res_xgb = evaluate_model(
            xgb_model,
            X_test,
            y_test,
            test_df,
            threshold=ALERT_THRESHOLD
        )

        print_report(
            res_xgb,
            "XGBoost"
        )

        results_summary["XGBoost"] = res_xgb

        if res_xgb["AUROC"] > res_rf["AUROC"]:
            best_model = xgb_model
            best_name = "XGBoost"

    # --------------------------------------------------
    # NEWS2 baseline
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("Evaluating NEWS2 baseline...")
    print("-" * 60)

    news2_res = compare_to_news2_baseline(
        test_df
    )

    print_report(
        news2_res,
        "NEWS2-style Clinical Baseline"
    )

    results_summary["NEWS2_baseline"] = news2_res

    print(
        f"\nSelected model for dashboard: {best_name}"
    )

    # --------------------------------------------------
    # Step 4: Explainability
    # --------------------------------------------------

    print("\nStep 4/5: Generating explainability...")

    try:

        explainer, shap_values = explain_with_shap(
            best_model,
            X_train,
            X_test,
            feature_cols
        )

        if shap_values is not None:

            top_features = global_feature_importance(
                shap_values,
                feature_cols
            )

        else:

            top_features = permutation_importance_fallback(
                best_model,
                X_test,
                y_test,
                feature_cols
            )

    except Exception as e:

        print(
            f"SHAP failed: {e}"
        )

        print(
            "Using permutation importance instead..."
        )

        top_features = permutation_importance_fallback(
            best_model,
            X_test,
            y_test,
            feature_cols
        )

    print("\nTop features:")
    print(top_features.to_string(index=False))

    top_features.to_csv(
        "outputs/top_features.csv",
        index=False
    )

    # --------------------------------------------------
    # Step 5: Save artifacts
    # --------------------------------------------------

    print("\nStep 5/5: Saving model artifacts...")

    joblib.dump(
        best_model,
        f"models/{best_name}.joblib"
    )

    joblib.dump(
        scaler,
        "models/scaler.joblib"
    )

    with open(
        "models/feature_cols.json",
        "w"
    ) as f:
        json.dump(
            feature_cols,
            f,
            indent=2
        )

    with open(
        "models/best_model_name.json",
        "w"
    ) as f:
        json.dump(
            {
                "model": best_name,
                "threshold": ALERT_THRESHOLD
            },
            f,
            indent=2
        )

    test_df.to_csv(
        "outputs/test_windows.csv",
        index=False
    )

    with open(
        "outputs/results_summary.json",
        "w"
    ) as f:
        json.dump(
            results_summary,
            f,
            indent=2,
            default=str
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nSaved files:")

    print(
        f"  models/{best_name}.joblib"
    )

    print(
        "  models/scaler.joblib"
    )

    print(
        "  models/feature_cols.json"
    )

    print(
        "  models/best_model_name.json"
    )

    print(
        "  outputs/test_windows.csv"
    )

    print(
        "  outputs/top_features.csv"
    )

    print(
        "  outputs/results_summary.json"
    )

    print(
        "\nNext step:"
    )

    print(
        "streamlit run dashboard/app.py"
    )


if __name__ == "__main__":
    main()
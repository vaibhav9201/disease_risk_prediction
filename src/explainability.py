"""
explainability.py
-------------------
Implements SRS Functionality 5 (Explainability):
"The system shall generate a SHAP-based explanation indicating which
features contributed most to each prediction."

Falls back to sklearn permutation importance if the `shap` package isn't
installed, so the pipeline still runs end-to-end offline; install shap
(`pip install shap`) to get true per-prediction SHAP values as specified
in the SRS.
"""

import numpy as np
import pandas as pd


def explain_with_shap(model, X_train, X_test, feature_names):
    try:
        import shap
    except ImportError:
        print("[explainability.py] shap not installed — falling back to "
              "permutation importance. Run: pip install shap")
        return None, None

    # TreeExplainer for tree models (RF/XGBoost), else generic Explainer
    model_type = type(model).__name__
    if "Forest" in model_type or "XGB" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):  # RF binary classifier returns [class0, class1]
            shap_values = shap_values[1]
    else:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test).values

    return explainer, shap_values


def global_feature_importance(shap_values, feature_names, top_n=10) -> pd.DataFrame:
    """Mean absolute SHAP value per feature = global importance ranking."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return imp.sort_values("mean_abs_shap", ascending=False).head(top_n)


def permutation_importance_fallback(model, X_test, y_test, feature_names, top_n=10) -> pd.DataFrame:
    """Used automatically when shap is unavailable. Measures how much AUROC
    drops when each feature is shuffled — a reasonable proxy for feature
    importance / explainability."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return imp.sort_values("importance_mean", ascending=False).head(top_n)


def explain_single_prediction(model, x_row, feature_names, background_X=None) -> pd.DataFrame:
    """Explains ONE patient's prediction — this is what the dashboard shows
    next to each risk score, per SRS 2.2 (dashboard shows 'explanation')."""
    try:
        import shap
        model_type = type(model).__name__
        if "Forest" in model_type or "XGB" in model_type:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x_row.reshape(1, -1))
            sv = sv[1][0] if isinstance(sv, list) else sv[0]
        else:
            explainer = shap.Explainer(model, background_X)
            sv = explainer(x_row.reshape(1, -1)).values[0]
        df = pd.DataFrame({"feature": feature_names, "contribution": sv})
        return df.reindex(df.contribution.abs().sort_values(ascending=False).index)
    except ImportError:
        # Fallback: for linear models use coef * value; for tree models use
        # feature_importances_ scaled by the (standardized) feature value.
        if hasattr(model, "coef_"):
            contrib = model.coef_[0] * x_row
        elif hasattr(model, "feature_importances_"):
            contrib = model.feature_importances_ * x_row
        else:
            contrib = np.zeros(len(feature_names))
        df = pd.DataFrame({"feature": feature_names, "contribution": contrib})
        return df.reindex(df.contribution.abs().sort_values(ascending=False).index)

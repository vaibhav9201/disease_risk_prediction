"""
dashboard/app.py
----------------
PhysioNet Sepsis Risk Prediction Dashboard

Run from project root:
    streamlit run dashboard/app.py
"""

import json
import sys
import os

import joblib
import pandas as pd
import streamlit as st

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.explainability import explain_single_prediction


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Disease Sepsis Risk Dashboard",
    page_icon="🏥",
    layout="wide"
)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

@st.cache_resource
def load_artifacts():

    with open("models/best_model_name.json") as f:
        meta = json.load(f)

    model = joblib.load(
        f"models/{meta['model']}.joblib"
    )

    scaler = joblib.load(
        "models/scaler.joblib"
    )

    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols, meta


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

@st.cache_data
def load_test_windows():

    return pd.read_csv(
        "outputs/test_windows.csv"
    )


# ---------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    st.title(
        "🏥 Disease Sepsis Risk Prediction Dashboard"
    )

    st.caption(
        "Early sepsis risk prediction using PhysioNet "
        "patient vital signs and clinical features"
    )

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if not os.path.exists(
        "models/best_model_name.json"
    ):

        st.error(
            "No trained model found. "
            "Run `python train.py` first."
        )

        return

    # -----------------------------------------------------
    # Load artifacts
    # -----------------------------------------------------

    model, scaler, feature_cols, meta = (
        load_artifacts()
    )

    test_df = load_test_windows().copy()

    # -----------------------------------------------------
    # Basic cleaning
    # -----------------------------------------------------

    test_df["patient_id"] = (
        test_df["patient_id"]
        .astype(str)
    )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "patient_id",
        "qsofa",
        "news2",
        "label"
    ]

    missing = [
        col
        for col in required_columns
        if col not in test_df.columns
    ]

    if missing:

        st.error(
            f"Missing columns in test_windows.csv: {missing}"
        )

        st.write(
            "Available columns:"
        )

        st.write(
            test_df.columns.tolist()
        )

        return

    # -----------------------------------------------------
    # Create timeline column
    # -----------------------------------------------------

    if "ICULOS" in test_df.columns:

        test_df["_timeline"] = pd.to_numeric(
            test_df["ICULOS"],
            errors="coerce"
        )

        timeline_name = "ICU Hour"

    else:

        # ICULOS is not present in test_windows.csv.
        # Create a sequential window number for each patient.

        test_df["_timeline"] = (
            test_df
            .groupby("patient_id")
            .cumcount()
        )

        timeline_name = "Window Number"

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    st.sidebar.header(
        "⚙️ Settings"
    )

    threshold = st.sidebar.slider(
        "Alert threshold",
        0.0,
        1.0,
        float(
            meta.get(
                "threshold",
                0.5
            )
        ),
        0.05
    )

    st.sidebar.caption(
        f"Model in use: **{meta['model']}**"
    )

    # -----------------------------------------------------
    # Patient selection
    # -----------------------------------------------------

    patient_ids = sorted(
        test_df["patient_id"]
        .unique()
    )

    selected_patient = st.sidebar.selectbox(
        "Select patient",
        patient_ids
    )

    # -----------------------------------------------------
    # Selected patient
    # -----------------------------------------------------

    patient_windows = (
        test_df[
            test_df["patient_id"]
            == selected_patient
        ]
        .sort_values("_timeline")
        .copy()
    )

    if patient_windows.empty:

        st.warning(
            "No data found for this patient."
        )

        return

    # -----------------------------------------------------
    # Prepare model input
    # -----------------------------------------------------

    missing_features = [
        col
        for col in feature_cols
        if col not in patient_windows.columns
    ]

    if missing_features:

        st.error(
            "Some model features are missing "
            "from test_windows.csv:"
        )

        st.write(
            missing_features
        )

        return

    X_patient = (
        patient_windows[
            feature_cols
        ].values
    )

    X = scaler.transform(
        X_patient
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    risk_scores = (
        model
        .predict_proba(X)[:, 1]
    )

    patient_windows[
        "risk_score"
    ] = risk_scores

    # -----------------------------------------------------
    # Latest prediction
    # -----------------------------------------------------

    latest = (
        patient_windows
        .iloc[-1]
    )

    latest_risk = float(
        latest["risk_score"]
    )

    # -----------------------------------------------------
    # Top metrics
    # -----------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Current Risk",
        f"{latest_risk:.2%}"
    )

    col2.metric(
        "qSOFA",
        int(latest["qsofa"])
    )

    col3.metric(
        "NEWS2",
        int(latest["news2"])
    )

    col4.metric(
        timeline_name,
        int(latest["_timeline"])
    )

    # -----------------------------------------------------
    # Alert
    # -----------------------------------------------------

    if latest_risk >= threshold:

        st.error(
            f"🚨 ALERT: Patient {selected_patient} "
            f"has a risk score of {latest_risk:.2%}, "
            f"which is above the configured "
            f"threshold of {threshold:.0%}."
        )

    else:

        st.success(
            f"✅ Patient {selected_patient} "
            f"is currently below the alert threshold."
        )

    # -----------------------------------------------------
    # Risk trend
    # -----------------------------------------------------

    st.subheader(
        "📈 Sepsis Risk Score Trend"
    )

    chart_data = (
        patient_windows[
            [
                "_timeline",
                "risk_score"
            ]
        ]
        .set_index("_timeline")
    )

    chart_data.index.name = (
        timeline_name
    )

    st.line_chart(
        chart_data
    )

    # -----------------------------------------------------
    # Clinical scores
    # -----------------------------------------------------

    st.subheader(
        "🩺 Clinical Risk Scores"
    )

    score_data = (
        patient_windows[
            [
                "_timeline",
                "qsofa",
                "news2"
            ]
        ]
        .set_index("_timeline")
    )

    score_data.index.name = (
        timeline_name
    )

    st.line_chart(
        score_data
    )

    # -----------------------------------------------------
    # Prediction explanation
    # -----------------------------------------------------

    st.subheader(
        "🔍 Why this prediction?"
    )

    x_row = X[-1]

    background = X[
        :min(50, len(X))
    ]

    try:

        explanation = (
            explain_single_prediction(
                model,
                x_row,
                feature_cols,
                background_X=background
            )
        )

        top_expl = (
            explanation
            .head(8)
        )

        st.bar_chart(
            top_expl
            .set_index("feature")[
                "contribution"
            ]
        )

        st.caption(
            "Positive values increase predicted "
            "risk; negative values decrease predicted risk."
        )

    except Exception as e:

        st.warning(
            f"Feature explanation unavailable: {e}"
        )

    # -----------------------------------------------------
    # Patient timeline
    # -----------------------------------------------------

    st.subheader(
        "👤 Patient Timeline"
    )

    timeline_columns = [
        "_timeline",
        "risk_score",
        "qsofa",
        "news2",
        "label"
    ]

    available_columns = [
        col
        for col in timeline_columns
        if col in patient_windows.columns
    ]

    timeline_display = (
        patient_windows[
            available_columns
        ]
        .rename(
            columns={
                "_timeline": timeline_name
            }
        )
    )

    st.dataframe(
        timeline_display,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Sepsis information
    # -----------------------------------------------------

    if "hours_to_onset" in patient_windows.columns:

        onset_values = (
            patient_windows[
                "hours_to_onset"
            ]
            .dropna()
        )

        if len(onset_values) > 0:

            st.info(
                f"Sepsis onset information is available "
                f"for {len(onset_values)} prediction windows."
            )

    # -----------------------------------------------------
    # Raw features
    # -----------------------------------------------------

    with st.expander(
        "📋 Show raw features"
    ):

        raw_display = (
            patient_windows
            .drop(
                columns=["_timeline"],
                errors="ignore"
            )
        )

        st.dataframe(
            raw_display,
            use_container_width=True
        )

    # -----------------------------------------------------
    # Cohort overview
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "👥 Cohort Overview"
    )

    latest_per_patient = (
        test_df
        .sort_values("_timeline")
        .groupby("patient_id")
        .tail(1)
        .copy()
    )

    X_all = scaler.transform(
        latest_per_patient[
            feature_cols
        ].values
    )

    latest_per_patient[
        "risk_score"
    ] = (
        model
        .predict_proba(X_all)[:, 1]
    )

    latest_per_patient[
        "alert"
    ] = (
        latest_per_patient[
            "risk_score"
        ] >= threshold
    )

    n_alerts = int(
        latest_per_patient[
            "alert"
        ].sum()
    )

    total_patients = len(
        latest_per_patient
    )

    st.write(
        f"**{n_alerts} / {total_patients}** "
        "patients are currently above "
        "the alert threshold."
    )

    overview_columns = [
        "patient_id",
        "risk_score",
        "qsofa",
        "news2",
        "alert"
    ]

    st.dataframe(
        latest_per_patient[
            overview_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .reset_index(drop=True),
        use_container_width=True
    )


# ---------------------------------------------------------
# Run dashboard
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
"""
generate_synthetic_data.py
---------------------------
Creates a sample patient-vitals dataset that mirrors the structure of public
wearable + EHR datasets referenced in the SRS (MIMIC-III/IV, PhysioNet Sepsis
Challenge, Kaggle 'EHR data' 30-day vitals sets).

This is SYNTHETIC data generated for local development/demo purposes so the
full pipeline (preprocessing -> features -> model -> explainability ->
dashboard) can run end-to-end without needing MIMIC credentialed access.

For your actual submission, replace this step by downloading the real
dataset (see README.md, section "Using a real dataset") and placing a CSV
with the same column names into data/raw/patient_vitals.csv.

Columns produced (one row = one patient-day):
    patient_id, day, bp_systolic, bp_diastolic, heart_rate, respiratory_rate,
    temperature, oxygen_saturation, med_adherence, symptom_severity,
    onset_day (NaN if patient never deteriorates), progressed_to_critical
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_PATIENTS = 400
N_DAYS = 30
POSITIVE_RATE = 0.25  # fraction of patients who deteriorate


def simulate_patient(patient_id, will_deteriorate):
    days = np.arange(1, N_DAYS + 1)

    # baseline "healthy" vitals with small daily noise
    hr = RNG.normal(78, 6, N_DAYS)
    rr = RNG.normal(16, 2, N_DAYS)
    spo2 = RNG.normal(97, 1, N_DAYS)
    temp = RNG.normal(98.4, 0.4, N_DAYS)
    bp_sys = RNG.normal(118, 8, N_DAYS)
    bp_dia = RNG.normal(76, 6, N_DAYS)
    med_adh = np.clip(RNG.normal(0.9, 0.08, N_DAYS), 0, 1)
    symptom = np.clip(RNG.normal(2, 1, N_DAYS), 0, 10)

    onset_day = np.nan
    if will_deteriorate:
        onset_day = int(RNG.integers(16, N_DAYS))  # event happens late in the window
        ramp_start = max(1, onset_day - 8)  # deterioration ramps up ~8 days before onset
        for d in range(ramp_start, onset_day + 1):
            idx = d - 1
            severity = (d - ramp_start) / max(1, (onset_day - ramp_start))  # 0 -> 1
            hr[idx] += 35 * severity + RNG.normal(0, 2)
            rr[idx] += 12 * severity + RNG.normal(0, 1)
            spo2[idx] -= 8 * severity + RNG.normal(0, 0.5)
            temp[idx] += 2.5 * severity + RNG.normal(0, 0.2)
            bp_sys[idx] -= 25 * severity + RNG.normal(0, 3)
            med_adh[idx] = np.clip(med_adh[idx] - 0.3 * severity, 0, 1)
            symptom[idx] = np.clip(symptom[idx] + 6 * severity, 0, 10)
        # after onset, vitals stay critical for the remaining days
        for d in range(onset_day + 1, N_DAYS + 1):
            idx = d - 1
            hr[idx] += 35 + RNG.normal(0, 3)
            rr[idx] += 12 + RNG.normal(0, 1.5)
            spo2[idx] -= 8 + RNG.normal(0, 0.7)
            temp[idx] += 2.5 + RNG.normal(0, 0.3)
            bp_sys[idx] -= 25 + RNG.normal(0, 4)
            symptom[idx] = np.clip(symptom[idx] + 6, 0, 10)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "day": days,
        "bp_systolic": np.round(bp_sys, 1),
        "bp_diastolic": np.round(bp_dia, 1),
        "heart_rate": np.round(hr, 1),
        "respiratory_rate": np.round(rr, 1),
        "temperature": np.round(temp, 1),
        "oxygen_saturation": np.round(np.clip(spo2, 70, 100), 1),
        "med_adherence": np.round(med_adh, 2),
        "symptom_severity": np.round(symptom, 1),
        "onset_day": onset_day,
        "progressed_to_critical": int(will_deteriorate),
    })
    return df


def generate(n_patients=N_PATIENTS, positive_rate=POSITIVE_RATE, seed=42):
    global RNG
    RNG = np.random.default_rng(seed)
    n_positive = int(n_patients * positive_rate)
    labels = np.array([True] * n_positive + [False] * (n_patients - n_positive))
    RNG.shuffle(labels)

    frames = [simulate_patient(pid + 1, bool(lab)) for pid, lab in enumerate(labels)]
    data = pd.concat(frames, ignore_index=True)

    # simulate a small amount of missingness / sensor dropout, as real wearables have
    missing_mask = RNG.random(len(data)) < 0.02
    cols_to_null = ["heart_rate", "respiratory_rate", "oxygen_saturation"]
    for c in cols_to_null:
        data.loc[missing_mask, c] = np.nan

    return data


if __name__ == "__main__":
    df = generate()
    out_path = "data/raw/patient_vitals.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows for {df['patient_id'].nunique()} patients -> {out_path}")
    print(f"Positive (deteriorated) patients: {df.groupby('patient_id')['progressed_to_critical'].first().sum()}")

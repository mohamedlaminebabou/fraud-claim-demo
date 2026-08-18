"""Data loading for the fraud demo.

Primary source: a real, publicly available auto-insurance claims dataset
(~1,000 claims, ~39 columns, a real 'fraud_reported' label), hosted on
GitHub and widely used across the ML community as the standard public
stand-in for this kind of demo — because actual insurers do not publish
their real fraud data (commercial confidentiality). See README.md for the
source, its known quirks, and why a couple of its columns are excluded.

Fallback: if the real dataset can't be fetched (no network, source moved,
rate limit), a seeded synthetic generator produces a similar-shaped
dataset so the app never breaks. Which one is active is always shown in
the UI — never a silent switch.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

REAL_DATA_URL = "https://raw.githubusercontent.com/mwitiderrick/insurancedata/master/insurance_claims.csv"

# Columns excluded from modelling, and why. Kept as a visible constant
# (not buried in a function body) because the "why" is part of the point.
EXCLUDED_COLUMNS = {
    "policy_number": "identifier, not a signal",
    "policy_bind_date": "raw date; superseded by months_as_customer",
    "insured_zip": "high-cardinality identifier-like field",
    "insured_hobbies": (
        "known data-leakage artifact in this public dataset: two categories "
        "('chess', 'cross-fit') separate fraud vs. not almost perfectly, which "
        "is not a plausible real fraud driver — a strong signal this field "
        "was synthetically assigned, not collected. Excluded on purpose."
    ),
    "incident_date": "raw date; superseded by incident_hour_of_the_day",
    "incident_location": "free-text / near-unique per row, not a usable feature as-is",
    "incident_city": "high-cardinality relative to ~1,000 rows",
    "auto_make": "high-cardinality relative to ~1,000 rows",
    "auto_model": "high-cardinality relative to ~1,000 rows",
    "auto_year": "weak, noisy signal at this sample size; excluded for scope",
}

NUMERIC_FEATURES = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital_gains", "capital_loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
]
CATEGORICAL_FEATURES = [
    "insured_sex", "insured_education_level", "insured_occupation", "insured_relationship",
    "policy_state", "policy_csl", "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "incident_state", "property_damage", "police_report_available",
]
TARGET = "is_fraud"


def _clean_real_claims(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [c.strip().replace("-", "_") for c in df.columns]

    if "fraud_reported" not in df.columns:
        raise ValueError("Expected column 'fraud_reported' not found — source format may have changed.")
    df[TARGET] = (df["fraud_reported"].astype(str).str.strip().str.upper() == "Y").astype(int)
    df = df.drop(columns=["fraud_reported"])

    # This dataset marks missing values with '?' instead of an empty cell.
    df = df.replace("?", np.nan)

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns] + [TARGET]
    df = df[keep]

    # Small, defensible imputation for a demo: median for numeric, an
    # explicit "Unknown" category for categorical — never silently dropped.
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df.reset_index(drop=True)


def load_real_claims(timeout: int = 10) -> pd.DataFrame:
    """Fetch and clean the real public dataset. Raises on any failure —
    callers decide how to handle that (see get_claims_with_status)."""
    import urllib.request

    req = urllib.request.Request(REAL_DATA_URL, headers={"User-Agent": "fraud-claim-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_bytes = resp.read()
    raw = pd.read_csv(io.BytesIO(raw_bytes))
    return _clean_real_claims(raw)


def generate_synthetic_claims(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Seeded synthetic fallback, shaped like the real dataset's feature
    schema so the rest of the pipeline (model, UI) doesn't need to branch
    on which source is active."""
    rng = np.random.default_rng(seed)

    incident_severities = ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
    incident_types = ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"]
    collision_types = ["Rear Collision", "Side Collision", "Front Collision", "Unknown"]
    authorities = ["Police", "Fire", "Ambulance", "None", "Other"]
    edu_levels = ["High School", "College", "Associate", "JD", "MD", "Masters", "PhD"]
    occupations = ["craft-repair", "sales", "exec-managerial", "tech-support", "machine-op-inspct", "other-service"]
    relationships = ["husband", "wife", "own-child", "unmarried", "not-in-family", "other-relative"]
    states = ["OH", "IN", "IL", "NY", "SC", "VA", "WV"]
    csl_levels = ["100/300", "250/500", "500/1000"]

    months_as_customer = rng.integers(1, 480, n)
    age = rng.integers(19, 65, n)
    policy_deductable = rng.choice([500, 1000, 2000], n)
    policy_annual_premium = rng.normal(1250, 250, n).clip(400, None)
    umbrella_limit = rng.choice([0, 0, 0, 1_000_000, 2_000_000, 5_000_000], n)
    capital_gains = rng.choice([0, 0, 0, 25000, 50000, 60000], n)
    capital_loss = -rng.choice([0, 0, 0, 30000, 45000, 60000], n)
    incident_hour = rng.integers(0, 24, n)
    n_vehicles = rng.integers(1, 4, n)
    bodily_injuries = rng.integers(0, 3, n)
    witnesses = rng.integers(0, 4, n)

    severity = rng.choice(incident_severities, n, p=[0.10, 0.35, 0.40, 0.15])
    sev_base = pd.Series(severity).map({
        "Trivial Damage": 2000, "Minor Damage": 8000, "Major Damage": 25000, "Total Loss": 45000,
    }).values
    total_claim = rng.gamma(2.0, sev_base / 2.0)
    injury_claim = total_claim * rng.uniform(0.05, 0.25, n)
    property_claim = total_claim * rng.uniform(0.1, 0.3, n)
    vehicle_claim = (total_claim - injury_claim - property_claim).clip(min=0)

    authorities_arr = rng.choice(authorities, n)
    property_damage_arr = rng.choice(["YES", "NO"], n)
    police_report_arr = rng.choice(["YES", "NO"], n)

    z = (
        -3.2
        + 2.6 * (severity == "Total Loss")
        + 1.5 * (severity == "Major Damage")
        + 1.1 * (bodily_injuries == 0)
        + 0.9 * (witnesses == 0)
        + 2.2e-5 * total_claim
        + 1.0 * (authorities_arr == "None")
        + 0.7 * (police_report_arr == "NO")
        - 0.35 * np.log1p(months_as_customer)
        + rng.normal(0, 0.55, n)
    )
    is_fraud = rng.binomial(1, 1 / (1 + np.exp(-z)))

    df = pd.DataFrame({
        "months_as_customer": months_as_customer, "age": age,
        "policy_deductable": policy_deductable,
        "policy_annual_premium": np.round(policy_annual_premium, 2),
        "umbrella_limit": umbrella_limit, "capital_gains": capital_gains, "capital_loss": capital_loss,
        "incident_hour_of_the_day": incident_hour, "number_of_vehicles_involved": n_vehicles,
        "bodily_injuries": bodily_injuries, "witnesses": witnesses,
        "total_claim_amount": np.round(total_claim, 2), "injury_claim": np.round(injury_claim, 2),
        "property_claim": np.round(property_claim, 2), "vehicle_claim": np.round(vehicle_claim, 2),
        "insured_sex": rng.choice(["MALE", "FEMALE"], n),
        "insured_education_level": rng.choice(edu_levels, n),
        "insured_occupation": rng.choice(occupations, n),
        "insured_relationship": rng.choice(relationships, n),
        "policy_state": rng.choice(states, n),
        "policy_csl": rng.choice(csl_levels, n),
        "incident_type": rng.choice(incident_types, n),
        "collision_type": rng.choice(collision_types, n),
        "incident_severity": severity,
        "authorities_contacted": authorities_arr,
        "incident_state": rng.choice(states, n),
        "property_damage": property_damage_arr,
        "police_report_available": police_report_arr,
        TARGET: is_fraud,
    })
    return df


def get_claims_with_status(prefer_real: bool = True) -> tuple[pd.DataFrame, dict]:
    """Single entry point the app uses. Returns (dataframe, status) where
    status always says exactly which source is live — the app must not
    hide this from the person using it."""
    if prefer_real:
        try:
            df = load_real_claims()
            return df, {
                "source": "real",
                "message": f"Real public dataset loaded — {len(df):,} claims.",
                "detail": REAL_DATA_URL,
            }
        except Exception as exc:  # noqa: BLE001 — intentionally broad: any
            # fetch/parse failure should fall back, not crash the app.
            df = generate_synthetic_claims()
            return df, {
                "source": "synthetic_fallback",
                "message": f"Could not load the real dataset ({exc}); using the synthetic fallback instead.",
                "detail": REAL_DATA_URL,
            }
    df = generate_synthetic_claims()
    return df, {"source": "synthetic", "message": f"Synthetic dataset — {len(df):,} claims.", "detail": None}


def to_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = pd.get_dummies(df[num + cat], columns=cat)
    y = df[TARGET]
    return X, y

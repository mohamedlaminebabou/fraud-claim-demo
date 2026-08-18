import numpy as np
import pandas as pd

from fraud_lib.data import generate_claims, to_model_matrix, CLAIM_TYPES


def test_shape_and_columns():
    df = generate_claims(n=500, seed=1)
    assert len(df) == 500
    expected_cols = {
        "claim_type", "claim_amount", "policy_value", "claim_to_policy_ratio",
        "policy_tenure_years", "prior_claims_count", "reported_delay_days",
        "weekend_incident", "has_prior_fraud_flag", "is_fraud",
    }
    assert expected_cols.issubset(set(df.columns))


def test_claim_types_are_valid():
    df = generate_claims(n=500, seed=1)
    assert set(df["claim_type"].unique()).issubset(set(CLAIM_TYPES))


def test_reproducibility_same_seed():
    df1 = generate_claims(n=200, seed=7)
    df2 = generate_claims(n=200, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seed_gives_different_data():
    df1 = generate_claims(n=200, seed=1)
    df2 = generate_claims(n=200, seed=2)
    assert not df1["claim_amount"].equals(df2["claim_amount"])


def test_fraud_label_is_binary_and_not_degenerate():
    df = generate_claims(n=4000, seed=42)
    assert set(df["is_fraud"].unique()) <= {0, 1}
    fraud_rate = df["is_fraud"].mean()
    # sanity band: the problem should neither be fraud-free nor mostly-fraud
    assert 0.03 < fraud_rate < 0.20


def test_no_negative_amounts_or_ratios():
    df = generate_claims(n=2000, seed=3)
    assert (df["claim_amount"] > 0).all()
    assert (df["policy_value"] > 0).all()
    assert (df["claim_to_policy_ratio"] > 0).all()


def test_to_model_matrix_one_hot_encodes_claim_type():
    df = generate_claims(n=300, seed=5)
    X, y = to_model_matrix(df)
    assert "claim_type" not in X.columns
    assert any(c.startswith("claim_type_") for c in X.columns)
    assert len(y) == len(df)
    assert set(y.unique()) <= {0, 1}

import numpy as np
import pandas as pd

from fraud_lib.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    generate_synthetic_claims,
    get_claims_with_status,
    to_model_matrix,
)


def test_synthetic_shape_and_columns():
    df = generate_synthetic_claims(n=500, seed=1)
    assert len(df) == 500
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    assert expected.issubset(set(df.columns))


def test_synthetic_reproducibility_same_seed():
    df1 = generate_synthetic_claims(n=200, seed=7)
    df2 = generate_synthetic_claims(n=200, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_different_seed_gives_different_data():
    df1 = generate_synthetic_claims(n=200, seed=1)
    df2 = generate_synthetic_claims(n=200, seed=2)
    assert not df1["total_claim_amount"].equals(df2["total_claim_amount"])


def test_synthetic_fraud_label_is_binary_and_not_degenerate():
    df = generate_synthetic_claims(n=1000, seed=42)
    assert set(df[TARGET].unique()) <= {0, 1}
    rate = df[TARGET].mean()
    assert 0.03 < rate < 0.30


def test_synthetic_no_negative_claim_amounts():
    df = generate_synthetic_claims(n=500, seed=3)
    for col in ["total_claim_amount", "injury_claim", "property_claim", "vehicle_claim"]:
        assert (df[col] >= 0).all()


def test_to_model_matrix_one_hot_encodes_categoricals():
    df = generate_synthetic_claims(n=300, seed=5)
    X, y = to_model_matrix(df)
    for cat_col in CATEGORICAL_FEATURES:
        assert cat_col not in X.columns
    assert any(c.startswith("incident_severity_") for c in X.columns)
    assert len(y) == len(df)
    assert set(y.unique()) <= {0, 1}


def test_get_claims_with_status_forced_synthetic():
    df, status = get_claims_with_status(prefer_real=False)
    assert status["source"] == "synthetic"
    assert len(df) > 0
    assert TARGET in df.columns


def test_get_claims_with_status_always_reports_a_message():
    # Whichever path is taken (real / fallback / forced synthetic), the
    # status dict must always explain what happened - never silent.
    df, status = get_claims_with_status(prefer_real=True)
    assert status["source"] in {"real", "synthetic_fallback"}
    assert isinstance(status["message"], str) and len(status["message"]) > 0

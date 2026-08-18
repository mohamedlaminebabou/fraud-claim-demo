import numpy as np
import pytest

from fraud_lib.data import generate_claims, to_model_matrix
from fraud_lib.model import (
    build_input_row,
    local_explanation,
    optimal_threshold,
    train_and_select,
)


@pytest.fixture(scope="module")
def bundle():
    # small n for fast tests; behaviour, not exact metrics, is what's checked
    return train_and_select(n=1500, seed=42)


def test_model_beats_random_guessing(bundle):
    assert bundle.auc > 0.6


def test_calibration_improves_or_stays_similar(bundle):
    # isotonic calibration should not make the Brier score meaningfully worse
    assert bundle.brier <= bundle.brier_uncalibrated + 1e-6


def test_candidate_scores_contains_both_models(bundle):
    assert set(bundle.candidate_scores.keys()) == {"logistic_regression", "random_forest"}
    for scores in bundle.candidate_scores.values():
        assert 0.0 <= scores["mean_auc"] <= 1.0


def test_selected_model_is_one_of_the_candidates(bundle):
    assert bundle.model_name in bundle.candidate_scores


def test_permutation_importance_sums_are_finite(bundle):
    assert np.isfinite(bundle.perm_importance.values).all()


def test_build_input_row_matches_feature_columns(bundle):
    inputs = dict(
        claim_type="fire", claim_amount=20000, policy_value=10000,
        policy_tenure_years=0.5, prior_claims_count=2, reported_delay_days=15,
        weekend_incident=1, has_prior_fraud_flag=1,
    )
    row = build_input_row(inputs, bundle.feature_columns)
    assert list(row.columns) == bundle.feature_columns
    assert row.iloc[0]["claim_type_fire"] == 1
    assert row.iloc[0]["claim_to_policy_ratio"] == pytest.approx(2.0)


def test_high_risk_claim_scores_higher_than_low_risk(bundle):
    high_risk = dict(
        claim_type="fire", claim_amount=27000, policy_value=11000,
        policy_tenure_years=0.1, prior_claims_count=3, reported_delay_days=30,
        weekend_incident=1, has_prior_fraud_flag=1,
    )
    low_risk = dict(
        claim_type="glass_breakage", claim_amount=350, policy_value=15000,
        policy_tenure_years=10.0, prior_claims_count=0, reported_delay_days=1,
        weekend_incident=0, has_prior_fraud_flag=0,
    )
    X_high = build_input_row(high_risk, bundle.feature_columns)
    X_low = build_input_row(low_risk, bundle.feature_columns)
    proba_high = bundle.model.predict_proba(X_high)[0, 1]
    proba_low = bundle.model.predict_proba(X_low)[0, 1]
    assert proba_high > proba_low


def test_optimal_threshold_is_within_bounds(bundle):
    thr, sweep = optimal_threshold(bundle.y_test, bundle.proba_test, cost_fp=80, cost_fn=5000)
    assert 0.0 < thr < 1.0
    assert set(["threshold", "fp", "fn", "tp", "tn", "review_rate", "expected_cost"]).issubset(sweep.columns)


def test_capacity_constraint_reduces_or_keeps_review_rate(bundle):
    thr_unconstrained, sweep = optimal_threshold(
        bundle.y_test, bundle.proba_test, cost_fp=80, cost_fn=5000, max_review_rate=1.0
    )
    thr_constrained, _ = optimal_threshold(
        bundle.y_test, bundle.proba_test, cost_fp=80, cost_fn=5000, max_review_rate=0.15
    )
    row_unconstrained = sweep.loc[sweep["threshold"].sub(thr_unconstrained).abs().idxmin()]
    row_constrained = sweep.loc[sweep["threshold"].sub(thr_constrained).abs().idxmin()]
    assert row_constrained["review_rate"] <= row_unconstrained["review_rate"] + 1e-6


def test_local_explanation_returns_requested_top_n(bundle):
    inputs = dict(
        claim_type="theft", claim_amount=5000, policy_value=12000,
        policy_tenure_years=2.0, prior_claims_count=1, reported_delay_days=5,
        weekend_incident=0, has_prior_fraud_flag=0,
    )
    row = build_input_row(inputs, bundle.feature_columns)
    expl = local_explanation(
        row.iloc[0], bundle.train_means, bundle.train_stds, bundle.perm_importance, top_n=4
    )
    assert len(expl) == 4

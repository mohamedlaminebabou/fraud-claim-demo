import numpy as np
import pytest

from fraud_lib.model import (
    build_input_row,
    local_explanation,
    optimal_threshold,
    train_and_select,
)

# prefer_real=False: tests must be deterministic and network-independent,
# so they always exercise the synthetic path regardless of environment.
SAMPLE_INPUT = dict(
    months_as_customer=6, age=28, policy_deductable=500, policy_annual_premium=1400.0,
    umbrella_limit=0, capital_gains=0, capital_loss=0, incident_hour_of_the_day=2,
    number_of_vehicles_involved=1, bodily_injuries=0, witnesses=0,
    total_claim_amount=52000.0, injury_claim=2000.0, property_claim=5000.0, vehicle_claim=45000.0,
    insured_sex="MALE", insured_education_level="College", insured_occupation="sales",
    insured_relationship="not-in-family", policy_state="OH", policy_csl="100/300",
    incident_type="Vehicle Theft", collision_type="Unknown", incident_severity="Total Loss",
    authorities_contacted="None", incident_state="NY", property_damage="NO",
    police_report_available="NO",
)

LOW_RISK_INPUT = dict(SAMPLE_INPUT, total_claim_amount=1500.0, injury_claim=100.0,
                       property_claim=300.0, vehicle_claim=1100.0,
                       incident_severity="Trivial Damage", bodily_injuries=1, witnesses=2,
                       months_as_customer=200, authorities_contacted="Police",
                       police_report_available="YES")


@pytest.fixture(scope="module")
def bundle():
    return train_and_select(prefer_real=False)


def test_data_status_reflects_forced_synthetic(bundle):
    assert bundle.data_status["source"] == "synthetic"


def test_model_beats_random_guessing(bundle):
    assert bundle.auc > 0.65


def test_calibration_improves_or_stays_similar(bundle):
    assert bundle.brier <= bundle.brier_uncalibrated + 1e-6


def test_candidate_scores_contains_both_models(bundle):
    assert set(bundle.candidate_scores.keys()) == {"logistic_regression", "random_forest"}


def test_selected_model_is_one_of_the_candidates(bundle):
    assert bundle.model_name in bundle.candidate_scores


def test_permutation_importance_is_finite(bundle):
    assert np.isfinite(bundle.perm_importance.values).all()


def test_excluded_columns_are_absent_from_features(bundle):
    for col in ["policy_number", "insured_hobbies", "insured_zip", "auto_make", "auto_model"]:
        assert all(not c.startswith(col) for c in bundle.feature_columns)


def test_build_input_row_matches_feature_columns(bundle):
    row = build_input_row(SAMPLE_INPUT, bundle.feature_columns)
    assert list(row.columns) == bundle.feature_columns
    assert row.iloc[0]["age"] == 28
    dummy_col = "incident_severity_Total Loss"
    if dummy_col in row.columns:
        assert row.iloc[0][dummy_col] == 1


def test_high_risk_claim_scores_higher_than_low_risk(bundle):
    X_high = build_input_row(SAMPLE_INPUT, bundle.feature_columns)
    X_low = build_input_row(LOW_RISK_INPUT, bundle.feature_columns)
    proba_high = bundle.model.predict_proba(X_high)[0, 1]
    proba_low = bundle.model.predict_proba(X_low)[0, 1]
    assert proba_high > proba_low


def test_optimal_threshold_is_within_bounds(bundle):
    thr, sweep = optimal_threshold(bundle.y_test, bundle.proba_test, cost_fp=80, cost_fn=5000)
    assert 0.0 < thr < 1.0
    assert {"threshold", "fp", "fn", "tp", "tn", "review_rate", "expected_cost"}.issubset(sweep.columns)


def test_capacity_constraint_reduces_or_keeps_review_rate(bundle):
    thr_u, sweep = optimal_threshold(bundle.y_test, bundle.proba_test, cost_fp=80, cost_fn=5000, max_review_rate=1.0)
    thr_c, _ = optimal_threshold(bundle.y_test, bundle.proba_test, cost_fp=80, cost_fn=5000, max_review_rate=0.15)
    row_u = sweep.loc[sweep["threshold"].sub(thr_u).abs().idxmin()]
    row_c = sweep.loc[sweep["threshold"].sub(thr_c).abs().idxmin()]
    assert row_c["review_rate"] <= row_u["review_rate"] + 1e-6


def test_local_explanation_returns_requested_top_n(bundle):
    row = build_input_row(SAMPLE_INPUT, bundle.feature_columns)
    expl = local_explanation(row.iloc[0], bundle.train_means, bundle.train_stds, bundle.perm_importance, top_n=4)
    assert len(expl) == 4

"""Model training, selection, and explainability for the fraud demo.

Design choices, spelled out because they matter as much as the code:

- Two candidate models (logistic regression baseline, random forest) are
  compared by 5-fold stratified cross-validated ROC AUC on the training
  split, and the best is refit on the full training split. This is a small
  but real model-selection step, not a single model chosen by assumption.
- The operating threshold is NOT a fixed 0.5. It is chosen to minimize an
  expected business cost: cost_fp per false positive (a manual review that
  turns out clean) vs. cost_fn per false negative (a fraud that gets paid
  out). The two costs are exposed to the app so the reviewer can see the
  threshold move as the cost ratio changes.
- Feature importance for the global view uses permutation importance on the
  held-out test set (model-agnostic, unlike a random forest's built-in
  impurity-based importances, which are known to be biased toward
  high-cardinality / high-variance features).
- The per-claim local explanation is a simple importance-weighted deviation
  from the training-set mean. It is explicitly NOT SHAP and the app says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_lib.data import generate_claims, to_model_matrix


@dataclass
class TrainedBundle:
    model: object
    model_name: str
    candidate_scores: dict
    feature_columns: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    proba_test: np.ndarray
    proba_test_uncalibrated: np.ndarray
    auc: float
    brier: float
    brier_uncalibrated: float
    train_means: pd.Series
    train_stds: pd.Series
    perm_importance: pd.Series


def _candidate_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=15,
            random_state=42, class_weight="balanced",
        ),
    }


def train_and_select(n: int = 4000, seed: int = 42, cv_folds: int = 5) -> TrainedBundle:
    df = generate_claims(n=n, seed=seed)
    X, y = to_model_matrix(df)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    candidates = _candidate_models()
    candidate_scores = {}
    for name, est in candidates.items():
        scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="roc_auc")
        candidate_scores[name] = {"mean_auc": float(scores.mean()), "std_auc": float(scores.std())}

    best_name = max(candidate_scores, key=lambda k: candidate_scores[k]["mean_auc"])
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)
    proba_test_uncalibrated = best_model.predict_proba(X_test)[:, 1]
    brier_uncalibrated = brier_score_loss(y_test, proba_test_uncalibrated)

    # class_weight="balanced" (needed here because fraud is a ~8% minority
    # class) shifts the decision function and reliably produces overconfident,
    # poorly-calibrated probabilities. Isotonic calibration on top, fit via
    # internal cross-validation on the training split, fixes this without
    # touching the discrimination (AUC is rank-based, unaffected by
    # monotonic calibration; Brier score, which does care about calibration,
    # is reported before/after so the fix is verifiable, not just claimed).
    calibrated_model = CalibratedClassifierCV(best_model, method="isotonic", cv=5)
    calibrated_model.fit(X_train, y_train)
    proba_test = calibrated_model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, proba_test)

    perm = permutation_importance(
        calibrated_model, X_test, y_test, n_repeats=20, random_state=seed, scoring="roc_auc"
    )
    perm_importance = pd.Series(perm.importances_mean, index=feature_columns).sort_values(ascending=False)

    return TrainedBundle(
        model=calibrated_model,
        model_name=best_name,
        candidate_scores=candidate_scores,
        feature_columns=feature_columns,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        proba_test=proba_test,
        proba_test_uncalibrated=proba_test_uncalibrated,
        auc=roc_auc_score(y_test, proba_test),
        brier=brier,
        brier_uncalibrated=brier_uncalibrated,
        train_means=X_train.mean(),
        train_stds=X_train.std().replace(0, 1),
        perm_importance=perm_importance,
    )


def optimal_threshold(y_true, proba, cost_fp: float, cost_fn: float,
                       max_review_rate: float = 1.0) -> tuple[float, pd.DataFrame]:
    """Sweep thresholds and return the one minimizing expected business cost,
    subject to an operational capacity constraint.

    cost_fp: cost of flagging a clean claim for manual review (analyst time).
    cost_fn: cost of missing an actual fraud (the payout itself).
    max_review_rate: maximum share of claims the review team can realistically
        process (e.g. 0.15 = at most 15% of claims flagged). Pure cost
        minimization with cost_fn >> cost_fp tends to flag almost everything,
        which is not operationally realistic — this constraint makes the
        optimization reflect an actual review team's capacity.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    y_true = np.asarray(y_true)
    n = len(y_true)
    for t in thresholds:
        pred = (proba >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        review_rate = (fp + tp) / n
        expected_cost = fp * cost_fp + fn * cost_fn
        rows.append({
            "threshold": t, "fp": fp, "fn": fn, "tp": tp, "tn": tn,
            "review_rate": review_rate, "expected_cost": expected_cost,
        })
    sweep = pd.DataFrame(rows)
    feasible = sweep[sweep["review_rate"] <= max_review_rate]
    pool = feasible if len(feasible) > 0 else sweep  # fall back if constraint is infeasible
    best_row = pool.loc[pool["expected_cost"].idxmin()]
    return float(best_row["threshold"]), sweep


def calibration_points(y_true, proba, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def build_input_row(inputs: dict, feature_columns: list) -> pd.DataFrame:
    row = {c: 0 for c in feature_columns}
    for k in ["claim_amount", "policy_value", "policy_tenure_years",
              "prior_claims_count", "reported_delay_days",
              "weekend_incident", "has_prior_fraud_flag"]:
        row[k] = inputs[k]
    row["claim_to_policy_ratio"] = inputs["claim_amount"] / inputs["policy_value"]
    dummy_col = f"claim_type_{inputs['claim_type']}"
    if dummy_col in row:
        row[dummy_col] = 1
    return pd.DataFrame([row])[feature_columns]


def local_explanation(x_row: pd.Series, train_means: pd.Series, train_stds: pd.Series,
                       importance: pd.Series, top_n: int = 6) -> pd.Series:
    """Approximate, non-SHAP local explanation: z-scored deviation from the
    training mean, weighted by (permutation) feature importance. Sign shows
    direction, magnitude shows relative influence for THIS claim only."""
    z = (x_row - train_means) / train_stds
    contrib = z * importance.reindex(x_row.index).fillna(0)
    return contrib.sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)

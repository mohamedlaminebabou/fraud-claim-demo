# Insurance Claim Fraud Detection — small live demo

A small, self-contained fraud-scoring demo: score a synthetic insurance
claim, see the calibrated probability, see why, and see the model that
produced it. Built to be opened and used by someone who wasn't in the room
when it was written.

**Live app:** _add your deployed Streamlit Community Cloud URL here_

## Why this scope

The goal was a small demo that still reflects how a fraud-scoring model
should actually be built and evaluated, not just "train a classifier, wire
it to a slider." Concretely, that meant:

- **Compare, don't assume, the model.** Logistic regression and random
  forest are compared by 5-fold cross-validated ROC AUC on the training
  split; the better one is kept. On this dataset the two are close — the
  comparison result (not a foregone conclusion) is shown in the app.
- **Check calibration, don't assume it.** `class_weight="balanced"` is
  needed here because fraud is a ~8% minority class, but it reliably makes
  predicted probabilities overconfident. The app measures this (Brier score
  and calibration curve, before/after) and fixes it with isotonic
  calibration (`CalibratedClassifierCV`, fit only on the training split).
- **Pick the threshold from business costs, not from habit.** Instead of a
  bare 0.5 cutoff, the operating threshold minimizes expected cost
  (false-alarm review cost vs. missed-fraud payout cost) subject to a
  review-capacity constraint — because pure cost minimization with a large
  cost asymmetry tends to flag almost every claim, which is not
  operationally realistic.
- **Explain with the right tool for a demo.** Global importance uses
  permutation importance on the held-out test set (model-agnostic, not
  biased toward high-variance features the way impurity-based importance
  can be). The per-claim explanation is an importance-weighted deviation
  from the training mean — clearly labelled as an approximation, not SHAP.

## What's deliberately left out

See the "Scope & what I left out" tab in the app itself — it's part of the
deliverable, not an afterthought. Short version: no real data (100%
synthetic, see below), no temporal/drift validation, no SHAP, no
monitoring/auth/persistence, no hyperparameter search.

## Data

100% synthetic. `fraud_lib/data.py` generates ~4,000 fictional insurance
claims from a hand-written logistic rule (claim-to-policy-value ratio,
reporting delay, prior fraud flags, policy tenure, etc., plus noise) — nothing
here comes from any employer or real policyholder. The generator is seeded,
so results are reproducible.

## Project layout

```
app.py                 Streamlit UI — imports fraud_lib, no business logic itself
fraud_lib/
  data.py              synthetic data generation
  model.py             model comparison, calibration, cost-sensitive threshold, explainability
tests/
  test_data.py         sanity checks on the data generator
  test_model.py        sanity checks on model selection, calibration, threshold logic
requirements.txt        runtime dependencies (pinned)
requirements-dev.txt     test-only dependencies
.streamlit/config.toml   theming
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app".
3. Pick the repository, branch, and set the main file path to `app.py`.
4. Deploy. First build takes a minute or two (installing dependencies);
   the app itself trains its models in a few seconds on startup and caches
   them (`st.cache_resource`) for the rest of the session.

No secrets, no API keys, no external services — the app is fully
self-contained.

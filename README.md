# Insurance Claim Fraud Detection — small live demo

A small, self-contained fraud-scoring demo on a real public claims dataset: score a
claim, see the calibrated probability, see why, and see the model that produced it —
built to be opened and used by someone who wasn't in the room when it was written.

**Live app:** _add your deployed Streamlit Community Cloud URL here_

## Data

**Primary source:** a real, publicly available auto-insurance claims dataset (~1,000
claims, US, GitHub: [`mwitiderrick/insurancedata`](https://github.com/mwitiderrick/insurancedata)),
widely reused across ML tutorials and academic write-ups as the field's standard public
stand-in — actual insurers do not publish their real fraud data (commercial
confidentiality), so this is the closest broadly-available substitute rather than a
one-off toy set built for this demo.

**Fallback:** if that source can't be reached (no network, moved, rate-limited), the app
automatically falls back to a seeded synthetic dataset shaped like the same schema —
and always says so visibly in the UI banner at the top. Toggle between the two anytime
in the sidebar ("Use real public dataset").

**A deliberate exclusion:** `insured_hobbies` is dropped from modelling. Two of its
categories (`chess`, `cross-fit`) separate fraud vs. not almost perfectly in this
dataset — not a plausible real fraud driver, and a clear sign of synthetic label
leakage. `insured_occupation`, `insured_education_level`, and `insured_relationship`
also use category values identical to the classic UCI Census Income dataset, suggesting
they were backfilled rather than collected; they're kept (excluding every field with any
doubt would leave too little data) but not treated as trustworthy signal on their own.
Full rationale for every excluded column is in `fraud_lib/data.py` (`EXCLUDED_COLUMNS`).

**License note:** public and reused everywhere, but without a formal license attached at
the source — flagged here rather than glossed over.

## Why this scope

- **Compare, don't assume, the model.** Logistic regression and random forest are
  compared by cross-validated ROC AUC on the training split; the better one is kept and
  shown either way.
- **Check calibration, don't assume it.** `class_weight="balanced"` (needed for the
  fraud minority class) reliably makes predicted probabilities overconfident. The app
  measures this (Brier score and calibration curve, before/after) and fixes it with
  isotonic calibration (`CalibratedClassifierCV`, fit only on the training split).
- **Pick the threshold from business costs, not from habit.** The operating threshold
  minimizes expected cost (false-alarm review cost vs. missed-fraud payout cost) subject
  to a review-capacity constraint, instead of a bare 0.5 cutoff.
- **Explain with the right tool for the right reader.** Global importance uses
  permutation importance on the held-out test set (model-agnostic). The per-claim
  explanation is generated in two layers: the raw importance-weighted deviation feeds a
  technical chart (for data scientists / audit, tucked in an expander) AND a
  deterministic, template-filled plain-English sentence (the default view, written for a
  claims handler who has never seen a feature-importance chart). Both come from the exact
  same numbers — nothing is invented, only relabelled and assembled into a sentence.
- **Fail safe and say so.** The real-data fetch is wrapped end-to-end; any failure falls
  back to synthetic data and the UI states exactly what happened, never a silent switch.

## What's in the app

- **Score a claim** — a curated subset of the modelled fields (sidebar), a calibrated
  fraud probability with an animated gauge, a cost- and capacity-aware decision, and a
  plain-English explanation of why (a deterministic, template-filled sentence — the same
  numbers as the technical view, relabelled for a claims handler, not a data scientist;
  the technical chart is still there, tucked in an expander for audit).
- **Model & data** — the dataset as loaded (previewable and downloadable as CSV),
  model comparison, calibration before/after, the cost-based threshold curve, and
  permutation importance — all as interactive Plotly charts.
- **Security & governance** — what security-by-design actually means for this specific
  app, and a short, honest FinOps note.
- **Scope & decisions** — data source rationale and exclusions, a short
  architecture-decision log, and an explicit list of what was left out and why.

## Project layout

```
app.py                 Streamlit UI — imports fraud_lib, no business logic itself
fraud_lib/
  data.py              real-data loader (with visible synthetic fallback), schema, exclusions
  model.py             model comparison, calibration, cost-sensitive threshold, explainability
tests/
  test_data.py         sanity checks on data loading and the synthetic fallback
  test_model.py        sanity checks on model selection, calibration, threshold logic
requirements.txt        runtime dependencies
requirements-dev.txt     test-only dependencies
.streamlit/config.toml   theming
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

First load trains and calibrates the model (~15-25s, cached afterward via
`st.cache_resource`) and, on a machine with internet access, fetches the real dataset —
watch for the green "real dataset loaded" banner at the top to confirm.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests always force the synthetic path (`prefer_real=False`) where determinism matters,
so they pass identically with or without network access.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click
   "New app".
3. Pick the repository, branch, and set the main file path to `app.py`.
4. Deploy. First build takes 1-3 minutes (installing dependencies); the app then trains
   and caches its model for the rest of the session.

No secrets, no API keys — the app is self-contained beyond the one public CSV fetch.

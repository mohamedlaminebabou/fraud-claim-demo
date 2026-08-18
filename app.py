"""
Insurance Claim Fraud Detection — small live demo
----------------------------------------------------
Primary data: a real, public auto-insurance claims dataset (GitHub). Falls
back automatically (and visibly) to a seeded synthetic dataset if the real
source can't be reached. No employer data of any kind, in either case.

Repo layout:
  app.py             <- this file, Streamlit UI only, no modelling logic
  fraud_lib/data.py  <- real-data loader + synthetic fallback
  fraud_lib/model.py <- model comparison, calibration, cost-sensitive threshold, explainability
  tests/             <- pytest sanity checks on the above
"""
import html as html_lib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud_lib.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from fraud_lib.model import (
    build_input_row,
    calibration_points,
    local_explanation,
    optimal_threshold,
    train_and_select,
)

st.set_page_config(page_title="Claim Fraud Detection — demo", page_icon="🔎", layout="wide")

# ============================================================================
# Design tokens — same palette used across the accompanying report.
# ============================================================================
NAVY = "#12233f"
BLUE = "#2c4a7c"
GREEN = "#1e8a5f"
RED = "#c0392b"
AMBER = "#c68414"
INK = "#1c2530"
MUTED = "#5a6472"
BORDER = "#e3e8f0"
BG = "#f7f9fc"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.6rem; max-width: 1220px; }}

    h1, h2, h3 {{ color: {NAVY} !important; font-weight: 700 !important; }}
    p, li, label {{ color: {INK}; }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .card {{
        background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 1.2rem 1.3rem; margin-bottom: 1rem;
        animation: fadeInUp .45s ease-out both;
        transition: box-shadow .18s ease, transform .18s ease;
    }}
    .card:hover {{ box-shadow: 0 6px 20px rgba(18,35,63,0.08); transform: translateY(-1px); }}
    .card h4 {{ color: {NAVY}; margin-top: 0; font-size: 1rem; }}
    .subtle {{ color: {MUTED}; font-size: .87rem; }}

    .hero {{
        background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
        border-radius: 14px; padding: 1.8rem 2rem; margin-bottom: 1.2rem;
        animation: fadeInUp .5s ease-out both;
    }}
    .hero h1 {{ color: #ffffff !important; font-size: 1.65rem !important; margin: 0 0 .35rem 0; }}
    .hero p {{ color: #dbe4f5 !important; margin: 0; font-size: .95rem; }}
    .chip {{
        display: inline-block; padding: .18rem .65rem; border-radius: 999px;
        font-size: .72rem; font-weight: 600; letter-spacing: .03em; margin-right: .4rem;
        background: rgba(255,255,255,0.14); color: #eef3fb; border: 1px solid rgba(255,255,255,0.28);
    }}

    .status-banner {{
        border-radius: 10px; padding: .55rem .9rem; margin-bottom: 1rem;
        font-size: .85rem; animation: fadeInUp .4s ease-out both;
    }}
    .status-real {{ background: #e8f5ef; border: 1px solid #bfe3d0; color: {GREEN}; }}
    .status-fallback {{ background: #fdf3e3; border: 1px solid #f0dba9; color: {AMBER}; }}

    .verdict-badge {{
        display: inline-block; padding: .45rem .9rem; border-radius: 8px;
        font-weight: 700; font-size: .95rem; margin-top: .4rem;
        transition: background-color .25s ease, color .25s ease;
    }}
    .verdict-review {{ background: #fbeae8; color: {RED}; border: 1px solid #f2c6c0; }}
    .verdict-clear {{ background: #e8f5ef; color: {GREEN}; border: 1px solid #bfe3d0; }}

    section[data-testid="stSidebar"] {{ background-color: #ffffff; border-right: 1px solid {BORDER}; }}
    section[data-testid="stSidebar"] h3 {{ font-size: .92rem !important; }}
    div[data-testid="stMetric"] {{
        background: #ffffff; border: 1px solid {BORDER}; border-radius: 10px; padding: .7rem .9rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #ffffff; border-radius: 8px 8px 0 0; padding: .5rem 1rem;
        border: 1px solid {BORDER}; border-bottom: none;
    }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(color=INK, size=12, family="sans-serif"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def chip(text):
    return f'<span class="chip">{text}</span>'


def animated_number(value_pct: float, key: str, size="2.6rem", color=NAVY, duration_ms=700):
    """Small self-contained JS component: animates counting up to the given
    percentage. This is the one deliberate motion element tied to the single
    most important number in the app, not decoration for its own sake."""
    import streamlit.components.v1 as components
    components.html(f"""
    <div id="num-{key}" style="font-size:{size}; font-weight:700; color:{color};
         font-family: sans-serif;">0.0%</div>
    <script>
      const el = document.getElementById("num-{key}");
      const target = {value_pct};
      const duration = {duration_ms};
      const start = performance.now();
      function tick(now) {{
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        el.textContent = (target * eased).toFixed(1) + "%";
        if (t < 1) requestAnimationFrame(tick);
      }}
      requestAnimationFrame(tick);
    </script>
    """, height=60)


def plotly_gauge(value: float, threshold: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 1}},  # number hidden; animated_number() shows it
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"size": 9}},
            "bar": {"color": NAVY, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "#e8f5ef"},
                {"range": [threshold * 100, 100], "color": "#fbeae8"},
            ],
            "threshold": {
                "line": {"color": RED, "width": 3},
                "thickness": 0.85,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=180, **{**PLOTLY_LAYOUT, "margin": dict(l=20, r=20, t=15, b=5)})
    return fig


@st.cache_resource(show_spinner="Training and calibrating the model (first load only, ~15-25s)…")
def get_bundle(prefer_real: bool):
    return train_and_select(prefer_real=prefer_real)


# ============================================================================
# Header + data source toggle
# ============================================================================
st.markdown(f"""
<div class="hero">
  <div style="margin-bottom:.5rem;">
    {chip("PUBLIC DATA ONLY")}{chip("NO EMPLOYER DATA")}{chip("STATELESS · NO PERSISTENCE")}
  </div>
  <h1>🔎 Insurance Claim Fraud Detection</h1>
  <p>A small, self-contained fraud-scoring demo on a real public claims dataset —
  calibrated probabilities, a cost-aware decision threshold, and the model behind them,
  all inspectable in this app.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Data source")
    prefer_real = st.toggle("Use real public dataset", value=True,
                             help="Off = always use the seeded synthetic fallback instead.")

bundle = get_bundle(prefer_real)
status = bundle.data_status
if status["source"] == "real":
    st.markdown(
        f'<div class="status-banner status-real">✅ {status["message"]} '
        f'Source: <code>{status["detail"]}</code></div>',
        unsafe_allow_html=True,
    )
elif status["source"] == "synthetic_fallback":
    st.markdown(
        f'<div class="status-banner status-fallback">⚠️ {html_lib.escape(status["message"])} '
        "The app still works — nothing else changes.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="status-banner status-fallback">ℹ️ {status["message"]} '
        "(synthetic fallback selected in the sidebar)</div>",
        unsafe_allow_html=True,
    )

model = bundle.model
feature_columns = bundle.feature_columns
df = bundle.df

# ============================================================================
# Sidebar — curated inputs (see Scope & decisions for the fields fixed at
# their dataset median/mode to keep this form usable)
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### Policyholder")
    age = st.slider("Age", 18, 70, 32)
    months_as_customer = st.slider("Months as customer", 0, 480, 120)
    insured_sex = st.selectbox("Sex", ["MALE", "FEMALE"])
    insured_education_level = st.selectbox(
        "Education level",
        ["High School", "Associate", "College", "JD", "MD", "Masters", "PhD"],
    )

    st.markdown("### Policy")
    policy_annual_premium = st.slider("Annual premium (€)", 400, 2500, 1250)
    policy_deductable = st.select_slider("Deductible (€)", [500, 1000, 2000], value=1000)
    umbrella_limit = st.select_slider(
        "Umbrella limit (€)", [0, 1_000_000, 2_000_000, 5_000_000], value=0,
        format_func=lambda v: f"{v:,}",
    )
    policy_csl = st.selectbox("Combined single limit", ["100/300", "250/500", "500/1000"])

    st.markdown("### Incident")
    incident_severity = st.selectbox(
        "Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"], index=2,
    )
    incident_type = st.selectbox(
        "Incident type",
        ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"],
    )
    collision_type = st.selectbox("Collision type", ["Rear Collision", "Side Collision", "Front Collision", "Unknown"])
    number_of_vehicles_involved = st.slider("Vehicles involved", 1, 4, 1)
    bodily_injuries = st.slider("Bodily injuries", 0, 2, 0)
    witnesses = st.slider("Witnesses", 0, 3, 0)
    incident_hour_of_the_day = st.slider("Hour of incident", 0, 23, 2)
    authorities_contacted = st.selectbox("Authorities contacted", ["Police", "Fire", "Ambulance", "Other", "None"])
    property_damage = st.selectbox("Property damage noted", ["YES", "NO"])
    police_report_available = st.selectbox("Police report available", ["YES", "NO"])

    st.markdown("### Claim amounts")
    total_claim_amount = st.slider("Total claim (€)", 100, 100000, 15000, step=100)
    injury_claim = st.slider("Injury claim (€)", 0, 30000, 2000, step=100)
    property_claim = st.slider("Property claim (€)", 0, 30000, 3000, step=100)
    vehicle_claim = st.slider("Vehicle claim (€)", 0, 80000, 10000, step=100)

    st.markdown("---")
    st.markdown("### Business cost assumptions")
    cost_fp = st.number_input("Cost of a false alarm (€)", 10, 1000, 80, step=10)
    cost_fn = st.number_input("Cost of a missed fraud (€)", 500, 20000, 8000, step=100)
    max_review_rate = st.slider("Max share of claims reviewable", 0.02, 1.0, 0.15, step=0.01)

# fields not exposed in the form: fixed at the loaded dataset's median/mode
_fixed = {}
for col in NUMERIC_FEATURES:
    if col not in ["age", "months_as_customer", "policy_annual_premium", "policy_deductable",
                    "umbrella_limit", "incident_hour_of_the_day", "number_of_vehicles_involved",
                    "bodily_injuries", "witnesses", "total_claim_amount", "injury_claim",
                    "property_claim", "vehicle_claim"] and col in df.columns:
        _fixed[col] = float(df[col].median())
for col in CATEGORICAL_FEATURES:
    if col not in ["insured_sex", "insured_education_level", "policy_csl", "incident_type",
                    "collision_type", "incident_severity", "authorities_contacted",
                    "property_damage", "police_report_available"] and col in df.columns:
        _fixed[col] = df[col].mode().iloc[0]

inputs = dict(
    age=age, months_as_customer=months_as_customer, insured_sex=insured_sex,
    insured_education_level=insured_education_level, policy_annual_premium=policy_annual_premium,
    policy_deductable=policy_deductable, umbrella_limit=umbrella_limit, policy_csl=policy_csl,
    incident_severity=incident_severity, incident_type=incident_type, collision_type=collision_type,
    number_of_vehicles_involved=number_of_vehicles_involved, bodily_injuries=bodily_injuries,
    witnesses=witnesses, incident_hour_of_the_day=incident_hour_of_the_day,
    authorities_contacted=authorities_contacted, property_damage=property_damage,
    police_report_available=police_report_available, total_claim_amount=total_claim_amount,
    injury_claim=injury_claim, property_claim=property_claim, vehicle_claim=vehicle_claim,
    **_fixed,
)
X_new = build_input_row(inputs, feature_columns)
proba = model.predict_proba(X_new)[0, 1]
threshold, sweep = optimal_threshold(
    bundle.y_test, bundle.proba_test, cost_fp=cost_fp, cost_fn=cost_fn,
    max_review_rate=max_review_rate,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Score a claim", "Model & data", "Security & governance", "Scope & decisions",
])

# ---- TAB 1 : score a claim -------------------------------------------------
with tab1:
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Fraud probability")
        animated_number(proba * 100, key="proba")
        st.plotly_chart(plotly_gauge(proba, threshold), use_container_width=True, config={"displayModeBar": False})
        if proba >= threshold:
            st.markdown('<span class="verdict-badge verdict-review">→ Route to manual review</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="verdict-badge verdict-clear">→ Auto-approve</span>', unsafe_allow_html=True)
        st.markdown(
            f"<p class='subtle' style='margin-top:.7rem;'>Operating threshold "
            f"<b>{threshold:.0%}</b>, chosen by minimizing expected cost on the held-out "
            f"test set for the cost assumptions in the sidebar, subject to reviewing at "
            f"most {max_review_rate:.0%} of claims.</p>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Why this score")
        st.markdown(
            "<p class='subtle'>Deviation from the training-set average, weighted by "
            "permutation importance — an honest approximation, not a SHAP attribution "
            "(see Scope &amp; decisions).</p>",
            unsafe_allow_html=True,
        )
        expl = local_explanation(X_new.iloc[0], bundle.train_means, bundle.train_stds, bundle.perm_importance)
        fig = go.Figure(go.Bar(
            x=expl.values[::-1], y=expl.index[::-1], orientation="h",
            marker_color=[RED if v > 0 else BLUE for v in expl.values[::-1]],
        ))
        fig.update_layout(height=300, xaxis_title="push toward fraud (red) / away (blue)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# ---- TAB 2 : model & data ---------------------------------------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Underlying dataset")
    src_note = (
        "Real public dataset (see Scope &amp; decisions for source, license note, and "
        "an excluded leakage-prone column)."
        if status["source"] == "real" else
        "Synthetic fallback, seeded and shaped like the real dataset's schema."
    )
    st.markdown(f"<p class='subtle'>{src_note}</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Claims loaded", f"{len(df):,}")
    c2.metric("Fraud rate", f"{df['is_fraud'].mean():.1%}")
    c3.metric("Test-set ROC AUC", f"{bundle.auc:.3f}")
    st.dataframe(df.head(50), use_container_width=True, height=220)
    st.download_button(
        "⬇ Download this dataset as loaded (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="claims_as_loaded.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Fraud rate by incident severity")
        rates = df.groupby("incident_severity")["is_fraud"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=rates.index, y=rates.values, marker_color=BLUE))
        fig.update_layout(height=300, yaxis_title="fraud rate", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Total claim amount — fraud vs. not")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df.loc[df.is_fraud == 0, "total_claim_amount"], name="not fraud",
                                    marker_color=BLUE, opacity=0.75, nbinsx=40))
        fig.add_trace(go.Histogram(x=df.loc[df.is_fraud == 1, "total_claim_amount"], name="fraud",
                                    marker_color=RED, opacity=0.75, nbinsx=40))
        fig.update_layout(height=300, barmode="overlay", xaxis_title="total claim (€)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Model comparison (cross-validated ROC AUC)")
    cmp_df = pd.DataFrame(bundle.candidate_scores).T
    cmp_df.index.name = "model"
    st.dataframe(cmp_df.style.format({"mean_auc": "{:.3f}", "std_auc": "{:.3f}"}), use_container_width=True)
    st.markdown(
        f"<p class='subtle'>Selected: <b>{bundle.model_name.replace('_', ' ')}</b> "
        "(highest mean CV AUC) — chosen by measurement, not assumption.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Calibration — before vs. after")
        mp_before, fp_before = calibration_points(bundle.y_test, bundle.proba_test_uncalibrated)
        mp_after, fp_after = calibration_points(bundle.y_test, bundle.proba_test)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=BORDER), name="perfect"))
        fig.add_trace(go.Scatter(x=mp_before, y=fp_before, mode="lines+markers", line=dict(color=RED),
                                  name=f"before ({bundle.brier_uncalibrated:.3f})"))
        fig.add_trace(go.Scatter(x=mp_after, y=fp_after, mode="lines+markers", line=dict(color=GREEN),
                                  name=f"after ({bundle.brier:.3f})"))
        fig.update_layout(height=330, xaxis_title="mean predicted probability",
                           yaxis_title="observed fraud rate", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Why calibration was added")
        st.markdown(
            "<p class='subtle'><code>class_weight=\"balanced\"</code> is needed for the "
            "fraud minority class, but it reliably makes probabilities overconfident — the "
            "red curve sits below the diagonal at high scores. Isotonic calibration "
            "(fit on the training split only, evaluated on the untouched test split) "
            "brings it back toward the diagonal and lowers the Brier score. This matters "
            "because the cost-based threshold on the left is only trustworthy if the "
            "probability feeding it is honest.</p>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Cost-based threshold sweep")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["expected_cost"], mode="lines", line=dict(color=BLUE, width=2.5)))
    fig.add_vline(x=threshold, line=dict(color=RED, dash="dash"), annotation_text=f"selected = {threshold:.2f}")
    fig.update_layout(height=280, xaxis_title="threshold", yaxis_title="expected cost on test set (€)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<p class='subtle'>Expected cost = (false positives × false-alarm cost) + "
        "(false negatives × missed-fraud cost), on the held-out test set, subject to the "
        "review-capacity constraint. Change the cost inputs in the sidebar and this curve "
        "moves with them.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Global feature importance (permutation, test set, top 15)")
    imp = bundle.perm_importance.sort_values().tail(15)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=GREEN))
    fig.update_layout(height=420, xaxis_title="mean drop in ROC AUC when the feature is shuffled", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"<p class='subtle'>{len(bundle.feature_columns)} features after one-hot encoding "
        "(15 numeric + 13 categorical fields). Permutation importance rather than a "
        "model's built-in impurity importance: model-agnostic, measured on the held-out "
        "test set.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---- TAB 3 : security & governance -----------------------------------------
with tab3:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Security by design")
        st.markdown(
            """
- **No real or personal data.** The primary source is a long-standing public dataset with
  no link to any named insurer or real policyholder; the fallback is generated by a seeded
  rule. Either way, there is nothing here to leak.
- **No secrets, no credentials.** No API keys, tokens, or connection strings anywhere;
  `.gitignore` excludes `.streamlit/secrets.toml` by default.
- **Outbound network calls are scoped and fail safely.** The only external call is a
  read-only fetch of one public CSV, wrapped in a try/except that falls back to synthetic
  data on any failure — timeout, HTTP error, schema change — and says so in the UI rather
  than crashing or silently substituting data.
- **Bounded inputs, not free text.** Every field in the scoring form is a slider or a
  select box with a fixed set of options — no free-text field reaches a query, a file
  path, or a template, so there is no injection surface to defend.
- **Stateless.** Nothing is written to disk or a database between sessions.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### FinOps — honest note")
        st.markdown(
            """
Runs on Streamlit Community Cloud's free tier: **zero infrastructure cost**. Two choices
keep it that way:

- Model training (cross-validation, calibration, permutation importance) is wrapped in
  `st.cache_resource`, so it runs **once per app restart**, not once per click. Cut from an
  initial ~230s to ~15-20s by trimming forest size, permutation repeats, and calibration
  folds — a deliberate speed/precision trade-off, not an oversight.
- Data loading is cached separately (`st.cache_data`), and the real-data fetch is a single
  small CSV, not a recurring or paginated call.

At production scale, the real cost drivers would be retraining cadence, monitoring, and
data storage — not shown here because a stateless demo has none of those, not because
they don't matter.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ---- TAB 4 : scope & decisions ----------------------------------------------
with tab4:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Data source & a deliberate exclusion")
        st.markdown(
            """
**Why not "real data from a real named insurer"?** Actual insurers do not publish their
real fraud data — commercial confidentiality. The dataset used here (~1,000 US auto
claims, GitHub: `mwitiderrick/insurancedata`) is the field's de facto public stand-in,
reused across dozens of tutorials and academic write-ups. It is real in the sense of
being an independently published, fixed, external dataset with genuine messiness
(missing values marked `?`, mixed types) — not something generated for this demo.

**`insured_hobbies` is excluded on purpose.** Two of its categories (`chess`,
`cross-fit`) separate fraud vs. not almost perfectly in this dataset — not a plausible
real fraud driver, and a strong sign of synthetic label leakage. Similarly,
`insured_occupation`, `insured_education_level`, and `insured_relationship` use category
values identical to the classic UCI Census Income dataset, suggesting those fields were
backfilled rather than collected. They're kept (dropping every field with any doubt would
leave little data left) but not treated as trustworthy signal on their own.

**License note.** Public and widely reused, but without a formal license attached at the
source — noted here rather than glossed over.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Architecture decisions")
        st.markdown(
            """
**Model: compare, don't assume.** Logistic regression and random forest are compared by
cross-validated ROC AUC; the better one is kept, shown either way.

**Calibration: measure, then fix.** `class_weight="balanced"` reliably overconfidence
probabilities; isotonic calibration is applied and the before/after Brier score is shown.

**Threshold: cost- and capacity-aware, not 0.5.** The sweep makes the false-alarm-vs-
missed-fraud trade-off, and a review team's real capacity, explicit and adjustable.

**Form: curated, not exhaustive.** ~22 of the ~28 modelled fields are exposed as inputs;
the remainder (capital gains/loss, occupation, relationship, policy/incident state) are
fixed at the loaded dataset's median/mode to keep the form usable — full set trained on
and shown in Model & data.

**Explainability: proportionate to scope.** Permutation importance (global) and an
importance-weighted deviation from the mean (local) — not SHAP; see below.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Left out on purpose")
    st.markdown(
        """
- **No temporal validation.** The split is random, not time-based; a real fraud model
  needs validation on *future* claims relative to training, to catch drift.
- **No SHAP / game-theoretic explainability.** Good enough to reason about one claim in a
  demo, not for an explanation shown to a client or auditor.
- **No monitoring, auth, logging, or persistence** — a scoring demo, not a deployable
  production service.
- **No resampling strategy beyond `class_weight="balanced"`.** No SMOTE, no systematic
  comparison against resampling.
- **No hyperparameter search.** Reasonable, speed-conscious defaults, not tuned by
  grid/random/Bayesian search.

Happy to talk through how each of these would actually be tackled in production.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    "<p class='subtle' style='text-align:center; margin-top:1rem;'>Source: this app's "
    "GitHub repository — <code>fraud_lib/</code> for logic, <code>tests/</code> for sanity checks.</p>",
    unsafe_allow_html=True,
)

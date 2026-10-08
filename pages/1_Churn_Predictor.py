import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from churn_features import ADDON_COLS, RAW_COLS  # noqa: E402  (needed to unpickle the pipeline)

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "churn_metrics.json"


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    bundle = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return bundle["model"], float(bundle["threshold"]), metrics


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_churn.py` first, then commit `models/`.")
    st.stop()

model, threshold, metrics = load_artifacts()

# ---------------------------------------------------------------- state / presets
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENTS = ["Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"]

DEFAULTS = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic",
    **{c: "No" for c in ADDON_COLS},
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0,
}
PRESETS = {
    "High-risk newcomer": {**DEFAULTS, "tenure": 2, "MonthlyCharges": 85.0},
    "Loyal long-term customer": {
        **DEFAULTS, "tenure": 60, "Contract": "Two year", "InternetService": "DSL",
        "PaymentMethod": "Bank transfer (automatic)", "PaperlessBilling": "No",
        "Partner": "Yes", "Dependents": "Yes", "MonthlyCharges": 55.0,
        **{c: "Yes" for c in ADDON_COLS},
    },
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    preset = PRESETS.get(st.session_state["preset"])
    if preset:
        st.session_state.update(preset)


# ---------------------------------------------------------------- header
st.title("📉 Customer Churn Predictor")
st.caption("Random Forest + SMOTE (applied only inside training folds) in a single Scikit-Learn pipeline.")
tab_predict, tab_perf, tab_how = st.tabs(["🔮 Predict", "📊 Model performance", "⚙️ How it works"])

# ---------------------------------------------------------------- predict tab
with tab_predict:
    st.selectbox("Quick-load an example profile", ["Custom", *PRESETS], key="preset",
                 on_change=apply_preset)

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Account")
            st.slider("Tenure (months)", 0, 72, key="tenure")
            st.selectbox("Contract", CONTRACTS, key="Contract")
            st.selectbox("Payment method", PAYMENTS, key="PaymentMethod")
            st.selectbox("Paperless billing", ["Yes", "No"], key="PaperlessBilling")
            st.slider("Monthly charges", 18.0, 120.0, step=0.5, key="MonthlyCharges")
        with c2:
            st.subheader("Services")
            st.selectbox("Internet service", INTERNET, key="InternetService")
            st.selectbox("Phone service", ["Yes", "No"], key="PhoneService")
            st.selectbox("Multiple lines", ["No", "Yes"], key="MultipleLines")
            for col in ADDON_COLS:
                st.selectbox(col, ["No", "Yes"], key=col)
        with c3:
            st.subheader("Demographics")
            st.selectbox("Gender", ["Female", "Male"], key="gender")
            st.selectbox("Senior citizen", ["No", "Yes"], key="SeniorCitizen")
            st.selectbox("Has partner", ["No", "Yes"], key="Partner")
            st.selectbox("Has dependents", ["No", "Yes"], key="Dependents")
            st.info("Total charges are estimated as tenure × monthly charges. "
                    "Add-ons are ignored automatically if there is no internet/phone service.")
        submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

    if submitted:
        s = st.session_state
        row = {c: s[c] for c in RAW_COLS if c not in ("SeniorCitizen", "TotalCharges")}
        row["SeniorCitizen"] = 1 if s["SeniorCitizen"] == "Yes" else 0
        if s["PhoneService"] == "No":
            row["MultipleLines"] = "No phone service"
        if s["InternetService"] == "No":
            row.update({c: "No internet service" for c in ADDON_COLS})
        row["TotalCharges"] = round(s["tenure"] * s["MonthlyCharges"], 2)
        X = pd.DataFrame([row])[RAW_COLS]

        proba = float(model.predict_proba(X)[0, 1])
        flagged = proba >= threshold

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn probability", f"{proba:.1%}")
        r2.metric("Decision threshold", f"{threshold:.2f}",
                  help="Tuned to favour recall: missing a churner costs more than a wasted offer.")
        r3.metric("Verdict", "⚠️ Likely to churn" if flagged else "✅ Likely to stay")
        st.progress(min(proba, 1.0))

        if flagged:
            st.error("**Recommended action:** add to the retention campaign. Offer a contract "
                     "upgrade incentive or a bundle of add-on services.")
        elif proba >= threshold * 0.6:
            st.warning("**Watch-list:** below the alert threshold but elevated. Monitor engagement.")
        else:
            st.success("**Low risk:** no intervention needed. Candidate for loyalty or upsell offers.")

        with st.expander("Exact record sent to the model"):
            st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)

# ---------------------------------------------------------------- performance tab
with tab_perf:
    if not metrics:
        st.info("Run `train_churn.py` to generate held-out test metrics.")
    else:
        st.caption(f"Held-out test set: {metrics['n_test']} customers · "
                   f"churn rate in data: {metrics['churn_rate']:.1%} · accuracy deliberately not used.")
        m = st.columns(4)
        m[0].metric("Precision", f"{metrics['precision']:.3f}")
        m[1].metric("Recall", f"{metrics['recall']:.3f}")
        m[2].metric("F1-score", f"{metrics['f1']:.3f}")
        m[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            cm = pd.DataFrame(metrics["confusion_matrix"],
                              index=["Actual: stay", "Actual: churn"],
                              columns=["Pred: stay", "Pred: churn"])
            st.dataframe(cm, use_container_width=True)
            st.caption("Bottom-left = missed churners (false negatives), the costly error.")
        with right:
            st.subheader("Top drivers of churn")
            fi = pd.DataFrame(metrics["top_features"], columns=["feature", "importance"]).set_index("feature")
            st.bar_chart(fi)

# ---------------------------------------------------------------- how it works tab
with tab_how:
    st.markdown(
        """
**Pipeline (one object, identical at train and inference time)**
1. **Feature engineering**: `TotalCharges` coerced to numeric; derived *tenure bucket*, *average spend per month*, *add-on count*.
2. **ColumnTransformer**: median-impute + standardise numerics; most-frequent-impute + one-hot categoricals (`handle_unknown='ignore'`).
3. **SMOTE** (k=5): synthetic minority samples, run *only on training folds* → no leakage.
4. **Random Forest**: tuned with `RandomizedSearchCV` (stratified 5-fold, optimising churn-class F1).
5. **Decision threshold**: chosen on out-of-fold predictions (F2-score) instead of the default 0.5.
        """
    )
    if metrics.get("best_params"):
        st.caption("Selected hyper-parameters")
        st.json(metrics["best_params"])

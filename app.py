from xgboost import XGBClassifier
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Risk Advisor", layout="wide")

RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

PAYMENTS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
]


@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model("churn_model.json") 
    with open("model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def prepare_input(raw, columns):
    # Keep the serving encoder consistent with the training columns.
    encoded = pd.get_dummies(raw[RAW_COLS])
    return encoded.reindex(columns=columns, fill_value=0).astype(float)


model, meta = load_artifacts()
COLS = meta["feature_columns"]


def score(raw):
    return model.predict_proba(prepare_input(raw, COLS))[:, 1]


st.title("Customer Churn Risk Advisor")
st.caption(
    f"{meta['model_name']} | CV AUC "
    f"{meta['cv_auc']:.4f} ± {meta['cv_auc_std']:.4f} "
    "| Decision support only"
)

with st.sidebar:
    st.header("Customer profile")

    tenure = st.slider("Tenure (months)", 0, 72, 4)
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    monthly = st.number_input(
        "Monthly charges (USD)", 18.0, 120.0, 85.0, step=0.5
    )
    internet = st.selectbox(
        "Internet service", ["Fiber optic", "DSL", "No"]
    )

    services = []
    if internet != "No":
        services = st.multiselect(
            "Add-on services", SERVICES, default=["StreamingTV"]
        )

    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
    multi = (
        st.radio("Multiple lines", ["No", "Yes"], horizontal=True)
        if phone == "Yes" else "No phone service"
    )
    payment = st.selectbox("Payment method", PAYMENTS)
    paperless = st.radio(
        "Paperless billing", ["Yes", "No"], horizontal=True
    )

    with st.expander("Demographics"):
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has partner")
        dependents = st.checkbox("Has dependents")

    threshold = st.slider(
        "Contact threshold", 0.05, 0.95,
        float(meta["threshold"]), 0.05
    )

row = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": "Yes" if partner else "No",
    "Dependents": "Yes" if dependents else "No",
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multi,
    "InternetService": internet,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly
}

for service in SERVICES:
    row[service] = (
        "No internet service" if internet == "No"
        else "Yes" if service in services else "No"
    )

customer = pd.DataFrame([row])

tab1, tab2, tab3 = st.tabs(
    ["One customer", "Batch scoring", "About"]
)

with tab1:
    p = float(score(customer)[0])
    band = "HIGH" if p >= threshold else (
        "WATCH" if p >= threshold / 2 else "LOW"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{p:.1%}")
    c2.metric("Risk band", band)
    c3.metric(
        "Suggested action",
        "Contact customer" if band == "HIGH" else "Monitor"
    )

    st.progress(float(np.clip(p, 0, 1)))
    st.caption(
        "This is a model estimate, not a guarantee that the customer will leave."
    )

    st.subheader("What would change the risk?")

    changes = [
        ("Contract", value)
        for value in ["One year", "Two year"]
        if value != contract
    ]

    if payment == "Electronic check":
        changes.append(
            ("PaymentMethod", "Credit card (automatic)")
        )

    if internet != "No" and "TechSupport" not in services:
        changes.append(("TechSupport", "Yes"))

    results = []
    for feature, value in changes:
        alternative = customer.copy()
        alternative.loc[:, feature] = value
        new_p = float(score(alternative)[0])

        results.append({
            "Change": f"{feature}: {value}",
            "New probability": round(new_p, 3),
            "Difference": round(new_p - p, 3)
        })

    if results:
        st.dataframe(pd.DataFrame(results), hide_index=True)
    else:
        st.info("No alternative changes are available.")

    st.caption(
        "What-if results show learned associations, not guaranteed causal effects."
    )

with tab2:
    st.subheader("Score multiple customers")

    uploaded = st.file_uploader("Upload a customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [c for c in RAW_COLS if c not in batch.columns]

            if missing:
                st.error(
                    "Missing required columns: " + ", ".join(missing)
                )
            else:
                probabilities = score(batch)
                output = batch.copy()
                output["ChurnProbability"] = probabilities.round(4)
                output["RiskBand"] = [
                    "HIGH" if x >= threshold else
                    "WATCH" if x >= threshold / 2 else "LOW"
                    for x in probabilities
                ]

                st.write(f"Scored {len(output)} customers.")
                st.dataframe(output.head(100), hide_index=True)

                st.download_button(
                    "Download scored customers",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="scored_customers.csv",
                    mime="text/csv"
                )

        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

with tab3:
    st.subheader("About this model")
    st.write("Model:", meta["model_name"])
    st.write("Features:", len(COLS))
    st.write("scikit-learn version:", meta["sklearn_version"])
    st.write("Cross-validation AUC:", meta["cv_auc"])
    st.write("CV AUC standard deviation:", meta["cv_auc_std"])
    st.write("Test AUC:", meta["test_auc"])
    st.write("Training data:", meta["training_data"])
    st.write("Model version:", meta["version"])

    st.warning(
        "This app supports retention decisions only. "
        "It does not prove that any particular action will prevent churn."
    )

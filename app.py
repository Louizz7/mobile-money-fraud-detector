# STREAMLIT PRODUCTION APPLICATION

import streamlit as st
import pandas as pd
import json
from pathlib import Path

from predictor import predict_transaction

#LOAD MODEL METADATA
PROJECT_DIR = Path(__file__).resolve().parent
METADATA_PATH = (PROJECT_DIR/ "models"/ "model_metadata.json")

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    MODEL_METADATA = json.load(file)
    
#PAGE CONFIGURATION
st.set_page_config(page_title="Mobile-Money Fraud Detection", page_icon="🔐", layout="wide")

#APPLICATION HEADER
st.title("🔐 Mobile-Money Fraud Detection")
st.markdown(
    """
    ### AI-Powered Transaction Risk Analysis
    Analyze mobile-money transactions and identify potential
    fraudulent activity using a calibrated machine-learning model.
    """
)
st.divider()

# TRANSACTION INPUT
st.subheader("Transaction Details")
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox("Transaction Type", options=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100000.0, step=1000.0)
    oldbalance_org = st.number_input("Origin Balance Before Transaction", min_value=0.0, value=100000.0, step=1000.0)

with col2:
    newbalance_orig = st.number_input("Origin Balance After Transaction", min_value=0.0, value=0.0, step=1000.0)
    oldbalance_dest = st.number_input("Destination Balance Before Transaction", min_value=0.0, value=0.0, step=1000.0)
    newbalance_dest = st.number_input("Destination Balance After Transaction", min_value=0.0, value=100000.0, step=1000.0)
    
st.divider()

# ANALYZE TRANSACTION
if st.button("Analyze Transaction", type="primary", use_container_width=True
):
    transaction = {"step": 1, "type": transaction_type, "amount": amount, "oldbalanceOrg": oldbalance_org,
                   "newbalanceOrig": newbalance_orig, "oldbalanceDest": oldbalance_dest, "newbalanceDest": newbalance_dest}

    try:
        prediction, engineered = predict_transaction(transaction)

        result = prediction.iloc[0]

        st.session_state["prediction"] = result
        st.session_state["engineered"] = engineered.iloc[0]

    except (AttributeError, KeyError, TypeError, ValueError) as error:
        st.error(
            f"Unable to analyze transaction: {error}"
        )

# DISPLAY RESULTS
if "prediction" in st.session_state:

    result = st.session_state["prediction"]
    st.divider()
    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        probability = float(result["fraud_probability"])
        st.metric("Fraud Probability", f"{probability:.2%}")

    with col2:
        status = result["fraud_status"]

        if status == "FRAUD":
            st.error(f"🚨 {status}")
        else:
            st.success(f"✓ {status}")

    with col3:
        risk = result["risk_level"]
        st.metric("Risk Level", risk)

    # ANOMALY ANALYSIS
    st.subheader("Anomaly Analysis")
    anomaly_count = int(result["anomaly_count"])
    st.metric("Detected Anomaly Flags", anomaly_count)
    st.info(result["anomaly_description"])

    # ANOMALY FLAG DETAILS
    anomaly_flags = {
        "large_transaction": "Large transaction",
        "origin_zero_balance": "Origin started with zero balance",
        "destination_zero_balance": "Destination started with zero balance",
        "origin_drained": "Origin balance ended at zero",
        "full_balance_transfer": "Full balance transfer"
    }

    detected_flags = []

    for flag, description in anomaly_flags.items():

        if int(result[flag]) == 1:
            detected_flags.append(description)

    if detected_flags:

        for flag_description in detected_flags:
            st.warning(f"⚠️ {flag_description}")

    else:
        st.success("✓ No major anomaly flags detected.")

    # MODEL DECISION
    st.subheader("Model Decision")
    st.write(f"**Fraud probability:** " f"{probability:.4f}")
    st.write(f"**Fraud status:** " f"{status}")
    st.write(f"**Risk classification:** " f"{risk}")

    # TRANSACTION SUMMARY
    st.subheader("Transaction Summary")

    summary = pd.DataFrame(
        {
            "Feature": ["Transaction Type", "Amount", "Origin Balance Before", "Origin Balance After", "Destination Balance Before",
                        "Destination Balance After"],
            "Value": [transaction_type, f"{amount:,.2f}", f"{oldbalance_org:,.2f}", f"{newbalance_orig:,.2f}", f"{oldbalance_dest:,.2f}",
                      f"{newbalance_dest:,.2f}"]
        }
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)
    
#MODEL INFORMATION
st.divider()
st.subheader("Model Information")
info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:
    st.metric("Model", MODEL_METADATA["model"])

with info_col2:
    st.metric("Calibration", MODEL_METADATA["calibration_method"])

with info_col3:
    st.metric("Features", MODEL_METADATA["feature_count"])

with info_col4:
    st.metric("Decision Threshold", f'{MODEL_METADATA["decision_threshold"]:.2f}')
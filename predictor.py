import joblib
import numpy as np
import pandas as pd

from pathlib import Path

# PROJECT PATHS
PROJECT_DIR = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODELS_DIR / "final_calibrated_fraud_model.pkl"
ENCODER_PATH = MODELS_DIR / "onehot_encoder.pkl"
THRESHOLD_PATH = MODELS_DIR / "fraud_decision_threshold.pkl"
LARGE_TRANSACTION_THRESHOLD_PATH = MODELS_DIR / "large_transaction_threshold.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.pkl"

# PRODUCTION ARTIFACT VALIDATION
required_artifacts = {
    "Final calibrated model": MODEL_PATH,
    "One-Hot Encoder": ENCODER_PATH,
    "Fraud decision threshold": THRESHOLD_PATH,
    "Large transaction threshold": LARGE_TRANSACTION_THRESHOLD_PATH,
    "Feature configuration": FEATURE_CONFIG_PATH
}
missing_artifacts = [
    name
    for name, path in required_artifacts.items()
    if not path.exists()
]
if missing_artifacts:
    raise FileNotFoundError(
        "Missing production artifacts:\n"
        + "\n".join(
            f"- {item}"
            for item in missing_artifacts
        )
    )

# LOAD PRODUCTION ARTIFACTS
final_model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
fraud_threshold = float(joblib.load(THRESHOLD_PATH))
large_transaction_threshold = float(joblib.load(LARGE_TRANSACTION_THRESHOLD_PATH))
feature_config = joblib.load(FEATURE_CONFIG_PATH)

print("Production artifacts loaded successfully.")

# FEATURE CONFIGURATION
TARGET = feature_config["target"]
CATEGORICAL_FEATURES = feature_config["categorical_features"]
NUMERIC_FEATURES = feature_config["numeric_features"]
EXPECTED_FEATURE_NAMES = feature_config["encoded_feature_names"]
print(f"Expected model features: " f"{len(EXPECTED_FEATURE_NAMES)}")

# FEATURE ENGINEERING FUNCTION
def create_fraud_features(df):
    """
    Reproduce the feature engineering logic
    used during model training.
    """
    data = df.copy()
    EPSILON = 1e-6

    # Balance differences
    data["origin_balance_diff"] = (data["oldbalanceOrg"] - data["newbalanceOrig"])
    data["destination_balance_diff"] = (data["newbalanceDest"] - data["oldbalanceDest"])

    # Amount-to-balance ratios
    data["origin_amount_ratio"] = (data["amount"] / (data["oldbalanceOrg"] + EPSILON))
    data["destination_amount_ratio"] = (data["amount"] / (data["oldbalanceDest"] + EPSILON))

    # Training-derived large transaction flag
    data["large_transaction"] = (data["amount"] >= large_transaction_threshold).astype(int)

    # Zero balance flags
    data["origin_zero_balance"] = (data["oldbalanceOrg"] == 0).astype(int)
    data["destination_zero_balance"] = (data["oldbalanceDest"] == 0).astype(int)

    # Drained origin account
    data["origin_drained"] = (data["newbalanceOrig"] == 0).astype(int)

    # Time features
    data["hour"] = (data["step"] % 24)
    data["day"] = (data["step"] // 24)
    data["day_of_week"] = (data["day"] % 7)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Log transaction amount
    data["log_amount"] = np.log1p(data["amount"])

    # Origin balance change percentage
    data["origin_balance_change_pct"] = (data["origin_balance_diff"] / (data["oldbalanceOrg"] + EPSILON))

    # Full balance transfer
    data["full_balance_transfer"] = (np.isclose(data["amount"], data["oldbalanceOrg"], rtol=0, atol=1e-6)).astype(int)
    return data

# INPUT VALIDATION
REQUIRED_INPUT_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

def validate_input(df):
    """
    Validate raw transaction input.
    """
    missing_columns = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError("Missing required transaction fields: " + ", ".join(missing_columns))
    if df.empty:
        raise ValueError("Transaction input is empty.")

    numeric_columns = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' must be numeric.")

    if df["amount"].isna().any():
        raise ValueError("Transaction amount cannot be missing.")
        
    if (df["amount"] < 0).any():
        raise ValueError("Transaction amount cannot be negative.")

# PRODUCTION PREPROCESSING
def preprocess_transaction(df):
    """
    Apply the exact preprocessing required
    by the trained model.
    """
    data = create_fraud_features(df)

    # Remove training-only / non-predictive fields
    columns_to_remove = [
        column
        for column in ["nameOrig", "nameDest", "isFlaggedFraud", "isFraud"]
        
        if column in data.columns
    ]
    data = data.drop(columns=columns_to_remove)

    # Separate categorical and numeric features
    encoded_categorical = pd.DataFrame(encoder.transform(data[CATEGORICAL_FEATURES]),
                                       columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES), index=data.index)
    
    numeric_data = data.drop(columns=CATEGORICAL_FEATURES)

    # Combine numeric + encoded categorical features
    processed = pd.concat([numeric_data, encoded_categorical], axis=1)

    # Ensure exact training feature order
    processed = processed.reindex(columns=EXPECTED_FEATURE_NAMES, fill_value=0)
    return processed.astype(np.float32)

# FRAUD PREDICTION FUNCTION
def predict_transaction(transaction):
    """
    Predict fraud probability and fraud status
    for one or more transactions.
    """
    if isinstance(transaction, dict):
        transaction_df = pd.DataFrame([transaction])
        
    elif isinstance(transaction, pd.DataFrame):
        transaction_df = transaction.copy()

    else:
        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    # Validate input
    validate_input(transaction_df)

    # Keep anomaly information before preprocessing
    engineered = create_fraud_features(transaction_df)

    # Generate model-ready features
    X_processed = preprocess_transaction(transaction_df)

    # Generate calibrated fraud probability
    fraud_probability = (final_model.predict_proba(X_processed)[:, 1])

    # Apply optimized threshold
    fraud_prediction = fraud_probability >= fraud_threshold
    fraud_prediction = fraud_prediction.astype(int)

    results = transaction_df.copy()
    results["fraud_probability"] = (fraud_probability)
    results["fraud_prediction"] = (fraud_prediction)
    results["fraud_status"] = np.where(fraud_prediction == 1, "FRAUD", "NORMAL")
    
    # ANOMALY FLAGS
    anomaly_flags = ["large_transaction", "origin_zero_balance", "destination_zero_balance", "origin_drained", "full_balance_transfer"]

    for flag in anomaly_flags:
        results[flag] = engineered[flag].astype(int).values

    # TOTAL NUMBER OF ANOMALY FLAGS
    results["anomaly_count"] = (engineered[anomaly_flags] .sum(axis=1) .astype(int) .values)

    # HUMAN-READABLE ANOMALY DESCRIPTION
    def get_anomaly_description(row):
        detected = []
        if row["large_transaction"] == 1:
            detected.append("Large transaction")

        if row["origin_zero_balance"] == 1:
            detected.append("Origin account had zero balance")

        if row["destination_zero_balance"] == 1:
            detected.append("Destination account had zero balance")

        if row["origin_drained"] == 1:
            detected.append("Origin account drained")

        if row["full_balance_transfer"] == 1:
            detected.append("Full balance transfer")

        if not detected:
            return "No major anomaly flags"

        return "; ".join(detected)

    results["anomaly_description"] = results.apply(get_anomaly_description, axis=1)


    # RISK LEVEL
    def assign_risk_level(probability):

        if probability >= 0.90:
            return "CRITICAL"

        elif probability >= 0.50:
            return "HIGH"

        elif probability >= fraud_threshold:
            return "MEDIUM"

        else:
            return "LOW"

    results["risk_level"] = [assign_risk_level(probability)
        for probability in fraud_probability
    ]

    return results, engineered

# TEST THE PRODUCTION PREDICTION PIPELINE
if __name__ == "__main__":

    test_transaction = {"step": 1, "type": "TRANSFER", "amount": 100000.0, "oldbalanceOrg": 100000.0, "newbalanceOrig": 0.0,
                        "oldbalanceDest": 0.0, "newbalanceDest": 100000.0}

    prediction, engineered = predict_transaction(test_transaction)

    print("\nPRODUCTION PREDICTION TEST")

    print("\nPREDICTION RESULTS")
    print(prediction.to_string(index=False))

    print("\nENGINEERED FRAUD FEATURES")
    print(engineered.to_string(index=False))
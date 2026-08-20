# ============================================================
# MOBILE-MONEY FRAUD DETECTION 3MTT PROJECT
# Mobile-Money Fraud Detection Using Machine Learning

# Author : Louis Mbagwu
# IDE    : Visual Studio Code

# Objective:
# Build a model that is capable of detecting fraudulent mobile-money transactions using the PaySim dataset. 
# The final solution will include data preprocessing, feature engineering, model comparison, evaluation,
# fraud risk scoring, and deployment through a Streamlit web application.

# Dataset: PaySim dataset
# ============================================================

# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json

from sklearn.model_selection import (train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import (LabelEncoder, StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, 
                             confusion_matrix, classification_report, roc_curve, matthews_corrcoef, brier_score_loss, precision_recall_curve)
from sklearn.calibration import (calibration_curve, CalibratedClassifierCV)

from sklearn.base import clone

from scipy.stats import randint
from scipy.stats import uniform

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# DISPLAY SETTINGS
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.2f}".format)

# VISUALIZATION SETTINGS
plt.style.use("ggplot")
sns.set_theme(style="whitegrid", context="notebook", palette="deep")

# PROJECT PATHS
PROJECT_DIR = Path.cwd()

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_DIR / "models"

OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#LOAD DATASET
DATASET_FILE = RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv"
df = pd.read_csv(DATASET_FILE)
print("Dataset loaded successfully.")

rows, columns = df.shape
print(f"Number of Rows    : {rows:,}")
print(f"Number of Columns : {columns}")

df.head()
df.tail()

print("Dataset Columns:\n")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

df.info()

#INITIAL DATASET HEALTH CHECK
print("\nDATASET HEALTH CHECK")

print(f"Rows               : {df.shape[0]:,}")
print(f"Columns            : {df.shape[1]}")
print(f"Missing Values     : {df.isnull().sum().sum():,}")
print(f"Duplicate Rows     : {df.duplicated().sum():,}")

print("\nTarget Distribution")
print(df["isFraud"].value_counts())

print("\nFraud Percentage")
fraud_percentage = (df["isFraud"].mean() * 100)
print(f"{fraud_percentage:.4f}%")

#DATA INSPECTION

#descriptive statistics
print(df.describe().T)

#categorical summary
print(df.describe(include="object").T)

#missing values
missing_values = pd.DataFrame({"Missing Values": df.isnull().sum(), "Percentage (%)": (df.isnull().sum() / len(df)) * 100})

missing_values = missing_values[missing_values["Missing Values"] > 0].sort_values(by="Missing Values", ascending=False)

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values)

#duplicate values
duplicates = df.duplicated().sum()
print(f"Duplicate Rows: {duplicates:,}")
if duplicates:
    print[df.duplicated()].head()

#data types
dtype_summary = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.values})
print(dtype_summary)

#unique values
unique_values = pd.DataFrame({"Column": df.columns, "Unique Values": [df[col].nunique() for col in df.columns]})
print(unique_values)

#transaction types
print(df["type"].value_counts())

#target distribution - to confirms the extent of class imbalance
target_distribution = pd.DataFrame({"Count": df["isFraud"].value_counts(), 
                                    "Percentage": (df["isFraud"].value_counts(normalize=True) * 100)})
print(target_distribution)

#Inspect Numeric and Categorical Features
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
print(numeric_columns)

categorical_columns = df.select_dtypes(include="object").columns.tolist()
print(categorical_columns)

#Check for negative values and zero values
columns_to_check = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

for col in columns_to_check:
    negatives = (df[col] < 0).sum()
    print(f"{col:<20}: {negatives:,}")

for col in columns_to_check:
    zeros = (df[col] == 0).sum()
    percentage = (zeros / len(df)) * 100

    print(f"{col:<20}: " f"{zeros:,} ({percentage:.2f}%)")
    
print("\nDATA QUALITY REPORT")

print(f"Rows                    : {df.shape[0]:,}")
print(f"Columns                 : {df.shape[1]}")
print(f"Missing Values          : {df.isnull().sum().sum():,}")
print(f"Duplicate Rows          : {df.duplicated().sum():,}")
print(f"Fraud Cases             : {df['isFraud'].sum():,}")
print(f"Legitimate Cases        : {(df['isFraud'] == 0).sum():,}")
print(f"Fraud Percentage        : {(df['isFraud'].mean() * 100):.4f}%")
print(f"Numeric Features        : {len(numeric_columns)}")
print(f"Categorical Features    : {len(categorical_columns)}")
print("=" * 60)

#DATA CLEANING
#working copy
df_clean = df.copy()
print("Working copy created successfully.")

#check for invalid transaction amounts
invalid_amounts = (df_clean["amount"] < 0).sum()
print(f"Negative Amounts: {invalid_amounts}")

#Validate account balance - Neative balances
balance_columns = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
for column in balance_columns:
    negatives = (df_clean[column] < 0).sum()
    print(f"{column:<20}: {negatives}")
    
#Balance consistency flags - creating flags instead of deleting rows, as this is important for fraud detection and unsual accounting patterns
#so these impossible transactions and suspicious behaviours like zero, negative and incosistent balances won't be removed, so it will be kept
df_clean["origin_balance_error"] = ((df_clean["oldbalanceOrg"] - df_clean["amount"])
                                    != df_clean["newbalanceOrig"]).astype(int)

df_clean["destination_balance_error"] = ((df_clean["oldbalanceDest"] + df_clean["amount"])
                                         != df_clean["newbalanceDest"]).astype(int)

print("Balance inconsistency flags created.")

#Verify targets and target types
print(df_clean["isFraud"].value_counts())
print("\nUnique Values:")
print(df_clean["isFraud"].unique())

print(df_clean["type"].value_counts())
print("\nUnique Types:")
print(df_clean["type"].unique())

#saved clean dataset
clean_dataset_path = (PROCESSED_DATA_DIR /"paysim_clean.csv")
df_clean.to_csv(clean_dataset_path, index=False)
print("Clean dataset saved successfully.")
print(clean_dataset_path)

#Exploratory Data Analysis (EDA)
#Dataset overview
#data shape
print(f"Rows    : {df_clean.shape[0]:,}")
print(f"Columns : {df_clean.shape[1]}")

#target distribution
fraud_counts = df_clean["isFraud"].value_counts()
fraud_percent = (df_clean["isFraud"].value_counts(normalize=True) *10)
summary = pd.DataFrame({"Count": fraud_counts, "Percentage": fraud_percent})
print(summary)

#Fraud distribution visualization
plt.figure(figsize=(6,5))
sns.countplot(data=df_clean, x="isFraud")
plt.title("Fraud Distribution")
plt.xlabel("Fraud")
plt.ylabel("Count")
plt.show()

#Univariate Analysis
#transaction type distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df_clean, x="type", order=df_clean["type"].value_counts().index)
plt.title("Transaction Types")
plt.xticks(rotation=45)
plt.show()

#transaction amount distribution
plt.figure(figsize=(8,5))
plt.hist(np.log1p(df_clean["amount"]), bins=50)
plt.title("Log Transaction Amount Distribution")
plt.xlabel("Log(Amount + 1)")
plt.show()

#boxplot of transaction amount
plt.figure(figsize=(8,5))
sns.boxplot(x=df_clean["amount"])
plt.title("Transaction Amount")
plt.show()

#Bivariate Analysis
#fruad by transaction type
plt.figure(figsize=(8,5))
sns.countplot(data=df_clean, x="type", hue="isFraud")
plt.title("Fraud by Transaction Type")
plt.xticks(rotation=45)
plt.show()

#transaction amount vs fraud - to know if fraudulaent transactions are associated with larger amounts
plt.figure(figsize=(8,5))
sns.countplot(data=df_clean, x="type", hue="isFraud")
plt.title("Fraud by Transaction Type")
plt.xticks(rotation=45)
plt.show()

#original balance before transction
plt.figure(figsize=(8,5))
sns.boxplot(data=df_clean, x="isFraud", y="oldbalanceOrg")
plt.yscale("log")
plt.title("Origin Balance Before Transaction")
plt.show()

#destination balance before transaction
plt.figure(figsize=(8,5))
sns.boxplot(data=df_clean, x="isFraud", y="oldbalanceDest")
plt.yscale("log")
plt.title("Destination Balance Before Transaction")
plt.show()

#Time Analysis
#Transaction Activity Over Time
transactions_per_step = (df_clean.groupby("step").size())
plt.figure(figsize=(12,5))
transactions_per_step.plot()
plt.title("Transactions Over Time")
plt.xlabel("Step")
plt.ylabel("Transactions")
plt.show()

#Fraud Occurrence Over Time
fraud_per_step = (df_clean[df_clean["isFraud"]==1].groupby("step").size())
plt.figure(figsize=(12,5))
fraud_per_step.plot()
plt.title("Fraud Cases Over Time")
plt.show()

#Correlation Analysis
#correlation metrics
numeric_df = df_clean.select_dtypes(include=np.number)
corr = numeric_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#Fraud Profile
#average transaction amount
df_clean.groupby("isFraud")["amount"].mean()

#average balances
df_clean.groupby("isFraud")[["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].mean()

#fraud rate by transaction rate
fraud_rate = (df_clean.groupby("type")["isFraud"].mean().sort_values(ascending=False))
print(fraud_rate)

#Normalized Fraud Rate by Transaction Type
#FRAUD RATE BY TRANSACTION TYPE
fraud_rate = (df_clean.groupby("type")["isFraud"].agg(Total_Transactions="count", Fraud_Cases="sum", Fraud_Rate="mean"))
fraud_rate["Fraud_Rate (%)"] = fraud_rate["Fraud_Rate"] * 100
fraud_rate = fraud_rate.drop(columns="Fraud_Rate")
fraud_rate.sort_values(by="Fraud_Rate (%)", ascending=False, inplace=True)
print(fraud_rate)
#visualization of fruad rate
plt.figure(figsize=(8,5))
sns.barplot(data=fraud_rate.reset_index(), x="type", y="Fraud_Rate (%)")
plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.show()

#Evaluation of Balance Inconsistency Flags
#BALANCE ERROR FLAGS
balance_error_summary = (df_clean.groupby("isFraud")[["origin_balance_error", "destination_balance_error"]].mean())
print(balance_error_summary)
#visualization of balance error flag
balance_error_plot = (balance_error_summary.reset_index().melt(id_vars="isFraud", var_name="Balance Error", value_name="Proportion"))
plt.figure(figsize=(8,5))
sns.barplot(data=balance_error_plot, x="Balance Error", y="Proportion", hue="isFraud")
plt.title("Balance Inconsistency by Fraud Status")
plt.ylabel("Proportion of Transactions")
plt.show()

#Analyzing the Rule-Based Fraud Flag (isFlaggedFraud)
# RULE-BASED FLAG ANALYSIS(cross-tabulation)
flag_summary = pd.crosstab(df_clean["isFlaggedFraud"], df_clean["isFraud"], margins=True)
print(flag_summary)

#Detection Performance
#RULE PERFORMANCE
flagged_transactions = (df_clean["isFlaggedFraud"] == 1).sum()
actual_fraud = (df_clean["isFraud"] == 1).sum()
flagged_fraud = df_clean[(df_clean["isFlaggedFraud"] == 1) &(df_clean["isFraud"] == 1)].shape[0]

print(f"Transactions Flagged : {flagged_transactions:,}")
print(f"Actual Fraud Cases   : {actual_fraud:,}")
print(f"Correctly Flagged    : {flagged_fraud:,}")
print(f"\nRule Detection Rate: " f"{flagged_fraud / actual_fraud * 100:.2f}%")
# Visualize the Rule Performance
plt.figure(figsize=(6,5))
sns.countplot(data=df_clean, x="isFlaggedFraud", hue="isFraud")
plt.title("Rule-Based Fraud Flag vs Actual Fraud")
plt.xlabel("isFlaggedFraud")
plt.ylabel("Number of Transactions")
plt.show()

#FEATURE ENGINEERIING
#FEATURE ENGINEERING DATASET
df_features = df_clean.copy()
print("Feature engineering dataset created successfully.")

#ORIGIN BALANCE DIFFERENCE
df_features["origin_balance_diff"] = (df_features["oldbalanceOrg"] - df_features["newbalanceOrig"])
#DESTINATION BALANCE DIFFERENCE
df_features["destination_balance_diff"] = (df_features["newbalanceDest"] - df_features["oldbalanceDest"])

#AMOUNT TO ORIGIN BALANCE RATIO
EPSILON = 1e-6
df_features["origin_amount_ratio"] = (df_features["amount"] / (df_features["oldbalanceOrg"] + EPSILON))
#AMOUNT TO DESTINATION BALANCE RATIO
df_features["destination_amount_ratio"] = (df_features["amount"] / (df_features["oldbalanceDest"] + EPSILON))

#LARGE TRANSACTION FLAG
large_transaction_threshold = (df_features["amount"].quantile(0.95))
df_features["large_transaction"] = (df_features["amount"] >= large_transaction_threshold).astype(int)

#ZERO BALANCE FLAGS
df_features["origin_zero_balance"] = (df_features["oldbalanceOrg"] == 0).astype(int)
df_features["destination_zero_balance"] = (df_features["oldbalanceDest"] == 0).astype(int)

#EMPTY ORIGIN ACCOUNT
df_features["origin_drained"] = (df_features["newbalanceOrig"] == 0).astype(int)

#HOUR OF DAY
df_features["hour"] = (df_features["step"] % 24)
#DAY NUMBER
df_features["day"] = (df_features["step"] // 24)
#WEEKEND FLAG
df_features["day_of_week"] = (df_features["day"] % 7)
df_features["is_weekend"] = (df_features["day_of_week"] >= 5).astype(int)

#LOG AMOUNT - amount are highly skewed
df_features["log_amount"] = np.log1p(df_features["amount"])

#BALANCE CHANGE PERCENTAGE
df_features["origin_balance_change_pct"] = (df_features["origin_balance_diff"] / (df_features["oldbalanceOrg"] + EPSILON))

#FULL BALANCE TRANSFER
df_features["full_balance_transfer"] = (np.isclose(df_features["amount"], df_features["oldbalanceOrg"], rtol=0, atol=1e-6)).astype(int)

#NEW FEATURES CREATED
new_features = ["origin_balance_diff", "destination_balance_diff", "origin_amount_ratio", "destination_amount_ratio", 
                "large_transaction", "origin_zero_balance", "destination_zero_balance", "origin_drained", "hour", "day", "day_of_week",
                "is_weekend", "log_amount", "origin_balance_change_pct", "full_balance_transfer"]
print("New Features Created:\n")

for i, feature in enumerate(new_features, start=1):
    print(f"{i}. {feature}")
print(f"\nTotal New Features: {len(new_features)}")

#PREVIEW ENGINEERED DATASET
df_features[new_features].head()

#SAVED FEATURE DATASET
feature_dataset_path = (PROCESSED_DATA_DIR / "paysim_features.csv")
df_features.to_csv(feature_dataset_path, index=False)
print("Feature dataset saved successfully.")
print(feature_dataset_path)

#FEATURE SELECTION
#FEATURE SELECTION DATASET
df_final = df_features.copy()
print("Feature selection dataset created successfully.")

#REVIEW FEATURES
print(f"Total Columns: {len(df_final.columns)}\n")
for i, col in enumerate(df_final.columns, start=1):
    print(f"{i:>2}. {col}")
    
#REMOVE NON-PREDICTIVE FEATURES
columns_to_remove = ["nameOrig", "nameDest", "isFlaggedFraud"]
df_final.drop(columns=columns_to_remove, inplace=True)
print("Columns removed successfully.")
print("\nRemaining Features:", len(df_final.columns))

#IDENTIFY TARGET VARIABLE - Separate predictors from the target conceptually (without splitting yet).
TARGET = "isFraud"
print("Target Variable:", TARGET)

#REVIEW FINAL FEATURES
predictor_columns = [
    col for col in df_final.columns
    if col != TARGET
]
print(f"Predictor Features: {len(predictor_columns)}\n")
for feature in predictor_columns:
    print(feature)
    
#FEATURE TYPES
feature_summary = pd.DataFrame({"Feature": predictor_columns, "Data Type": df_final[predictor_columns].dtypes.values})
print(feature_summary)

#SAVED MODELING DATASET
model_dataset_path = (PROCESSED_DATA_DIR / "paysim_model_dataset.csv")
df_final.to_csv(model_dataset_path, index=False)
print("Model dataset saved successfully.")
print(model_dataset_path)

#Train-Test Split & Training-Only Preprocessing

#REMOVING LEAKAGE-PRONE FEATURE
if "large_transaction" in df_final.columns:
    df_final.drop(columns=["large_transaction"], inplace=True)
print("'large_transaction' removed successfully.")

#Predictors(x) and Target(y)
TARGET = "isFraud"
X = df_final.drop(columns=[TARGET])
y = df_final[TARGET]
print(f"Predictors Shape : {X.shape}")
print(f"Target Shape     : {y.shape}")

#Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Training Samples : {len(X_train):,}")
print(f"Testing Samples  : {len(X_test):,}")

#STRATIFICATION VERIFICATION
print("\nTraining Target Distribution")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting Target Distribution")
print(y_test.value_counts(normalize=True) * 100)

#Recreating large_transaction Using Only the Training Data
large_transaction_threshold = X_train["amount"].quantile(0.95)
X_train["large_transaction"] = (X_train["amount"] >= large_transaction_threshold).astype(int)
X_test["large_transaction"] = (X_test["amount"] >= large_transaction_threshold).astype(int)
print(f"95th Percentile Threshold: {large_transaction_threshold:,.2f}")

#Categorical and Numeric Features
categorical_features = ["type"]

numeric_features = [
    col for col in X_train.columns
    if col not in categorical_features
]
print("Categorical Features:")
print(categorical_features)

print("\nNumeric Features:")
print(numeric_features)

#Fitting the Encoder on the Training Set - will be using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(X_train[categorical_features])

#TRANSFORMING CATEGORICAL FEATURES
encoded_train = pd.DataFrame(encoder.transform(X_train[categorical_features]), columns=encoder.get_feature_names_out(categorical_features),
                             index=X_train.index)

encoded_test = pd.DataFrame(encoder.transform(X_test[categorical_features]), columns=encoder.get_feature_names_out(categorical_features),
                            index=X_test.index)

#Combining Encoded and Numeric Features to build final model dataset
X_train_encoded = pd.concat([X_train.drop(columns=categorical_features), encoded_train], axis=1)
X_test_encoded = pd.concat([X_test.drop(columns=categorical_features), encoded_test], axis=1)

print("Training Shape:", X_train_encoded.shape)
print("Testing Shape :", X_test_encoded.shape)

#Scaling the Numeric Features (Only for Linear Models, as tree-based models don't require scaling)
#STANDARD SCALING
scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train_encoded[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test_encoded[numeric_features])

#SAVE PREPROCESSING OBJECTS
joblib.dump(encoder, MODELS_DIR / "onehot_encoder.pkl")
joblib.dump(scaler, MODELS_DIR / "standard_scaler.pkl")
joblib.dump(large_transaction_threshold, MODELS_DIR / "large_transaction_threshold.pkl")

print("Preprocessing objects saved successfully.")

#DATASET SUMMARY
print("\nTRAINING DATA SUMMARY")

print(f"Training Samples      : {len(X_train):,}")
print(f"Testing Samples       : {len(X_test):,}")

print(f"Encoded Features      : {X_train_encoded.shape[1]}")

print(f"Fraud Cases (Train)   : {y_train.sum():,}")
print(f"Fraud Cases (Test)    : {y_test.sum():,}")

#Handling Class Imbalance
# PaySim contains millions of transactions and an extremely rare fraud class.
# Full-dataset SMOTE is intentionally NOT used because it attempts to create millions of synthetic observations and can exhaust RAM.
# Instead, class weighting is used for the main modeling workflow.

#Check Class Distribution Before Resampling
print("\nTraining Set")
print(y_train.value_counts())
print("\nPercentage")
print(y_train.value_counts(normalize=True) * 100)

#Visualizing class distribution (Imbalance)
plt.figure(figsize=(6,5))
sns.countplot(x=y_train)
plt.title("Training Set Class Distribution")
plt.xlabel("Fraud")
plt.ylabel("Transactions")
plt.show()

#CREATE CALIBRATION AND THRESHOLD VALIDATION HOLDOUTS
#First reserve 10% of the original training data for calibration + threshold optimization.
X_train_model, X_holdout, y_train_model, y_holdout = train_test_split(X_train_encoded, y_train, 
                                                                      test_size=0.10, stratify=y_train, random_state=42)

# Splitting the 10% holdout equally: 5% for calibration, 5% for threshold validation
X_calibration, X_threshold_validation, y_calibration, y_threshold_validation = train_test_split(
    X_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=42)

# Reduce memory usage
X_train_model = X_train_model.astype(np.float32)
X_calibration = X_calibration.astype(np.float32)
X_threshold_validation = X_threshold_validation.astype(np.float32)

y_train_model = y_train_model.copy()
y_calibration = y_calibration.copy()
y_threshold_validation = y_threshold_validation.copy()

# Scaled versions for Logistic Regression
X_train_scaled_model = X_train_scaled.loc[X_train_model.index].astype(np.float32)
X_calibration_scaled = X_train_scaled.loc[X_calibration.index].astype(np.float32)
X_threshold_validation_scaled = X_train_scaled.loc[X_threshold_validation.index].astype(np.float32)

print("\nTraining, calibration and threshold-validation datasets prepared.")

print(f"Model training shape       : {X_train_model.shape}")
print(f"Calibration shape          : {X_calibration.shape}")
print(f"Threshold validation shape : {X_threshold_validation.shape}")

print(f"Training fraud cases       : {y_train_model.sum():,}")
print(f"Calibration fraud cases    : {y_calibration.sum():,}")
print(f"Threshold fraud cases      : {y_threshold_validation.sum():,}")

# HYPERPARAMETER TUNING DATASET
# Using a 300,000-row stratified subset of the ORIGINAL training distribution. No synthetic observations are created.
X_tune_scaled, _, y_tune_scaled, _ = train_test_split(X_train_scaled_model, y_train_model, 
                                                      train_size=300000, stratify=y_train_model, random_state=42)

X_tune_tree, _, y_tune_tree, _ = train_test_split(X_train_model, y_train_model, train_size=300000, stratify=y_train_model, random_state=42)

print("\nHyperparameter tuning datasets created.")
print(f"Scaled tuning shape: {X_tune_scaled.shape}")
print(f"Tree-model tuning shape: {X_tune_tree.shape}")
print(f"Tuning fraud cases: {y_tune_tree.sum():,}")
print(f"Tuning fraud rate: {y_tune_tree.mean():.6%}")

# CLASS-WEIGHTED MODEL BUILDING
negative_class = (y_train_model == 0).sum()
positive_class = (y_train_model == 1).sum()
scale_pos_weight = negative_class / positive_class

print(f"\nMajority Class : {negative_class:,}")
print(f"Minority Class : {positive_class:,}")
print(f"scale_pos_weight : {scale_pos_weight:.2f}")

# Baseline models using class weighting / scale_pos_weight.
logistic_model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
decision_tree_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
random_forest_model = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42, n_jobs=1)
xgboost_model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric="logloss", n_estimators=200,
                              learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=1)

# Train baseline weighted models.
print("\nTraining baseline class-weighted models...")

logistic_model.fit(X_train_scaled_model, y_train_model)
print("Logistic Regression training complete.")

decision_tree_model.fit(X_train_model, y_train_model)
print("Decision Tree training complete.")

random_forest_model.fit(X_train_model, y_train_model)
print("Random Forest training complete.")

xgboost_model.fit(X_train_model, y_train_model)
print("XGBoost training complete.")

# Save baseline weighted models.
joblib.dump(logistic_model, MODELS_DIR / "weighted_logistic_regression.pkl")
joblib.dump(decision_tree_model, MODELS_DIR / "weighted_decision_tree.pkl")
joblib.dump(random_forest_model, MODELS_DIR / "weighted_random_forest.pkl")
joblib.dump(xgboost_model, MODELS_DIR / "weighted_xgboost.pkl")
print("Weighted baseline models saved successfully.")

# OPTIONAL ORIGINAL BASELINE MODELS
# These models are retained as an experimental comparison against the class-weighted strategy. They use the original class distribution.

models_original = {"Logistic Regression": LogisticRegression(random_state=42, max_iter=1000), 
                   "Decision Tree": DecisionTreeClassifier(random_state=42), 
                   "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
                   "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=200, n_jobs=1)}

models_weighted = {"Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
                   "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
                   "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42, n_jobs=1),
                   "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric="logloss", n_estimators=200, n_jobs=1)}

# Reusable training function.
def train_models(model_dict, X, y):
    trained_models = {}

    for name, model in model_dict.items():
        print(f"Training {name}...")
        model.fit(X, y)
        trained_models[name] = model

    return trained_models

# Original strategy.
trained_original = {}
trained_original["Logistic Regression"] = models_original["Logistic Regression"].fit(X_train_scaled_model, y_train_model)

for model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
    trained_original[model_name] = models_original[model_name].fit(X_train_model, y_train_model)

# Weighted strategy.
trained_weighted = {}
trained_weighted["Logistic Regression"] = models_weighted["Logistic Regression"].fit(X_train_scaled_model, y_train_model)

for model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
    trained_weighted[model_name] = models_weighted[model_name].fit(X_train_model, y_train_model)

# Save original and weighted models.
for strategy_name, model_collection in {"original": trained_original, "weighted": trained_weighted}.items():

    for model_name, model in model_collection.items():
        filename = (f"{strategy_name}_" f"{model_name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, MODELS_DIR / filename)

print("Original and weighted models saved successfully.")

# HYPERPARAMETER TUNING (300K STRATIFIED SUBSET)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
SCORING = "average_precision"

# Hyperparameter space.
logistic_params = {"C": uniform(0.001, 5), "solver": ["lbfgs", "liblinear"], "penalty": ["l2"]}

tree_params = {"criterion": ["gini", "entropy"], "max_depth": randint(3, 30), "min_samples_split": randint(2, 20), 
               "min_samples_leaf": randint(1, 10)}

rf_params = {"n_estimators": randint(100, 300), "max_depth": randint(5, 30), "min_samples_split": randint(2, 10), 
             "min_samples_leaf": randint(1, 5), "max_features": ["sqrt", "log2"]}

xgb_params = {"n_estimators": randint(100, 400), "max_depth": randint(3, 12), "learning_rate": uniform(0.01, 0.25),
              "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4)}

# Reusable randomized-search function.
def tune_model(model, param_grid, X, y, model_name, n_iter=10):
    print(f"\nTuning {model_name}...")

    search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, scoring=SCORING, 
                                cv=cv, random_state=42, n_jobs=1, verbose=2, refit=True)

    search.fit(X, y)
    print("\nBest Score:")
    print(search.best_score_)
    print("\nBest Parameters:")
    print(search.best_params_)

    return search

# Tune weighted models on the 300K representative subset.
logistic_search = tune_model(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
                             logistic_params, X_tune_scaled, y_tune_scaled, "Logistic Regression")

tree_search = tune_model(DecisionTreeClassifier(class_weight="balanced", random_state=42), tree_params, X_tune_tree, y_tune_tree, "Decision Tree")

rf_search = tune_model(RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1), 
                       rf_params, X_tune_tree, y_tune_tree, "Random Forest")

xgb_search = tune_model(XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric="logloss", n_jobs=1),
                        xgb_params, X_tune_tree, y_tune_tree, "XGBoost")

# Extract optimized models.
best_models = {"Logistic Regression": logistic_search.best_estimator_, "Decision Tree": tree_search.best_estimator_, 
               "Random Forest": rf_search.best_estimator_, "XGBoost": xgb_search.best_estimator_}
print("\nRetraining best weighted models on the full model-development training data...")

# Retrain the selected hyperparameters on all available model-development training data.
best_models["Logistic Regression"].fit(X_train_scaled_model, y_train_model)

for model_name in ["Decision Tree", "Random Forest", "XGBoost"]: best_models[model_name].fit(X_train_model, y_train_model)

print("Retraining complete.")

# Save the optimized models.
for name, model in best_models.items():
    filename = ("best_" + name.lower().replace(" ", "_") + ".pkl")
    joblib.dump(model, MODELS_DIR / filename)

print("Optimized models saved successfully.")

#FINAL MODEL EVALUATION
#FINAL MODELS
optimized_models = {"Logistic Regression": best_models["Logistic Regression"], "Decision Tree": best_models["Decision Tree"],
                    "Random Forest": best_models["Random Forest"], "XGBoost": best_models["XGBoost"]}

#MODEL EVALUATION FUNCTION - creating a reusable evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {"Accuracy": accuracy_score(y_test, y_pred), "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
               "Precision": precision_score(y_test, y_pred, zero_division=0), "Recall": recall_score(y_test, y_pred, zero_division=0),
               "F1 Score": f1_score(y_test, y_pred, zero_division=0), "ROC-AUC": roc_auc_score(y_test, y_prob),
               "PR-AUC": average_precision_score(y_test, y_prob), "MCC": matthews_corrcoef(y_test, y_pred)}
    
    return metrics, y_pred, y_prob

#MODEL EVALUATION
results = []
predictions = {}
probabilities = {}

for model_name, model in optimized_models.items():
    print(f"Evaluating {model_name}...")

    if model_name == "Logistic Regression":
        X_eval = X_test_scaled

    else:
        X_eval = X_test_encoded

    metrics, y_pred, y_prob = evaluate_model(model, X_eval, y_test)
    metrics["Model"] = model_name
    results.append(metrics)
    predictions[model_name] = y_pred
    probabilities[model_name] = y_prob

print("Evaluation complete.")

#RESULTS TABLE
results_df = pd.DataFrame(results)
results_df = results_df[["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC", "MCC"]]
print(results_df)

#MODEL RANKING - priotirizing fraud detection metrics, rather than ranking by accuracy
results_df = results_df.sort_values(by=["PR-AUC", "Recall", "F1 Score", "MCC"], ascending=False).reset_index(drop=True)
results_df.index += 1
print(results_df)

#CLASSIFICATION REPORTS
for model_name, model in optimized_models.items():
    print(model_name)

    if model_name == "Logistic Regression":
        X_eval = X_test_scaled

    else:
        X_eval = X_test_encoded

    y_pred = model.predict(X_eval)

    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    
#CONFUSION MATRICES
for model_name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
#ROC CURVES
plt.figure(figsize=(8,6))
for model_name, y_prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=model_name)

plt.plot([0,1], [0,1], linestyle="--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

#PRECISION-RECALL CURVES
plt.figure(figsize=(8,6))
for model_name, y_prob in probabilities.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=model_name)

plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

#SAVED RESULTS
evaluation_path = (RESULTS_DIR /"optimized_model_results.csv")
results_df.to_csv(evaluation_path, index=False)
print("Evaluation results saved successfully.")
print(evaluation_path)

#MODEL CALIBRATION
#IDENTIFY BEST-PERFORMING MODEL
best_model_name = results_df.iloc[0]["Model"]
best_model = best_models[best_model_name]
print("\nBest-performing model selected for calibration:")
print(best_model_name)

#SELECT CALIBRATION FEATURES
if best_model_name == "Logistic Regression":
    X_calibration_eval = X_calibration_scaled

else:
    X_calibration_eval = X_calibration

print(f"Calibration feature shape: " f"{X_calibration_eval.shape}")

#UNCALIBRATED PROBABILITIES
uncalibrated_probabilities = best_model.predict_proba(X_calibration_eval)[:, 1]
print("Uncalibrated fraud probabilities generated.")

#UNCALIBRATED BRIER SCORE
uncalibrated_brier = brier_score_loss(y_calibration, uncalibrated_probabilities)
print(f"Uncalibrated Brier Score: " f"{uncalibrated_brier:.6f}")

#UNCALIBRATED RELIABILITY DIAGRAM
prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(y_calibration, uncalibrated_probabilities, n_bins=10, strategy="quantile")

plt.figure(figsize=(8, 6))
plt.plot(prob_pred_uncalibrated, prob_true_uncalibrated, marker="o", label="Uncalibrated Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.title(f"Calibration Curve - {best_model_name}")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Fraud Rate")
plt.legend()
plt.show()

#SIGMOID / PLATT CALIBRATION
try:
    from sklearn.frozen import FrozenEstimator
    calibrated_sigmoid = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method="sigmoid")

except ImportError:
    calibrated_sigmoid = CalibratedClassifierCV(estimator=best_model, method="sigmoid", cv="prefit")
    
calibrated_sigmoid.fit(X_calibration_eval, y_calibration)
sigmoid_probabilities = calibrated_sigmoid.predict_proba(X_calibration_eval)[:, 1]
sigmoid_brier = brier_score_loss(y_calibration, sigmoid_probabilities)

print(f"Sigmoid Brier Score: " f"{sigmoid_brier:.6f}")

#ISOTONIC CALIBRATION
try:
    calibrated_isotonic = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method="isotonic")

except NameError:
    calibrated_isotonic = CalibratedClassifierCV(estimator=best_model, method="isotonic", cv="prefit")

calibrated_isotonic.fit(X_calibration_eval, y_calibration)
isotonic_probabilities = calibrated_isotonic.predict_proba(X_calibration_eval)[:, 1]
isotonic_brier = brier_score_loss(y_calibration, isotonic_probabilities)

print(f"Isotonic Brier Score: " f"{isotonic_brier:.6f}")

#CALIBRATION COMPARISON
calibration_results = pd.DataFrame({"Method": ["Uncalibrated", "Sigmoid", "Isotonic"], 
                                    "Brier Score": [uncalibrated_brier, sigmoid_brier, isotonic_brier]})

calibration_results = calibration_results.sort_values(by="Brier Score", ascending=True).reset_index(drop=True)
print("\nCalibration Results:")
print(calibration_results)

#CALIBRATION CURVE COMPARISON
prob_true_sigmoid, prob_pred_sigmoid = calibration_curve(y_calibration, sigmoid_probabilities, n_bins=10, strategy="quantile")
prob_true_isotonic, prob_pred_isotonic = calibration_curve(y_calibration, isotonic_probabilities, n_bins=10, strategy="quantile")

plt.figure(figsize=(8, 6))
plt.plot(prob_pred_uncalibrated, prob_true_uncalibrated, marker="o", label="Uncalibrated")
plt.plot(prob_pred_sigmoid, prob_true_sigmoid, marker="o", label="Sigmoid")
plt.plot(prob_pred_isotonic, prob_true_isotonic, marker="o", label="Isotonic")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.title(f"Probability Calibration - {best_model_name}")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Fraud Rate")
plt.legend()
plt.show()

#SELECT BEST CALIBRATION METHOD
best_calibration_method = calibration_results.iloc[0]["Method"]
print(f"\nBest calibration method: " f"{best_calibration_method}")

#SELECT FINAL CALIBRATED MODEL
if best_calibration_method == "Sigmoid":
    final_calibrated_model = calibrated_sigmoid

elif best_calibration_method == "Isotonic":
    final_calibrated_model = calibrated_isotonic

else:
    final_calibrated_model = best_model

print(f"Final calibration selected: " f"{best_calibration_method}")

# SAVE CALIBRATION RESULTS
calibration_results_path = (RESULTS_DIR / "model_calibration_results.csv")
calibration_results.to_csv(calibration_results_path, index=False)
print("\nCalibration results saved successfully.")
print(calibration_results_path)

# SAVE FINAL CALIBRATED MODEL
calibrated_model_path = (MODELS_DIR / "final_calibrated_fraud_model.pkl")
joblib.dump(final_calibrated_model, calibrated_model_path)
print("Final calibrated model saved successfully.")
print(calibrated_model_path)

print(f"{uncalibrated_brier:.12f}")

#FRAUD DECISION THRESHOLD OPTIMIZATION
#THRESHOLD-VALIDATION PROBABILITIES

if best_model_name == "Logistic Regression":
    X_threshold_eval = X_threshold_validation_scaled
else:
    X_threshold_eval = X_threshold_validation

threshold_probabilities = final_calibrated_model.predict_proba(X_threshold_eval)[:, 1]
print("\nThreshold-validation fraud probabilities generated.")
print(f"Threshold-validation feature shape: " f"{X_threshold_eval.shape}")

#DEFINE DECISION THRESHOLDS
thresholds = np.arange(0.01, 1.00, 0.01)
threshold_results = []

#EVALUATE EACH DECISION THRESHOLD
for threshold in thresholds:
    y_pred_threshold = (threshold_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_threshold_validation, y_pred_threshold).ravel()
    precision = precision_score(y_threshold_validation, y_pred_threshold, zero_division=0)
    recall = recall_score(y_threshold_validation, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_threshold_validation, y_pred_threshold, zero_division=0)

    threshold_results.append({"Threshold": threshold, "Precision": precision, "Recall": recall, "F1 Score": f1, "True Negatives": tn,
                              "False Positives": fp, "False Negatives": fn, "True Positives": tp})
    
threshold_results_df = pd.DataFrame(threshold_results)
print("\nThreshold analysis completed.")
print(threshold_results_df.head())

#IDENTIFY BEST F1 THRESHOLD
best_f1_row = threshold_results_df.loc[threshold_results_df["F1 Score"].idxmax()]
best_f1_threshold = best_f1_row["Threshold"]
print("\nBEST F1-BASED FRAUD THRESHOLD")

print(f"Threshold       : {best_f1_threshold:.2f}")
print(f"Precision       : {best_f1_row['Precision']:.6f}")
print(f"Recall          : {best_f1_row['Recall']:.6f}")
print(f"F1 Score        : {best_f1_row['F1 Score']:.6f}")
print(f"False Positives : {int(best_f1_row['False Positives']):,}")
print(f"False Negatives : {int(best_f1_row['False Negatives']):,}")

#THRESHOLD PERFORMANCE CURVES
plt.figure(figsize=(10, 6))
plt.plot(threshold_results_df["Threshold"], threshold_results_df["Precision"], label="Precision")
plt.plot(threshold_results_df["Threshold"], threshold_results_df["Recall"], label="Recall")
plt.plot(threshold_results_df["Threshold"], threshold_results_df["F1 Score"], label="F1 Score")
plt.axvline(best_f1_threshold, linestyle="--", label=(f"Best F1 Threshold = " f"{best_f1_threshold:.2f}"))
plt.title("Fraud Detection Performance by Decision Threshold")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

#COMPARE OPTIMIZED THRESHOLD WITH 0.50
default_threshold_row = threshold_results_df.loc[np.isclose(threshold_results_df["Threshold"], 0.50)].iloc[0]
threshold_comparison = pd.DataFrame({"Metric": ["Threshold", "Precision", "Recall", "F1 Score", "False Positives", "False Negatives"],
                                     "Default 0.50": [0.50, default_threshold_row["Precision"], default_threshold_row["Recall"],
                                                      default_threshold_row["F1 Score"], default_threshold_row["False Positives"],
                                                      default_threshold_row["False Negatives"]],
                                     "Optimized": [best_f1_threshold, best_f1_row["Precision"], best_f1_row["Recall"], best_f1_row["F1 Score"],
                                                   best_f1_row["False Positives"],best_f1_row["False Negatives"]]})

print("\nThreshold Comparison:")
print(threshold_comparison)

#SAVE THRESHOLD ANALYSIS
threshold_results_path = (RESULTS_DIR /"fraud_threshold_analysis.csv")
threshold_results_df.to_csv(threshold_results_path, index=False)
print("\nThreshold analysis saved successfully.")
print(threshold_results_path)

#SAVE RECOMMENDED FRAUD THRESHOLD
threshold_config = {"model": best_model_name, "calibration_method": best_calibration_method,
                    "recommended_threshold": float(best_f1_threshold),"optimization_metric": "F1 Score"}

threshold_config_path = (RESULTS_DIR /"fraud_threshold_config.json")
with open(threshold_config_path, "w") as f:
    json.dump(threshold_config, f, indent=4)

print("\nRecommended threshold configuration saved.")
print(threshold_config_path)

#FINAL UNBIASED TEST EVALUATION
print("\n" + "=" * 70)
print("FINAL UNBIASED TEST EVALUATION")
print("=" * 70)

# The final test set has remained completely untouched throughout model training, hyperparameter tuning, calibration, and threshold optimization.
# Select the correct test representation
if best_model_name == "Logistic Regression":
    X_final_test = X_test_scaled
else:
    X_final_test = X_test_encoded

print(f"\nFinal test feature shape: " f"{X_final_test.shape}")
print(f"Final test samples: " f"{len(y_test):,}")
print(f"Final test fraud cases: " f"{y_test.sum():,}")
print(f"Final test fraud rate: " f"{y_test.mean() * 100:.6f}%")

#GENERATE FINAL CALIBRATED TEST PROBABILITIES
final_test_probabilities = (final_calibrated_model.predict_proba(X_final_test)[:, 1])
print("\nFinal calibrated test probabilities generated.")
print(f"Probability range: " f"{final_test_probabilities.min():.8f} " f"to " f"{final_test_probabilities.max():.8f}")

#APPLY FINAL OPTIMIZED FRAUD THRESHOLD
final_test_predictions = (final_test_probabilities >= best_f1_threshold).astype(int)

print(f"\nFinal fraud decision threshold: " f"{best_f1_threshold:.4f}")
print("\nFinal prediction distribution:")
print(pd.Series(final_test_predictions).value_counts().sort_index())

#FINAL TEST METRICS
final_accuracy = accuracy_score(y_test, final_test_predictions)
final_balanced_accuracy = balanced_accuracy_score(y_test, final_test_predictions)
final_precision = precision_score(y_test, final_test_predictions, zero_division=0)
final_recall = recall_score(y_test, final_test_predictions, zero_division=0)
final_f1 = f1_score(y_test, final_test_predictions, zero_division=0)
final_roc_auc = roc_auc_score(y_test, final_test_probabilities)
final_pr_auc = average_precision_score(y_test, final_test_probabilities)
final_mcc = matthews_corrcoef(y_test, final_test_predictions)
final_brier = brier_score_loss(y_test, final_test_probabilities)

#FRAUD-SPECIFIC METRICS
tn, fp, fn, tp = confusion_matrix(y_test, final_test_predictions).ravel()

final_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
final_specificity = (tn / (tn + fp)if (tn + fp) > 0 else 0)
final_fraud_detection_rate = (tp / (tp + fn)if (tp + fn) > 0 else 0)

print("\nFINAL TEST PERFORMANCE")

print(f"Model                 : {best_model_name}")
print(f"Calibration            : {best_calibration_method}")
print(f"Decision Threshold     : {best_f1_threshold:.6f}")

print(f"\nAccuracy              : {final_accuracy:.6f}")
print(f"Balanced Accuracy     : {final_balanced_accuracy:.6f}")
print(f"Precision             : {final_precision:.6f}")
print(f"Recall                : {final_recall:.6f}")
print(f"F1 Score              : {final_f1:.6f}")
print(f"ROC-AUC               : {final_roc_auc:.6f}")
print(f"PR-AUC                : {final_pr_auc:.6f}")
print(f"MCC                   : {final_mcc:.6f}")
print(f"Brier Score           : {final_brier:.12f}")

print("\nFraud Detection Metrics")
print(f"True Positives        : {tp:,}")
print(f"False Positives       : {fp:,}")
print(f"False Negatives       : {fn:,}")
print(f"True Negatives        : {tn:,}")
print(f"Fraud Detection Rate  : {final_fraud_detection_rate:.6f}")
print(f"False Positive Rate   : {final_fpr:.6f}")
print(f"Specificity           : {final_specificity:.6f}")

#FINAL CLASSIFICATION REPORT
print("\nFINAL CLASSIFICATION REPORT")
print(classification_report(y_test, final_test_predictions, target_names=["Normal", "Fraud"], digits=6, zero_division=0))

#FINAL CONFUSION MATRIX
final_cm = confusion_matrix(y_test, final_test_predictions)
plt.figure(figsize=(7, 6))
sns.heatmap(final_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
plt.title("Final Calibrated Fraud Detection - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

#FINAL ROC CURVE
final_fpr_curve, final_tpr_curve, _ = roc_curve(y_test, final_test_probabilities)
plt.figure(figsize=(8, 6))
plt.plot(final_fpr_curve, final_tpr_curve, label=f"ROC-AUC = {final_roc_auc:.6f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.title("Final ROC Curve - Calibrated Fraud Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#FINAL PRECISION-RECALL CURVE
final_precision_curve, final_recall_curve, _ = (precision_recall_curve(y_test, final_test_probabilities))
plt.figure(figsize=(8, 6))
plt.plot(final_recall_curve, final_precision_curve, label=f"PR-AUC = {final_pr_auc:.6f}")
plt.title("Final Precision-Recall Curve - Calibrated Fraud Model")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#FINAL TEST CALIBRATION CURVE
test_prob_true, test_prob_pred = calibration_curve(y_test, final_test_probabilities, n_bins=10, strategy="quantile")
plt.figure(figsize=(8, 6))
plt.plot(test_prob_pred, test_prob_true, marker="o", label="Final Calibrated Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.title("Final Calibration Curve - Unseen Test Set")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Fraud Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#FINAL RESULTS TABLE
final_results = pd.DataFrame({
    "Model": [best_model_name], "Calibration Method": [best_calibration_method], "Decision Threshold": [best_f1_threshold],
    "Accuracy": [final_accuracy], "Balanced Accuracy": [final_balanced_accuracy], "Precision": [final_precision],
    "Recall": [final_recall], "F1 Score": [final_f1], "ROC-AUC": [final_roc_auc], "PR-AUC": [final_pr_auc],
    "MCC": [final_mcc], "Brier Score": [final_brier], "True Positives": [tp], "False Positives": [fp],
    "False Negatives": [fn], "True Negatives": [tn], "False Positive Rate": [final_fpr], "Specificity": [final_specificity]
})
print("\nFINAL MODEL RESULTS TABLE")
print(final_results.to_string(index=False))

#SAVE FINAL TEST RESULTS
final_test_results_path = (RESULTS_DIR /"final_unbiased_test_results.csv")
final_results.to_csv(final_test_results_path, index=False)
print("\nFinal unbiased test results saved successfully.")
print(final_test_results_path)

#FINAL MODEL & ARTIFACT PACKAGING
print("\nFINAL MODEL & ARTIFACT PACKAGING")
print("\nChecking final model artifacts...")

#VERIFY CORE MODEL ARTIFACTS
required_model_artifacts = {
    "Final Calibrated Model": MODELS_DIR / "final_calibrated_fraud_model.pkl",
    "One-Hot Encoder": MODELS_DIR / "onehot_encoder.pkl",
    "Standard Scaler": MODELS_DIR / "standard_scaler.pkl",
    "Large Transaction Threshold": MODELS_DIR / "large_transaction_threshold.pkl"
}

for artifact_name, artifact_path in required_model_artifacts.items():
    if artifact_path.exists():
        print(f"✓ {artifact_name}: " f"{artifact_path.name}")
    else:
        print(f"✗ MISSING: " f"{artifact_name}")
        
#VERIFY FINAL MODEL LOADING
print("\nLoading final calibrated model...")
loaded_final_model = joblib.load(MODELS_DIR / "final_calibrated_fraud_model.pkl")
print("✓ Final calibrated model loaded successfully.")
print(f"Loaded model type: " f"{type(loaded_final_model).__name__}")

#SAVE FINAL FRAUD DECISION THRESHOLD
threshold_path = (MODELS_DIR /"fraud_decision_threshold.pkl")
joblib.dump(float(best_f1_threshold), threshold_path)
print("\n✓ Final fraud decision threshold saved.")
print(f"Threshold: {best_f1_threshold:.6f}")
print(f"Saved to: {threshold_path}")

#SAVE FINAL FEATURE CONFIGURATION
feature_config = {"target": TARGET, "categorical_features": categorical_features, "numeric_features": numeric_features,
                  "encoded_feature_names": X_train_encoded.columns.tolist()}

feature_config_path = (MODELS_DIR /"feature_config.pkl")
joblib.dump(feature_config, feature_config_path)
print("\n✓ Feature configuration saved.")
print(f"Encoded feature count: " f"{len(X_train_encoded.columns)}")
print(f"Saved to: {feature_config_path}")

#SAVE FINAL MODEL METADATA
model_metadata = {
    "project": "Mobile-Money Fraud Detection",
    "dataset": "PaySim",
    "model": best_model_name,
    "calibration_method": best_calibration_method,
    "decision_threshold": float(best_f1_threshold),
    "feature_count": len(X_train_encoded.columns),
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "training_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "test_fraud_cases": int(y_test.sum()),
    "test_fraud_rate": float(y_test.mean()),
    
    "final_metrics": {"accuracy": float(final_accuracy), "balanced_accuracy": float(final_balanced_accuracy),
                      "precision": float(final_precision), "recall": float(final_recall), "f1_score": float(final_f1),
                      "roc_auc": float(final_roc_auc), "pr_auc": float(final_pr_auc), "mcc": float(final_mcc), "brier_score": float(final_brier)}}

metadata_path = (MODELS_DIR /"model_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(model_metadata, f, indent=4)

print("\n✓ Model metadata saved successfully.")
print(metadata_path)

#CREATE DEPLOYMENT MANIFEST
deployment_manifest = {"model": "final_calibrated_fraud_model.pkl", "encoder": "onehot_encoder.pkl", "scaler": "standard_scaler.pkl",
                       "threshold": "fraud_decision_threshold.pkl", "feature_config": "feature_config.pkl", "metadata": "model_metadata.json"}

manifest_path = (MODELS_DIR /"deployment_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(deployment_manifest, f, indent=4)

print("\n✓ Deployment manifest created.")
print(manifest_path)

#ARTIFACT INTEGRITY TEST
print("\nARTIFACT INTEGRITY TEST")
loaded_model = joblib.load(MODELS_DIR / "final_calibrated_fraud_model.pkl")
loaded_encoder = joblib.load(MODELS_DIR / "onehot_encoder.pkl")
loaded_scaler = joblib.load(MODELS_DIR / "standard_scaler.pkl")
loaded_threshold = joblib.load(MODELS_DIR / "fraud_decision_threshold.pkl")
loaded_feature_config = joblib.load(MODELS_DIR / "feature_config.pkl")

print("✓ Model loaded")
print("✓ Encoder loaded")
print("✓ Scaler loaded")
print("✓ Threshold loaded")
print("✓ Feature configuration loaded")

print(f"\nLoaded threshold: "f"{loaded_threshold:.6f}")
print(f"Loaded feature count: " f"{len(loaded_feature_config['encoded_feature_names'])}")
print("\nArtifact integrity test completed successfully.")

#FINAL ARTIFACT INVENTORY
print("\nFINAL DEPLOYMENT ARTIFACTS")

deployment_files = ["final_calibrated_fraud_model.pkl", "onehot_encoder.pkl", "standard_scaler.pkl", "large_transaction_threshold.pkl",
                    "fraud_decision_threshold.pkl", "feature_config.pkl", "model_metadata.json", "deployment_manifest.json"]

for filename in deployment_files:
    path = MODELS_DIR / filename
    if path.exists():
        print(f"✓ {filename}")
    else:
        print(f"✗ MISSING: {filename}")

print("\nModel packaging completed successfully.")

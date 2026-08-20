3MTT DATA SCIENCE PROJECT
AUTHOR: Louis Mbagwu

Project Title: MOBILE-MONEY FRAUD DETECTION
Problem Context: Fraud hides in transactions.
Dataset: PaySim synthetic mobile-money transaction dataset
Primary Tools: Python, pandas, NumPy, scikit-learn, XGBoost, SciPy, joblib, Streamlit, VS Code


1. PROJECT OVERVIEW

This project develops an end-to-end machine-learning system for detecting potentially fraudulent mobile-money transactions.

The system goes beyond a simple fraud/normal prediction. It provides:

- Fraud probability
- Fraud/normal classification
- Risk level
- Five interpretable anomaly flags
- Anomaly count
- Human-readable anomaly description
- Calibrated probability estimates
- Optimized fraud decision threshold
- Production prediction pipeline
- Streamlit user interface

The project was designed around a Nigerian financial-technology problem:

"Fraud hides in transactions."


2. PROBLEM STATEMENT

Mobile-money and digital financial transactions generate large volumes of transaction data, making manual identification of fraudulent activity difficult.

The objective of this project is to develop a machine-learning system capable of identifying suspicious transactions and assigning a risk score to support fraud investigation and transaction monitoring.

Core MVP requirements:

- Data preparation
- Fraud detection model
- Anomaly flags
- Model evaluation


3. DATASET

The project uses the PaySim synthetic mobile-money transaction dataset. The dataset contains transaction-level information including:

- Transaction type
- Transaction amount
- Origin account balance
- Destination account balance
- Time step
- Fraud indicator

Target variable: isFraud

where:

    0 = Normal transaction
    1 = Fraudulent transaction

LIMITATION:
PaySim is a synthetic dataset. Therefore, the model's performance on PaySim should not be interpreted as equivalent to performance on live Nigerian financial transactions.


4. MACHINE LEARNING WORKFLOW

The project follows this workflow:

    Raw PaySim Dataset
            |
            v
    Data Inspection
            |
            v
    Data Cleaning
            |
            v
    Exploratory Data Analysis
            |
            v
    Feature Engineering
            |
            v
    Feature Selection
            |
            v
    Stratified Train/Test Split
            |
            v
    Training-Only Preprocessing
            |
            v
    Class-Imbalance Handling
            |
            v
    Baseline and Advanced Models
            |
            v
    Hyperparameter Optimization
            |
            v
    Model Selection
            |
            v
    Probability Calibration
            |
            v
    Threshold Optimization
            |
            v
    Final Unbiased Test
            |
            v
    Artifact Packaging
            |
            v
    Production Prediction Pipeline
            |
            v
    Streamlit Application


5. FEATURE ENGINEERING

Fraud-specific features were developed to capture transaction behavior.

Important engineered features include:

Balance differences:
- origin_balance_diff
- destination_balance_diff

Amount-to-balance ratios:
- origin_amount_ratio
- destination_amount_ratio

Zero-balance indicators:
- origin_zero_balance
- destination_zero_balance

Account-drain indicator:
- origin_drained

Time features:
- hour
- day
- day_of_week
- is_weekend

Transaction amount transformation:
- log_amount

Origin balance change:
- origin_balance_change_pct

Full balance transfer:
- full_balance_transfer

The production predictor reproduces the same feature-engineering logic used during training so that training and inference remain consistent.


6. DATA LEAKAGE PREVENTION

Leakage prevention was treated as an important part of the machine-learning workflow.

The final workflow:

- Separates predictors from the target.
- Uses a stratified train/test split.
- Fits preprocessing using training information.
- Keeps the final test set untouched.
- Removes non-predictive identifiers.
- Removes leakage-prone features where appropriate.
- Separates calibration and threshold-validation data.
- Uses the final test set only for unbiased evaluation.


7. CLASS IMBALANCE

Fraud represents a very small proportion of PaySim transactions. The final pipeline uses class weighting rather than relying on SMOTE.

Examples include:

    class_weight="balanced"

and, for XGBoost:

    scale_pos_weight

This approach was selected as a practical strategy for the very large PaySim dataset and also helped control RAM consumption during model development.

NOTE:
Earlier experimentation considered SMOTE, but the final production training workflow does not depend on SMOTE.


8. MODELS

The project evaluates several classification algorithms, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The final selected model is:

    Random Forest

The final calibrated model is stored as:

    models/final_calibrated_fraud_model.pkl


9. HYPERPARAMETER OPTIMIZATION

Hyperparameter tuning was performed using a computationally controlled strategy.

The optimization process used:

- 3-fold Stratified Cross-Validation
- 10 randomized search iterations
- n_jobs=1
- float32 numerical data where appropriate
- A 300,000-row stratified tuning subset

The tuning subset was selected from the training data.

This approach was introduced because the full PaySim dataset is very large and initially caused RAM constraints during model development.


10. MODEL CALIBRATION

After model selection, probability calibration was performed.

The project evaluated:

- Uncalibrated probabilities
- Sigmoid / Platt scaling
- Isotonic calibration

The final deployed calibration method is:

    Isotonic

Calibration was included because the system is intended to produce meaningful fraud probabilities rather than only binary predictions.


11. FRAUD DECISION THRESHOLD

The default classification threshold of 0.50 was not assumed to be optimal for fraud detection.

A threshold was selected using validation data with the objective of maximizing fraud-class F1 score.

The final production threshold is:

    0.09

It is stored in:

    models/fraud_decision_threshold.pkl


12. FINAL UNBIASED TEST RESULTS

The final evaluation was performed on the untouched test set.

Final test transactions: 1,272,524

Final fraud cases: 1,643

Final model: Random Forest

Calibration: Isotonic

Decision threshold: 0.09


Final performance:

    Accuracy              0.999993
    Balanced Accuracy     0.998781
    Precision              0.996959
    Recall                 0.997565
    F1 Score               0.997262
    ROC-AUC                 0.998783
    PR-AUC                  0.997569
    MCC                     0.997258
    Brier Score             0.00000325


Fraud detection results:

    True Positives          1,639
    False Positives             5
    False Negatives             4
    True Negatives      1,270,876

The model detected 1,639 of 1,643 fraud cases in the final test set.

INTERPRETATION:
These results are exceptionally strong, but they must be interpreted in the context of the synthetic PaySim dataset. Real-world deployment would require validation against real transaction data in Nigerian context and continuous monitoring for changes in fraud patterns.


13. PRODUCTION PREDICTION PIPELINE

The production prediction engine is: predictor.py

Its workflow is:

    Input Transaction
          |
          v
    Input Validation
          |
          v
    Feature Engineering
          |
          v
    Encoding
          |
          v
    Feature Alignment
          |
          v
    Calibrated Model
          |
          v
    Fraud Probability
          |
          v
    Decision Threshold
          |
          v
    Fraud Status
          |
          v
    Anomaly Analysis
          |
          v
    Risk Level


14. ANOMALY DETECTION

The production system calculates five anomaly flags:

1. large_transaction
2. origin_zero_balance
3. destination_zero_balance
4. origin_drained
5. full_balance_transfer

It also calculates:
    anomaly_count
and:
    anomaly_description

These features are intended to make model output more understandable to a fraud analyst.


15. RISK CLASSIFICATION

The production system translates fraud probability into four operational categories:

    Probability < 0.09       LOW
    0.09 - < 0.50            MEDIUM
    0.50 - < 0.90            HIGH
    >= 0.90                  CRITICAL

The risk classification is implemented in the production predictor.


16. STREAMLIT APPLICATION

The Streamlit application provides a user-friendly interface for transaction analysis.

Users can enter transaction information such as:

- Transaction type
- Transaction amount
- Origin balance before transaction
- Origin balance after transaction
- Destination balance before transaction
- Destination balance after transaction

The application returns:

- Fraud probability
- Fraud/Normal classification
- Risk level
- Number of anomaly flags
- Anomaly information
- Model information
- Decision threshold

Run the application with:

    python -m streamlit run app.py

The application should open in the browser.

Alternatively, you can use the provided Streamlit URL:

https://mobile-money-fraud-detector.streamlit.app/


17. PROJECT STRUCTURE

Recommended repository structure:

    MOBILE MONEY FRAUD DETECTOR/
    |
    +-- app.py
    +-- predictor.py
    +-- Mobile_Money_Fraud_Detection_no_SMOTE.py
    +-- README.txt
    +-- requirements.txt
    |
    +-- data/
    |   +-- raw/
    |   |   +-- PS_20174392719_1491204439457_log.csv
    |   |
    |   +-- processed/
    |
    +-- models/
    |   +-- final_calibrated_fraud_model.pkl
    |   +-- onehot_encoder.pkl
    |   +-- fraud_decision_threshold.pkl
    |   +-- large_transaction_threshold.pkl
    |   +-- feature_config.pkl
    |   +-- model_metadata.json
    |   +-- deployment_manifest.json
    |
    +-- outputs/
        +-- figures/
        +-- reports/
        +-- results/

IMPORTANT:
The large raw PaySim CSV won't be uploaded to GitHub. Rather, the dataset will be kept locally, or instructions will be provided for obtaining it.


18. INSTALLATION

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\activate

Install the required dependencies:

    pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy xgboost joblib streamlit


19. RUNNING THE PROJECT

Test the production predictor:

    python predictor.py

Start the Streamlit application:

    python -m streamlit run app.py


20. PRODUCTION ARTIFACTS

The models directory contains the deployment artifacts, including:

    final_calibrated_fraud_model.pkl
    onehot_encoder.pkl
    fraud_decision_threshold.pkl
    large_transaction_threshold.pkl
    feature_config.pkl
    model_metadata.json
    deployment_manifest.json

These artifacts allow the production prediction pipeline to reproduce the trained model's expected inference configuration.


21. LIMITATIONS

Synthetic data:
PaySim is simulated data and does not contain every signal available in real Nigerian financial systems.

Concept drift:
Fraud patterns change over time. A production system would require model monitoring and periodic retraining.

Real-world signals:
Future versions could incorporate:

- Device fingerprinting
- SIM/device changes
- IP/network information
- Geographic information
- Transaction velocity
- Account age
- Previous fraud history
- Beneficiary behavior
- Failed authentication attempts
- Customer behavioral profiles


22. NIGERIAN CONTEXT - FUTURE IMPROVEMENTS

For a real Nigerian mobile-money deployment, future versions could include:

1. Real-time transaction scoring
2. Transaction velocity monitoring
3. Device/SIM intelligence
4. Geographic anomaly detection
5. Customer behavioral profiling
6. Graph-based fraud detection
7. Fraud investigator dashboard
8. Alert prioritization
9. Human-in-the-loop feedback
10. Model drift monitoring
11. Cost-sensitive threshold optimization
12. Explainable AI
13. Audit logging
14. Automated retraining


23. ETHICAL CONSIDERATIONS

The model should be treated as a fraud-risk decision-support system, not as the sole authority for irreversible financial decisions.

A real deployment should include:

- Human review
- False-positive monitoring
- Customer protection mechanisms
- Secure transaction-data handling
- Access controls
- Audit trails
- Model monitoring
- Periodic validation


24. CONCLUSION

This project demonstrates a complete machine-learning pipeline for mobile-money fraud detection, from raw transaction data through production deployment.

The final system combines:

    Data Preparation
          ->
    Feature Engineering
          ->
    Model Training
          ->
    Optimization
          ->
    Calibration
          ->
    Threshold Selection
          ->
    Unbiased Evaluation
          ->
    Anomaly Detection
          ->
    Risk Classification
          ->
    Production Prediction
          ->
    Streamlit Application

This meets the core MVP requirements while laying the foundation for a more advanced Nigerian financial fraud monitoring system.


import os

#Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "insurance_claims.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COL = "fraud_reported"


# Preprocessing artifact paths
ENCODER_PATH = os.path.join(os.path.dirname(MODEL_PATH), "encoder.joblib")
SCALER_PATH = os.path.join(os.path.dirname(MODEL_PATH), "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(MODEL_PATH), "label_encoder.joblib")

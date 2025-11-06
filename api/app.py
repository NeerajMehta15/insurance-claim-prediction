import os
import sys

# Add project root to Python path BEFORE local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from src.data_preprocessing import preprocess_for_inference




# Add both project root and src folder to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for path in [PROJECT_ROOT, SRC_PATH]:
    if path not in sys.path:
        sys.path.append(path)


# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# Paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "xgb_model.pkl")

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Claim Prediction API",
    description="Predicts insurance claim fraud likelihood based on user data.",
    version="1.0.0"
)

# Load model at startup
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

from pydantic import BaseModel, Field

class ClaimData(BaseModel):
    months_as_customer: float
    age: float
    policy_number: float
    policy_bind_date: str
    policy_state: str
    policy_csl: str
    policy_deductable: float
    policy_annual_premium: float
    umbrella_limit: float
    insured_zip: float
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    insured_hobbies: str
    insured_relationship: str

    # ✅ Allow both formats: capital_gains and capital-gains
    capital_gains: float = Field(..., alias="capital-gains")
    capital_loss: float = Field(..., alias="capital-loss")

    incident_date: str
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    incident_state: str
    incident_city: str
    incident_location: str
    incident_hour_of_the_day: float
    number_of_vehicles_involved: float
    property_damage: str
    bodily_injuries: float
    witnesses: float
    police_report_available: str
    total_claim_amount: float
    injury_claim: float
    property_claim: float
    vehicle_claim: float
    auto_make: str
    auto_model: str
    auto_year: float

    class Config:
        # ✅ This allows using either alias (capital-gains) or field name (capital_gains)
        allow_population_by_field_name = True
        populate_by_name = True




@app.get("/")
def home():
    """Root endpoint to verify the API is running."""
    return {"message": "Insurance Claim Prediction API is running "}


@app.post("/predict")
def predict(data: ClaimData):
    """Predict the likelihood of a fraudulent insurance claim."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Preprocess incoming data using saved encoder/scaler
        processed_df = preprocess_for_inference(input_df)

        # Make prediction
        prediction = model.predict(processed_df)
        prediction_label = int(prediction[0])

        return {
            "prediction": prediction_label,
            "result": "Fraudulent Claim" if prediction_label == 1 else "Legitimate Claim"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

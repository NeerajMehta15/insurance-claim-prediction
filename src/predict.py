import joblib
import pandas as pd
from config import MODEL_PATH

def predict_claim(input_data: dict):
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return int(prediction[0])

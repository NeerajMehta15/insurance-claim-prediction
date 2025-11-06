import os
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_data, preprocess_data, split_data
from config import TARGET_COL, MODEL_PATH

def train_model():
    data = load_data()
    X, y = preprocess_data(data, TARGET_COL)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()

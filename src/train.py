import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_data, preprocess_data, split_data
from config import MODEL_PATH

def train_model():
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()

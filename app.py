import joblib 

model = joblib.load("models/xgboost_fraud_model.pkl")

threshold = joblib.load("models/fraud_threshold.pkl")

feature_columns = joblib.load("models/feature_columns.pkl")

from fastapi import FastAPI

from schemas.transaction import Transaction
from services.feature_engineering import create_features

app = FastAPI()

@app.post("/predict")
def predict(transaction: Transaction):


    transaction_dict = transaction.model_dump()

    features = create_features(transaction_dict)
    features = features.reindex(
        columns=feature_columns,
        fill_value=0
    )
    
    probability = model.predict_proba(features)[0][1]
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": float(probability),
        "prediction": prediction
    }


 

@app.get("/")
def home():
    return {
        "message": "Fraud Detection Project"
    }



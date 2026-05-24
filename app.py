from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("fraud_detection_pipeline.pkl")

@app.route("/")
def home():
    return "Fraud Detection Backend Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_data = pd.DataFrame([{
        "type": data["type"],
        "amount": data["amount"],
        "oldbalanceOrg": data["oldbalanceOrg"],
        "newbalanceOrig": data["newbalanceOrig"],
        "oldbalanceDest": data["oldbalanceDest"],
        "newbalanceDest": data["newbalanceDest"]
    }])

    prediction = int(model.predict(input_data)[0])

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run()
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})
# Load trained model
model = joblib.load("fraud_detection_pipeline.pkl")

@app.route("/")
def home():
    return "Backend Running Successfully"

@app.route("/predict", methods=["POST"])
def predict():

    try:

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

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
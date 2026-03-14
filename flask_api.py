from flask import Flask, request, jsonify
import joblib
import numpy as np

# load trained model and feature schema
model = joblib.load("farm_yield_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Farm Yield Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Build row in the same order as feature_columns
    row = []
    missing = []
    for col in feature_columns:
        if col not in data:
            missing.append(col)
            row.append(0.0)
        else:
            row.append(float(data[col]))

    if missing:
        return jsonify({
            "error": "Missing features",
            "missing": missing
        }), 400

    input_data = np.array([row])
    prediction = model.predict(input_data)

    return jsonify({
        "predicted_yield": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)









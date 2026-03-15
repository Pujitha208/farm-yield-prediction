from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return "Farm Yield API Running"

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    rainfall = data['rainfall']
    temperature = data['temperature']
    fertilizer = data['fertilizer']

    features = np.array([[rainfall, temperature, fertilizer]])

    prediction = model.predict(features)

    return jsonify({
        "prediction": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)

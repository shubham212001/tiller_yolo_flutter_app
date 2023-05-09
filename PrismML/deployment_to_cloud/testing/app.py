from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the saved model and vectorizer
classifier = load('model.joblib')
vectorizer = load('vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['text']
    new_data = [input_text]
    new_features = vectorizer.transform(new_data)
    predictions = classifier.predict_proba(new_features)
    col=['AIR CONDITIONER','REFRIDGERATOR','Television','Speaker','Bulb','Washing Machine']

    true_columns = []
    for i in range(predictions.shape[1]):
        if predictions[0][i] >= 0.1:
            true_columns.append(col[i])
    
    return jsonify({'result': true_columns})

if __name__ == '__main__':
    app.run()

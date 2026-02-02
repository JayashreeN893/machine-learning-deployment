import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("logistic_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load feature order and medians
X_columns = pickle.load(open("X_columns.pkl", "rb"))
medians = pickle.load(open("medians.pkl", "rb"))

columns_with_zeros_as_nan = list(medians.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()

    # Convert incoming data to DataFrame
    try:
        input_df = pd.DataFrame([data])
    except ValueError as e:
        return jsonify({'error': f'Invalid input data format: {e}'}), 400

    # Ensure correct features
    if not all(col in input_df.columns for col in X_columns):
        missing_cols = [col for col in X_columns if col not in input_df.columns]
        return jsonify({'error': f'Missing expected features: {missing_cols}'}), 400

    # Reorder columns
    input_df = input_df[X_columns]

    # Handle zero values using training medians
    for col in columns_with_zeros_as_nan:
        input_df[col] = input_df[col].replace(0, medians[col])

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run()

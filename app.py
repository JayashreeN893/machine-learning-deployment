import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Define filenames for the model and scaler
model_filename = 'logistic_regression_model.pkl'
sch_filename = 'scaler.pkl'

# Load the trained logistic regression model
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the scaler object
with open(sch_filename, 'rb') as file:
    scaler = pickle.load(file)

# Recreate X.columns for feature ordering and validation
# Assuming df (the original DataFrame after preprocessing) is available in the environment
# If df is not available, these columns would need to be manually defined or loaded.
# For this step, we will assume df is present as per the notebook's execution flow up to this point.
# As a fallback, if df is truly unavailable at this point, one would need to hardcode the column names:
# X_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# For now, we will dynamically derive it as it was in previous steps.

# To get the column names correctly, we need the original df. If this block is run
# independently or in a fresh kernel, df might not exist. Assuming previous cells
# have defined df and X as in the training phase.
# However, for a deployment-ready script, it's safer to have these hardcoded or explicitly passed.
# Given the subtask is to combine, we'll re-derive it based on the previous df content.
# If this fails because df is not in scope, a manual list would be needed.
# Reconstructing the 'columns_with_zeros_as_nan' as it was crucial for df preprocessing logic
columns_with_zeros_as_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Load the dataset again to get the original columns, then apply preprocessing steps to mimic the training data structure
# This is done to ensure X.columns is available in the final combined script for robustness.
# NOTE: In a real deployment, you might save X.columns as part of your model artifacts.
# For this exercise, we simulate it by reloading the data and performing initial preprocessing.
initial_df = pd.read_csv('/content/diabetes 2.csv')
initial_df[columns_with_zeros_as_nan] = initial_df[columns_with_zeros_as_nan].replace(0, np.nan)

# Impute missing values with the median of their respective columns, similar to training
for col in columns_with_zeros_as_nan:
    median_val = initial_df[col].median()
    initial_df[col] = initial_df[col].fillna(median_val)

X_columns = initial_df.drop('Outcome', axis=1).columns

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()

        # Convert incoming data to DataFrame
        try:
            input_df = pd.DataFrame([data])
        except ValueError as e:
            return jsonify({'error': f'Invalid input data format: {e}'}), 400

        # Ensure the order of columns matches the training data
        if not all(col in input_df.columns for col in X_columns):
            missing_cols = [col for col in X_columns if col not in input_df.columns]
            return jsonify({'error': f'Missing expected features: {missing_cols}'}), 400

        # Reorder columns to match the training data
        input_df = input_df[X_columns]

        # Handle 0 values in specific columns for new predictions, similar to training preprocessing
        for col in columns_with_zeros_as_nan:
            if 0 in input_df[col].values:
                # This is a simplification; in a real app, you would use the *training median*
                # For now, we will impute with the median of the current input if it's 0,
                # but the scaler will still handle the scale.
                # A more robust solution would be to pass the medians used during training.
                # For this subtask, we rely on the scaler to handle the data as it's been preprocessed.
                # We explicitly handle 0s to NaN first, then the scaler will process the numerical values.
                input_df[col] = input_df[col].replace(0, initial_df[col].median()) # Use training median for consistency


        # Preprocess the input data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Make prediction using the loaded model
        prediction = model.predict(input_scaled)

        # Return the prediction as JSON response
        return jsonify({'prediction': int(prediction[0])})
    else:
        return jsonify({'error': 'Request must be JSON'}), 400

print("Flask application combined, model and scaler loaded, and '/predict' endpoint defined.")

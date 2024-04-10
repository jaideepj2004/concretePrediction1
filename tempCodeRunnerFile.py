from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get the full path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

print("Model path:", model_path)  # Print the model path for debugging

# Load the model
try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("FileNotFoundError: Unable to locate the model.pkl file.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input variables from the form
    try:
        inputs = [float(request.form[field]) for field in ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']]
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([inputs], columns=['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'])
        print (f"inout data{input_data}")
        # Scale the input data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        # Make predictions
        prediction = model.predict(input_data_scaled)[0]
        logging.info("Predicted Compressive Strength: %f MPa", prediction)
        print(prediction)
        return redirect(url_for('result', prediction=prediction))
    except Exception as e:
        logging.error(f"Error predicting: {str(e)}")
        return f"Error predicting: {str(e)}"

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

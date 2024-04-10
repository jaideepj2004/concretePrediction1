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

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'C:/Users/jaide/OneDrive/Desktop/Data Science/Internship/pwskills concrete strength prediction/project_folder', 'model.pkl')
try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("FileNotFoundError: Unable to locate the model.pkl file.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")

# Load the scaler
scaler_path = os.path.join(os.path.dirname(__file__), 'C:/Users/jaide/OneDrive/Desktop/Data Science/Internship/pwskills concrete strength prediction/project_folder', 'scaler.pkl')
try:
    scaler = joblib.load(scaler_path)
    logging.info("Scaler loaded successfully.")
except FileNotFoundError:
    logging.error("FileNotFoundError: Unable to locate the scaler.pkl file.")
except Exception as e:
    logging.error(f"Error loading scaler: {str(e)}")

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input variables from the form
    try:
        inputs = [float(value) for value in request.form.values()]
        
        # Scale the input data
        input_data_scaled = scaler.transform([inputs])
        
        # Make predictions
        prediction = model.predict(input_data_scaled)[0]
        logging.info("Predicted Compressive Strength: %f MPa", prediction)
        
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


# Latency (Response Time): 0.011525869369506836 seconds
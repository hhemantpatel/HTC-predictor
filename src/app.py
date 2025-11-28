from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model and artifacts
MODEL_DIR = "models"
try:
    model = joblib.load(os.path.join(MODEL_DIR, "htc_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Ensure columns match training features
        # We expect the frontend to send keys matching feature_names
        # But frontend keys might be simple (e.g. 'velocity'), so we map them if needed.
        # For simplicity, let's expect the frontend to send the exact feature names or we map them here.
        
        # Mapping simple keys to complex CSV names
        mapping = {
            'vol_frac': 'Volume_Fraction (%)',
            'temp': 'Temperature (°C)',
            'velocity': 'Flow_Velocity (m/s)',
            'k': 'Thermal_Conductivity (W/mK)',
            'cp': 'Specific_Heat_Capacity (J/kgK)',
            'rho': 'Density (kg/m³)',
            'mu': 'Viscosity (Pa·s)'
        }
        
        # Rename keys
        renamed_data = {mapping.get(k, k): v for k, v in data.items()}
        input_df = pd.DataFrame([renamed_data])
        
        # Reorder columns
        input_df = input_df[feature_names]
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({'prediction': float(prediction)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

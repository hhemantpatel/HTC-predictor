import pandas as pd
import numpy as np
import joblib
import os
import sys

def predict_h(volume_fraction, temperature, velocity, conductivity, cp, density, viscosity):
    """
    Predicts Heat Transfer Coefficient (h) using the trained model.
    """
    model_path = os.path.join("models", "htc_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    feature_names_path = os.path.join("models", "feature_names.pkl")
    
    if not os.path.exists(model_path):
        print("Error: Model not found. Run train.py first.")
        return None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    
    # Prepare input dataframe
    # Must match the order and names used in training
    input_data = pd.DataFrame({
        'Volume_Fraction (%)': [volume_fraction],
        'Temperature (°C)': [temperature],
        'Flow_Velocity (m/s)': [velocity],
        'Thermal_Conductivity (W/mK)': [conductivity],
        'Specific_Heat_Capacity (J/kgK)': [cp],
        'Density (kg/m³)': [density],
        'Viscosity (Pa·s)': [viscosity]
    })
    
    # Ensure columns are in correct order
    input_data = input_data[feature_names]
    
    # Scale
    input_scaled = scaler.transform(input_data)
    
    # Predict
    h_pred = model.predict(input_scaled)[0]
    
    return h_pred

if __name__ == "__main__":
    print("--- Heat Transfer Coefficient Predictor ---")
    print("Enter the following parameters:")
    
    try:
        vol_frac = float(input("Volume Fraction (%): "))
        temp = float(input("Temperature (°C): "))
        velocity = float(input("Flow Velocity (m/s): "))
        k = float(input("Thermal Conductivity (W/mK): "))
        cp = float(input("Specific Heat Capacity (J/kgK): "))
        rho = float(input("Density (kg/m³): "))
        mu = float(input("Viscosity (Pa·s): "))
        
        print(f"\nPredicting h for:")
        print(f"Volume Fraction: {vol_frac}%")
        print(f"Temperature: {temp} C")
        print(f"Velocity: {velocity} m/s")
        
        h_pred = predict_h(vol_frac, temp, velocity, k, cp, rho, mu)
        
        if h_pred is not None:
            print(f"\nPredicted Heat Transfer Coefficient (h): {h_pred:.2f} W/(m^2.K)")
            
    except ValueError:
        print("Invalid input. Please enter numeric values.")

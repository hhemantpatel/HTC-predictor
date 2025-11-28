import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def train_model():
    # Load data
    # Check for real data first, then fall back to synthetic
    real_data_path = os.path.join("data", "nanofluid_dataset.csv")
    synthetic_data_path = os.path.join("data", "synthetic_data.csv")
    
    if os.path.exists(real_data_path):
        print(f"Loading real data from {real_data_path}...")
        df = pd.read_csv(real_data_path)
        is_real_data = True
    elif os.path.exists(synthetic_data_path):
        print(f"Loading synthetic data from {synthetic_data_path}...")
        df = pd.read_csv(synthetic_data_path)
        is_real_data = False
    else:
        print("Error: No data found. Run data_generator.py or download real data.")
        return

    # Feature Mapping
    if is_real_data:
        # Clean column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        
        # Target
        target_col = 'Heat_Transfer_Coefficient (W/m²K)'
        
        # Features
        # We will use the physical properties provided.
        # Note: Nanoparticle_Type and Base_Fluid are categorical. 
        # For a simple MLP, we can either drop them or one-hot encode.
        # Given the "tiny" model requirement, let's try using just the numerical properties first.
        # If performance is poor, we can add one-hot encoding.
        
        feature_cols = [
            'Volume_Fraction (%)',
            'Temperature (°C)',
            'Flow_Velocity (m/s)',
            'Thermal_Conductivity (W/mK)',
            'Specific_Heat_Capacity (J/kgK)',
            'Density (kg/m³)',
            'Viscosity (Pa·s)'
        ]
        
        # Calculate Re and Pr if possible to help the model?
        # The dataset doesn't have Diameter, so we can't calculate Re exactly without assuming D.
        # However, we have Velocity, Density, Viscosity.
        # Let's stick to the raw inputs provided in the dataset for now.
        
        print(f"Target column: {target_col}")
        print(f"Feature columns: {feature_cols}")
        
    else:
        # Synthetic data mapping
        feature_cols = ['Velocity_m_s', 'Diameter_m', 'Density_kg_m3', 'Viscosity_Pa_s', 
                        'Conductivity_W_mK', 'Specific_Heat_J_kgK', 'Reynolds', 'Prandtl']
        target_col = 'HTC_W_m2K'
    
    # Handle missing values
    df = df.dropna()
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP
    print("Training MLP Regressor...")
    # Increased complexity for real data which might be noisier
    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=2000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "htc_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    
    # Save feature names for inference
    joblib.dump(feature_cols, os.path.join("models", "feature_names.pkl"))
    
    print("Model, scaler, and feature names saved to models/")

if __name__ == "__main__":
    train_model()

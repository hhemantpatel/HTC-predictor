import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import r2_score

def generate_plots():
    # Load data
    data_path = os.path.join("data", "nanofluid_dataset.csv")
    if not os.path.exists(data_path):
        print("Data not found.")
        return

    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Load model artifacts
    model_path = os.path.join("models", "htc_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    feature_names_path = os.path.join("models", "feature_names.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    
    # Prepare X and y
    X = df[feature_names]
    y = df['Heat_Transfer_Coefficient (W/m²K)']
    
    # Predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    r2 = r2_score(y, y_pred)
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5, color='#2563eb')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual HTC (W/m²K)')
    plt.ylabel('Predicted HTC (W/m²K)')
    plt.title(f'Actual vs Predicted Heat Transfer Coefficient (R² = {r2:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/actual_vs_predicted.png', dpi=300)
    print("Saved images/actual_vs_predicted.png")
    
    # Plot 2: Residuals
    residuals = y - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='#059669', edgecolor='black', alpha=0.7)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Residuals)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/residuals.png', dpi=300)
    print("Saved images/residuals.png")

if __name__ == "__main__":
    generate_plots()

import numpy as np
import pandas as pd
import os

def generate_synthetic_data(n_samples=1000):
    """
    Generates synthetic data for forced convection in a smooth tube
    using the Dittus-Boelter correlation.
    
    Nu = 0.023 * Re^0.8 * Pr^0.4 (for heating)
    
    Inputs generated:
    - Velocity (U): m/s
    - Diameter (D): m
    - Temperature (T): K (used to vary properties slightly, though simplified here)
    - Density (rho): kg/m^3
    - Viscosity (mu): Pa.s
    - Thermal Conductivity (k): W/(m.K)
    - Specific Heat (Cp): J/(kg.K)
    
    Derived:
    - Reynolds (Re) = rho * U * D / mu
    - Prandtl (Pr) = Cp * mu / k
    
    Target:
    - Heat Transfer Coeff (h) = Nu * k / D
    """
    
    np.random.seed(42)
    
    # Random inputs within realistic ranges for water-like fluid in a tube
    U = np.random.uniform(0.5, 5.0, n_samples)  # Velocity 0.5 to 5 m/s
    D = np.random.uniform(0.01, 0.1, n_samples) # Diameter 1cm to 10cm
    
    # Fluid properties (Simplified: varying around water properties at 20-50C)
    # In a real scenario, these would be functions of T. Here we perturb them to create variance.
    rho = np.random.normal(997, 5, n_samples)   # ~997 kg/m3
    mu = np.random.normal(0.00089, 0.0001, n_samples) # ~0.00089 Pa.s
    k = np.random.normal(0.6, 0.02, n_samples)    # ~0.6 W/mK
    Cp = np.random.normal(4180, 20, n_samples)    # ~4180 J/kgK
    
    # Calculate dimensionless numbers
    Re = (rho * U * D) / mu
    Pr = (Cp * mu) / k
    
    # Filter for turbulent flow (Re > 10000 is best for Dittus-Boelter, but we'll allow > 2300)
    # We will just generate and then maybe flag them, or just assume the correlation holds "enough" for ML demo
    # Let's enforce Re > 2300 for validity of turbulent correlation, roughly.
    
    # Dittus-Boelter Correlation (Heating n=0.4)
    Nu = 0.023 * (Re**0.8) * (Pr**0.4)
    
    # Calculate h
    h = (Nu * k) / D
    
    data = pd.DataFrame({
        'Velocity_m_s': U,
        'Diameter_m': D,
        'Density_kg_m3': rho,
        'Viscosity_Pa_s': mu,
        'Conductivity_W_mK': k,
        'Specific_Heat_J_kgK': Cp,
        'Reynolds': Re,
        'Prandtl': Pr,
        'Nusselt': Nu,
        'HTC_W_m2K': h
    })
    
    return data

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data(2000)
    
    output_path = os.path.join("data", "synthetic_data.csv")
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(df.head())

import requests
import pandas as pd
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

# --- 1. THE DATA FETCHER (API) ---
def get_real_indian_rainfall(lat, lon, start_year, end_year):
    """
    Fetches daily rainfall from Open-Meteo Archive API.
    Coordinates: 
    - Mumbai: 19.0760, 72.8777
    - Delhi:  28.7041, 77.1025
    - Bangalore: 12.9716, 77.5946
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata"
    }
    
    print(f"📡 Fetching real data for coordinates ({lat}, {lon})...")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Clean and Format
        df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'rain': data['daily']['precipitation_sum']
        })
        
        # Handle missing values (assume 0 rain if data missing)
        df['rain'] = df['rain'].fillna(0)
        df.set_index('date', inplace=True)
        
        print(f"✅ Downloaded {len(df)} days of real weather history.")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# --- 2. YOUR STERN-COE MODEL (Re-used) ---
class SternCoeModel:
    def __init__(self):
        self.p01 = 0 
        self.p11 = 0 
        self.gamma_alpha = 0
        self.gamma_scale = 0
        
    def fit(self, df):
        # 1. Markov Chain (Occurrence)
        wet = (df['rain'] > 1.0).astype(int) # Threshold 1mm to ignore drizzle
        
        n01 = len(df[(wet.shift(1) == 0) & (wet == 1)])
        n00 = len(df[(wet.shift(1) == 0) & (wet == 0)])
        n11 = len(df[(wet.shift(1) == 1) & (wet == 1)])
        n10 = len(df[(wet.shift(1) == 1) & (wet == 0)])
        
        self.p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        self.p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        
        # 2. Gamma Distribution (Intensity)
        wet_days = df[df['rain'] > 1.0]['rain']
        if len(wet_days) > 0:
            params = gamma.fit(wet_days, floc=0)
            self.gamma_alpha = params[0]
            self.gamma_scale = params[2]
        
        print(f"\n--- MODEL CALIBRATED ON REAL DATA ---")
        print(f"Prob(Rain | Dry Yesterday): {self.p01:.3f} (Wait for storm)")
        print(f"Prob(Rain | Wet Yesterday): {self.p11:.3f} (Storm Persistence)")
        print(f"Avg Rain Intensity (Gamma): {self.gamma_alpha * self.gamma_scale:.2f} mm")

# --- 3. RUN IT ---
if __name__ == "__main__":
    # Mumbai Coordinates (Santacruz)
    MUMBAI_LAT = 19.0760
    MUMBAI_LON = 72.8777

    # Fetch last 20 years of REAL data
    df_real = get_real_indian_rainfall(MUMBAI_LAT, MUMBAI_LON, 2003, 2023)

    if not df_real.empty:
        # Train the Model
        model = SternCoeModel()
        model.fit(df_real)
        
        # Simple plot to visualize the fetched data
        plt.figure(figsize=(10, 5))
        plt.plot(df_real.index, df_real['rain'], label='Rainfall (mm)', color='blue', alpha=0.7)
        plt.title(f"Real Rainfall Data: Mumbai ({2003}-{2023})")
        plt.ylabel("Precipitation (mm)")
        plt.xlabel("Year")
        plt.legend()
        plt.show()

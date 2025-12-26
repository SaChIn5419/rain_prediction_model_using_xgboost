import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

import os

# --- 1. DATA PIPELINE (Real Indian Rainfall) ---
def get_real_indian_rainfall(lat, lon, start_year, end_year, cache_file="mumbai_rain_data.csv"):
    """
    Fetches daily rainfall from Open-Meteo Archive API.
    Checks for local cache first.
    """
    # 1. Check Cache
    if os.path.exists(cache_file):
        print(f"📂 Loading data from cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
        df.index = pd.to_datetime(df.index) # Ensure index is DatetimeIndex
        return df

    # 2. Fetch from API
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
        
        # 3. Save to Cache
        df.to_csv(cache_file)
        print(f"✅ Downloaded {len(df)} days of real weather history and saved to {cache_file}.")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# --- 2. BENCHMARK MODEL (Stern & Coe) ---
class SternCoeModel:
    def __init__(self):
        self.p01 = 0 # Prob Wet | Dry
        self.p11 = 0 # Prob Wet | Wet
        self.gamma_alpha = 0
        self.gamma_scale = 0
        
    def fit(self, df):
        # 1. Occurrence Model (Markov Chain)
        # Create binary series (Threshold 0.1mm)
        wet = (df['rain'] > 0.1).astype(int)
        
        # Count transitions
        n01 = len(df[(wet.shift(1) == 0) & (wet == 1)])
        n00 = len(df[(wet.shift(1) == 0) & (wet == 0)])
        n11 = len(df[(wet.shift(1) == 1) & (wet == 1)])
        n10 = len(df[(wet.shift(1) == 1) & (wet == 0)])
        
        self.p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        self.p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        
        # 2. Intensity Model (Gamma Distribution)
        wet_days = df[df['rain'] > 0.1]['rain']
        
        if len(wet_days) > 10:
            params = gamma.fit(wet_days, floc=0) 
            self.gamma_alpha = params[0]
            self.gamma_scale = params[2]
        
        print(f"[Stern-Coe] P(Wet|Dry): {self.p01:.3f}, P(Wet|Wet): {self.p11:.3f}")
        print(f"[Stern-Coe] Gamma Alpha: {self.gamma_alpha:.3f}, Scale: {self.gamma_scale:.3f}")

    def predict(self, n_days, start_state=0):
        preds = []
        current_state = start_state
        
        for _ in range(n_days):
            # Markov Step
            threshold = self.p11 if current_state == 1 else self.p01
            is_wet = np.random.rand() < threshold
            current_state = 1 if is_wet else 0
            
            # Gamma Step
            amt = 0
            if is_wet:
                amt = gamma.rvs(self.gamma_alpha, scale=self.gamma_scale)
            preds.append(amt)
            
        return np.array(preds)

# --- 3. CHALLENGER MODEL (XGBoost) ---
class XGBoostWeatherModel:
    def __init__(self):
        self.clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss')
        self.reg = xgb.XGBRegressor(n_estimators=100, max_depth=3)
        
    def preprocess(self, df):
        data = df.copy()
        # --- Feature Engineering ---
        # 1. Lag Features (Memory)
        data['lag_1'] = data['rain'].shift(1)
        data['lag_2'] = data['rain'].shift(2)
        
        # 2. Rolling Features (Trend)
        data['roll_mean_7'] = data['rain'].rolling(7).mean()
        
        # 3. Seasonality
        day_of_year = data.index.dayofyear
        data['sin_day'] = np.sin(2 * np.pi * day_of_year / 365)
        data['cos_day'] = np.cos(2 * np.pi * day_of_year / 365)

        # 4. Dry Spell Counter
        is_dry = (data['rain'] == 0).astype(int)
        groups = (data['rain'] > 0).astype(int).cumsum()
        data['dry_spell_count'] = is_dry.groupby(groups).cumsum()
        
        # Target
        data['is_wet'] = (data['rain'] > 0).astype(int)
        
        return data.dropna()

    def fit(self, df):
        processed = self.preprocess(df)
        
        features = ['lag_1', 'lag_2', 'roll_mean_7', 'sin_day', 'cos_day', 'dry_spell_count']
        X = processed[features]
        y_class = processed['is_wet']
        y_reg = processed['rain']
        
        self.clf.fit(X, y_class)
        
        wet_indices = processed['is_wet'] == 1
        if wet_indices.sum() > 0:
            self.reg.fit(X[wet_indices], y_reg[wet_indices])
        
        print("[XGBoost] Models Trained.")
        
    def predict(self, df):
        processed = self.preprocess(df)
        features = ['lag_1', 'lag_2', 'roll_mean_7', 'sin_day', 'cos_day', 'dry_spell_count']
        X = processed[features]
        
        prob_wet = self.clf.predict_proba(X)[:, 1]
        pred_amount = self.reg.predict(X)
        
        final_preds = np.where(prob_wet > 0.5, pred_amount, 0)
        return final_preds, processed['rain'].values

def run_backtest(predictions, actuals, threshold_buy=10, threshold_sell=2):
    """
    Simulates a P&L (Profit & Loss) curve for a trading strategy.
    """
    capital = 100000 # Starting Capital (₹)
    pnl_curve = [capital]
    tick_size = 1000 # ₹1000 per mm movement
    
    positions = [] # Track trade direction
    
    for pred, actual in zip(predictions, actuals):
        # 1. Generate Signal
        position = 0
        if pred > threshold_buy:
            position = 1  # Long (Betting on Heavy Rain)
        elif pred < threshold_sell:
            position = -1 # Short (Betting on Dry Day)
            
        # 2. Calculate Daily P&L
        # Profit = Position * (Actual Rain - Market Expectation)
        # We assume Market Expectation is the "Lag 1" (Naïve Forecast)
        # (In reality, market expectation is the consensus forecast)
        market_expectation = pred # Simplifying: Assuming we trade against a "dumb" market
        
        # If we went Long, we want Actual > Market
        daily_pnl = position * (actual - market_expectation) * tick_size
        
        # Transaction Cost (Bid-Ask Spread)
        cost = 500 if position != 0 else 0 
        
        capital += (daily_pnl - cost)
        pnl_curve.append(capital)
        positions.append(position)
        
    return pnl_curve, positions

def grid_search_optimization(predictions, actuals):
    # Define the "Parameter Space"
    buy_range = range(10, 50, 2)   # Test Buying at 10mm up to 50mm (heavier storms)
    sell_range = range(0, 5, 1)    # Test Selling at 0mm to 5mm (dry days)
    
    results = []
    
    print("⚙️ Running Grid Search on Strategy Parameters...")
    
    for b in buy_range:
        for s in sell_range:
            if b > s: 
                # Run Backtest
                equity_curve, _ = run_backtest(predictions, actuals, 
                                            threshold_buy=b, 
                                            threshold_sell=s)
                
                final_pnl = equity_curve[-1] - 100000 # Profit
                
                results.append({
                    'Buy_Threshold': b,
                    'Sell_Threshold': s,
                    'Net_Profit': final_pnl
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find the Best Combination
    if not results_df.empty:
        best_run = results_df.loc[results_df['Net_Profit'].idxmax()]
        
        print("-" * 30)
        print("OPTIMIZATION RESULTS")
        print("-" * 30)
        print(f"Best Buy Threshold:   {best_run['Buy_Threshold']} mm")
        print(f"Best Sell Threshold:  {best_run['Sell_Threshold']} mm")
        print(f"Max Profit Found:     ₹{best_run['Net_Profit']:,.0f}")
    else:
        print("No valid combinations found for grid search.")
    
    return results_df

# --- 4. EXECUTION PIPELINE ---
if __name__ == "__main__":
    # --- Configuration ---
    LOCATION = "Mumbai"
    LAT = 19.0760
    LON = 72.8777
    START_YEAR = 2000
    END_YEAR = 2023
    
    # 1. Fetch Real Data
    print(f"\n--- 1. Fetching Real Data for {LOCATION} ---")
    df = get_real_indian_rainfall(LAT, LON, START_YEAR, END_YEAR)
    
    if not df.empty:
        # 2. Split Data (Train on first 80%, Test on last 20%)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"Training Data: {train_df.index.min().date()} to {train_df.index.max().date()}")
        print(f"Testing Data:  {test_df.index.min().date()} to {test_df.index.max().date()}")

        # 3. Train Models
        print("\n--- 2. Training Benchmarks & Challengers ---")
        sc_model = SternCoeModel()
        sc_model.fit(train_df)

        xgb_model = XGBoostWeatherModel()
        xgb_model.fit(train_df)

        # 4. Evaluate
        print("\n--- 3. Evaluation ---")
        # Stern Coe (Simulation)
        sc_preds = sc_model.predict(len(test_df), start_state=0)
        
        # XGBoost (Prediction)
        xgb_preds, actuals = xgb_model.predict(test_df)

        # Align lengths
        min_len = min(len(sc_preds), len(xgb_preds))
        sc_preds = sc_preds[:min_len]
        xgb_preds = xgb_preds[:min_len]
        actuals = actuals[:min_len]

        # Metrics
        sc_error = np.sqrt(mean_squared_error(actuals, sc_preds))
        xgb_error = np.sqrt(mean_squared_error(actuals, xgb_preds))

        print("-" * 30)
        print(f"RMSE (Stern-Coe): {sc_error:.2f} (Baseline)")
        print(f"RMSE (XGBoost):   {xgb_error:.2f} (Challenger)")
        print("-" * 30)

        # 5. Visualize
        plt.figure(figsize=(12, 6))
        # Plot last 365 days of test set for clarity
        days_to_plot = 200 
        plt.plot(actuals[:days_to_plot], label='Actual Rain', color='black', alpha=0.6)
        plt.plot(xgb_preds[:days_to_plot], label='XGBoost', color='blue', alpha=0.8)
        plt.plot(sc_preds[:days_to_plot], label='Stern-Coe (Sim)', color='red', alpha=0.4, linestyle='--')
        plt.title(f"Real Data Model Comparison: {LOCATION} ({test_df.index[0].year}-{test_df.index[-1].year})")
        plt.ylabel("Rainfall (mm)")
        plt.legend()
        plt.draw()
        plt.pause(0.1) # Non-blocking pause to render plot

        # --- EXECUTE GRID SEARCH ---
        optimization_results = grid_search_optimization(xgb_preds, actuals)

        # --- VISUALIZE (The Heatmap) ---
        heatmap_data = optimization_results.pivot(index='Buy_Threshold', 
                                                  columns='Sell_Threshold', 
                                                  values='Net_Profit')

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="RdYlGn", cbar_kws={'label': 'Net Profit (₹)'})
        plt.title("Strategy Optimization Heatmap")
        plt.ylabel("Buy Threshold (mm)")
        plt.xlabel("Sell Threshold (mm)")
        plt.show()
    else:
        print("Failed to fetch data. Pipeline aborted.")

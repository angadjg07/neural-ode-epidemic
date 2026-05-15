import pandas as pd
import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from scipy.integrate import odeint
import os
import io
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Determine paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

class EpidemicDataset(Dataset):
    """PyTorch Dataset for epidemic sequences."""
    def __init__(self, data, seq_length=10, start_idx=0):
        # We store as float tensor
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.start_idx = start_idx
        self.samples = []
        self.times = []
        
        # Create sliding window sequences
        for i in range(len(self.data) - self.seq_length):
            # Target is the trajectory of length seq_length
            x = self.data[i:i+self.seq_length]
            t = torch.arange(self.start_idx + i, self.start_idx + i + self.seq_length, dtype=torch.float32)
            self.samples.append(x)
            self.times.append(t)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx], self.times[idx]

def fetch_jhu_covid_data():
    """Downloads Covid data from JHU."""
    print("Downloading JHU COVID-19 data...")
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    
    # Save raw
    df.to_csv(os.path.join(RAW_DIR, 'jhu_covid.csv'), index=False)
    
    # Data columns start at index 11 (1/22/20)
    date_cols = df.columns[11:] 
    daily_totals = df[date_cols].sum()
    daily_totals.index = pd.to_datetime(daily_totals.index)
    
    # Group into weekly bins by maximum cumulative cases
    weekly_totals = daily_totals.resample('W').max()
    
    # Convert cumulative to new cases
    weekly_new = weekly_totals.diff().fillna(0).clip(lower=0)
    
    return weekly_new.values, 330_000_000

def fetch_cdc_flu_data():
    """Downloads FluView data via Delphi Epidata API. (For completeness, though may fail)"""
    print("Downloading CDC Flu data via Delphi API...")
    url = "https://api.delphi.cmu.edu/epidata/fluview/?regions=nat&epiweeks=201501-201952"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json().get('epidata', [])
    if not data:
        raise ValueError("No data returned from Delphi API.")
    
    # Extract total ILI
    df = pd.DataFrame(data)
    weekly_new = df['ilitotal'].values
    return weekly_new, 330_000_000

def generate_synthetic_sir(N=330_000_000, days=700, beta=1.5, gamma=1.0):
    """
    Generates synthetic seasonal SIR trajectory safely fallback.
    Returns S, I, R state history.
    """
    print("Generating synthetic SIR data via ODE...")
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        # Add seasonality to beta (oscillates over a year)
        # Time is in weeks, so 52 weeks is one period
        seasonal_beta = beta * (1 + 0.3 * np.cos(2 * np.pi * t / 52))
        dSdt = -seasonal_beta * S * I / N
        dIdt = seasonal_beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions
    I0, R0 = 1000, 0
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    
    # Time grid (weekly integration)
    weeks = int(days / 7)
    t = np.linspace(0, weeks, weeks) 
    
    # Integrate ODE
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    
    # Return S, I, R and N
    return S, I, R, N

def fetch_cdc_dengue_data():
    """Downloads Puerto Rico Dengue data from CDC GitHub."""
    print("Downloading CDC Dengue Fever data...")
    url = "https://raw.githubusercontent.com/CDCgov/dengue_epidemic_thresholds/main/data/weekly_data_dengue_epidemic_alert_thresholds_1986_June2024.csv"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    
    # Save raw
    df.to_csv(os.path.join(RAW_DIR, 'cdc_dengue.csv'), index=False)
    
    # The dataset has 'conf_cases' column
    weekly_new = df['conf_cases'].fillna(0).clip(lower=0)
    
    return weekly_new.values, 3_200_000

def compute_sir_from_cases(cases, N):
    """
    Given an array of weekly new cases, estimate S, I, R.
    Assume infectious period ~1 week, so active I ~ new cases.
    """
    I = cases
    cumulative = np.cumsum(cases)
    
    # Recovered is roughly the cumulative minus current I
    R = cumulative - I
    R = np.clip(R, 0, None)
    
    # Susceptible is whatever is left
    S = N - I - R
    S = np.clip(S, 0, None)
    return S, I, R

def preprocess_data(S, I, R, N):
    """
    Smooths with a rolling average and normalizes to [0,1] fractions.
    Justification: Normalization ensures neural models converge more easily,
    while smoothing removes weekly reporting artifacts (like weekend batching).
    """
    df = pd.DataFrame({'S': S, 'I': I, 'R': R})
    
    # 3-week rolling average to cancel out reporting noise
    smoothed = df.rolling(window=3, min_periods=1).mean()
    
    # Normalize by population
    normalized = smoothed / N
    return normalized.values

def main():
    try:
        # Prefer empirical CDC Dengue data
        cases, N = fetch_cdc_dengue_data()
        S, I, R = compute_sir_from_cases(cases, N)
    except Exception as e:
        print(f"Failed to fetch empirical Dengue data: {e}")
        try:
            # Prefer empirical JHU COVID data
            cases, N = fetch_jhu_covid_data()
            S, I, R = compute_sir_from_cases(cases, N)
        except Exception as e2:
            print(f"Failed to fetch empirical COVID data: {e2}")
            # Robust synthetic fallback as per instructions
            S, I, R, N = generate_synthetic_sir()

    # Preprocess safely
    processed_data = preprocess_data(S, I, R, N)
    
    # Split strictly by time (80% train, 20% test); No shuffling! 
    # Justification: Never shuffle time series data, it leaks the future.
    split_idx = int(0.8 * len(processed_data))
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    # Save raw tensors to disk
    train_path = os.path.join(PROCESSED_DIR, 'train_data.pt')
    test_path = os.path.join(PROCESSED_DIR, 'test_data.pt')
    torch.save(torch.tensor(train_data, dtype=torch.float32), train_path)
    torch.save(torch.tensor(test_data, dtype=torch.float32), test_path)
    
    # Save CSV for humans
    df_out = pd.DataFrame(processed_data, columns=['S', 'I', 'R'])
    df_out.to_csv(os.path.join(PROCESSED_DIR, 'processed_sir.csv'), index=False)
    
    # Print metrics for verification
    print("\nData Pipeline Summary:")
    print(f"Total weeks: {len(processed_data)}")
    print(f"Train weeks: {len(train_data)}")
    print(f"Test weeks: {len(test_data)}")
    print(f"Initial S: {processed_data[0][0]:.4f}")
    print(f"Initial I: {processed_data[0][1]:.6f}")
    print(f"Initial R: {processed_data[0][2]:.6f}")
    if len(processed_data) > 0:
        print(f"Final I: {processed_data[-1][1]:.6f}")
    else:
        print("Final I: N/A")
    print("\nFiles saved successfully to data/processed/")

if __name__ == '__main__':
    main()

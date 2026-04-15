import pandas as pd
import numpy as np
import torch
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure models dir exists
os.makedirs(MODELS_DIR, exist_ok=True)

def sir_ode(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def simulate_sir(y0, t, beta, gamma):
    return odeint(sir_ode, y0, t, args=(beta, gamma))

def fit_sir_model(train_data):
    """
    Fits the SIR model by discovering parameters beta and gamma.
    Justification: We use multi-start optimization to avert local minima.
    """
    y0 = train_data[0]
    t = np.arange(len(train_data))
    
    def objective(params):
        beta, gamma = params
        y_pred = simulate_sir(y0, t, beta, gamma)
        # Minimize Mean Squared Error of the Infected compartment primarily
        # as it is the most crucial epidemiological curve.
        mse = np.mean((y_pred[:, 1] - train_data[:, 1])**2)
        return mse
        
    best_loss = np.inf
    best_params = None
    
    # 15 random restarts to avoid local minima
    np.random.seed(42)
    for _ in range(15):
        guess_beta = np.random.uniform(0.1, 5.0)
        guess_gamma = np.random.uniform(0.1, 2.0)
        
        res = minimize(
            objective, 
            [guess_beta, guess_gamma], 
            bounds=[(0.001, 15.0), (0.001, 5.0)],
            method='L-BFGS-B'
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
    return best_params[0], best_params[1]

def project_and_evaluate(beta, gamma, train_data, test_data):
    y0 = train_data[0]
    total_steps = len(train_data) + len(test_data)
    t_full = np.arange(total_steps)
    
    y_pred_full = simulate_sir(y0, t_full, beta, gamma)
    
    I_true_test = test_data[:, 1]
    # Prediction of testing phase starts right after training
    I_pred_test = y_pred_full[len(train_data):, 1]
    
    rmse = np.sqrt(mean_squared_error(I_true_test, I_pred_test))
    mae = mean_absolute_error(I_true_test, I_pred_test)
    
    # Peak timing error
    peak_true = np.argmax(I_true_test)
    peak_pred = np.argmax(I_pred_test)
    peak_timing_error = np.abs(peak_true - peak_pred)
    
    return y_pred_full, rmse, mae, peak_timing_error

def main():
    print("Loading data...")
    train_data = torch.load(os.path.join(PROCESSED_DIR, 'train_data.pt')).numpy()
    test_data = torch.load(os.path.join(PROCESSED_DIR, 'test_data.pt')).numpy()
    
    print("Fitting SIR model parameters via multiple restarts...")
    beta, gamma = fit_sir_model(train_data)
    R0 = beta / gamma
    
    print(f"Fitted beta: {beta:.4f}")
    print(f"Fitted gamma: {gamma:.4f}")
    print(f"R0 (Basic Reproduction Number): {R0:.4f}")
    
    y_pred_full, rmse, mae, peak_error = project_and_evaluate(beta, gamma, train_data, test_data)
    
    print("\nTest Set Evaluation (I compartment):")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Peak Timing Error (weeks): {peak_error}")
    
    # Save the parameters for our Neural/Hybrid models to start from
    result = {
        'beta': float(beta),
        'gamma': float(gamma),
        'R0': float(R0),
        'rmse': float(rmse),
        'mae': float(mae)
    }
    with open(os.path.join(MODELS_DIR, 'sir_result.pkl'), 'wb') as f:
        pickle.dump(result, f)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    t_train = np.arange(len(train_data))
    t_test = np.arange(len(train_data), len(train_data) + len(test_data))
    
    plt.plot(t_train, train_data[:, 1], 'b.', label='Train I (Observed)', alpha=0.5)
    plt.plot(t_test, test_data[:, 1], 'r.', label='Test I (Observed)', alpha=0.5)
    plt.plot(np.arange(len(y_pred_full)), y_pred_full[:, 1], 'k-', label='SIR Fit')
    plt.axvline(x=len(train_data), color='gray', linestyle='--', label='Train/Test Split')
    
    plt.title(f'Classical SIR Fit (R0 = {R0:.2f})')
    plt.xlabel('Weeks')
    plt.ylabel('Infected Fraction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(PROCESSED_DIR, 'sir_fit.png')
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}")
    print("Saved sir_result.pkl to models/")

if __name__ == '__main__':
    main()

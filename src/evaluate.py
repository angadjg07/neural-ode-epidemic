import os
import sys
import pickle
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.integrate import odeint

sys.path.append(os.path.dirname(__file__))
from node import LatentNeuralODE
from hybrid import HybridNeuralODE, analyse_learned_params
from sir import simulate_sir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def compute_metrics(y_true, y_pred):
    """Computes Evaluation standard mathematical metrics Scaled by 10k realistically."""
    I_true = y_true[:, 1]
    I_pred = y_pred[:, 1]
    
    # Scale physical magnitudes natively to [per 10k] representations structurally 
    # so error formats organically represent human readable 'Cases per 10,000' limits 
    # instead of microscopic fractions like 0.000008
    rmse = np.sqrt(mean_squared_error(I_true, I_pred)) * 10000.0
    mae = mean_absolute_error(I_true, I_pred) * 10000.0
    peak_true = np.argmax(I_true)
    peak_pred = np.argmax(I_pred)
    peak_error = np.abs(peak_true - peak_pred)
    
    return rmse, mae, peak_error

def generate_forecast_bands(model, x_context, forecast_horizon, n_samples=10, noise_std=1e-6, t_global=None):
    """
    Bootstraps the neural predictions by adding gaussian jitter to the input context.
    Returns the mean forecast and the 90% confidence interval standard deviation bounds.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            # Inject epistemic noise to context
            noisy_x = x_context + torch.randn_like(x_context) * noise_std
            noisy_x = torch.clamp(noisy_x, 0.0, 1.0)
            
            # Prediction is shape (batch_size=1, horizon, 3)
            pred = model(noisy_x, forecast_horizon=forecast_horizon, t_global=t_global)
            predictions.append(pred.squeeze(0).numpy())
            
    predictions = np.array(predictions) # shape: (n_samples, horizon, 3)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # 90% CI is ~1.645 * std
    lower_bound = np.clip(mean_pred - 1.645 * std_pred, 0, 1)
    upper_bound = np.clip(mean_pred + 1.645 * std_pred, 0, 1)
    
    return mean_pred, lower_bound, upper_bound

def evaluate_sir(train_data, test_data):
    """Evaluates the classical math baseline using the exact same 52-week rolling window logic."""
    with open(os.path.join(MODELS_DIR, 'sir_result.pkl'), 'rb') as f:
        sir_result = pickle.load(f)
        
    def sir_deriv(y, t, N, beta, gamma):
        S, I, R = y
        return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
        
    test_horizon = len(test_data)
    full_sir_pred = []
    
    for start_idx in range(0, test_horizon, 52):
        chunk_size = min(52, test_horizon - start_idx)
        # Use exact ground truth parameter arrays exclusively per independent window natively 
        y_test_0 = test_data[start_idx]
        t_chunk = np.arange(chunk_size)
        
        # Scipy uses floating integrals efficiently 
        chunk_sir_pred = odeint(sir_deriv, y_test_0, t_chunk, args=(1.0, sir_result['beta'], sir_result['gamma']))
        full_sir_pred.append(chunk_sir_pred)
        
    sir_pred = np.concatenate(full_sir_pred, axis=0)
    
    # Hide the "green spikes" Matplotlib artifacts physically connecting chunks.
    for boundary in range(51, test_horizon - 1, 52):
        sir_pred[boundary, :] = np.nan
        
    valid_idx = ~np.isnan(sir_pred[:, 1])
    rmse, mae, peak = compute_metrics(test_data[valid_idx], sir_pred[valid_idx])
    
    # We must return full trajectories encompassing train exactly mapped as baseline flat history natively
    # To retain backwards UI structural scaling, we stitch train and test identically.
    train_pred = np.zeros_like(train_data) # We don't evaluate train chunks physically for SIR 
    y_pred_full = np.concatenate([train_pred, sir_pred], axis=0)
    
    return y_pred_full, rmse, mae, peak

def evaluate_neural_model(model_cls, weights_name, train_data, test_data):
    """Initializes and evaluates a specific neural architecture using rolling 52-week forecasts."""
    model = model_cls(seq_length=5, hidden_dim=64)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, weights_name), map_location='cpu'))
    
    test_horizon = len(test_data)
    full_mean_pred, full_lower, full_upper = [], [], []
    
    # We stitch the last 5 weeks of training context seamlessly in front of the 406 test weeks
    full_context_timeline = np.concatenate([train_data[-5:], test_data], axis=0)
    
    # Evaluate progressively in realistic 52-week rolling segments
    for start_idx in range(0, test_horizon, 52):
        chunk_size = min(52, test_horizon - start_idx)
        
        # Grab the 5 ground-truth elements immediately preceding this forecast year chunk
        chunk_context = full_context_timeline[start_idx : start_idx + 5]
        context_tensor = torch.tensor(chunk_context, dtype=torch.float32).unsqueeze(0)
        
        t_global = torch.arange(len(train_data) + start_idx, len(train_data) + start_idx + chunk_size).unsqueeze(0).float()
        
        chunk_mean, chunk_lower, chunk_upper = generate_forecast_bands(
            model, context_tensor, forecast_horizon=chunk_size, n_samples=15, t_global=t_global
        )
        
        full_mean_pred.append(chunk_mean)
        full_lower.append(chunk_lower)
        full_upper.append(chunk_upper)
        
    
    mean_pred = np.concatenate(full_mean_pred, axis=0)
    lower = np.concatenate(full_lower, axis=0)
    upper = np.concatenate(full_upper, axis=0)
    
    # Visual Correction: Hide the "Green Spikes"
    # Matplotlib will organically draw straight lines between disjoint chunks if we concatenate them.
    # To stop the line dropping vertically between chunk boundaries, we insert `np.nan`
    # exactly at the chunk seams (every 52 weeks minus 1).
    for boundary in range(51, test_horizon - 1, 52):
        mean_pred[boundary, :] = np.nan
        
    valid_idx = ~np.isnan(mean_pred[:, 1])
    # Compute true localized metrics across the stitched sequential rollout
    rmse, mae, peak = compute_metrics(test_data[valid_idx], mean_pred[valid_idx])
    
    return model, mean_pred, lower, upper, (rmse, mae, peak)

def main():
    print("Loading data...")
    train_data = torch.load(os.path.join(PROCESSED_DIR, 'train_data.pt')).numpy()
    test_data = torch.load(os.path.join(PROCESSED_DIR, 'test_data.pt')).numpy()
    
    # 1. Classical SIR Check
    sir_full_traj, sir_rmse, sir_mae, sir_peak = evaluate_sir(train_data, test_data)
    sir_test_pred = sir_full_traj[len(train_data):]
    
    # 2. Neural ODE Check
    _, node_mean, node_lower, node_upper, node_metrics = evaluate_neural_model(
        LatentNeuralODE, 'final_latent_best.pt', train_data, test_data
    )
    
    # 3. Hybrid UDE Check
    hybrid_model, hybrid_mean, hybrid_lower, hybrid_upper, hybrid_metrics = evaluate_neural_model(
        HybridNeuralODE, 'final_hybrid_best.pt', train_data, test_data
    )

    # Output 1: Generate the Metrics CSV
    metrics_df = pd.DataFrame({
        'Model': ['Classical SIR', 'Latent Neural ODE', 'Hybrid UDE'],
        'RMSE (per 10k Cases)': [sir_rmse, node_metrics[0], hybrid_metrics[0]],
        'MAE (per 10k Cases)': [sir_mae, node_metrics[1], hybrid_metrics[1]],
        'Peak Timing Error (Weeks)': [sir_peak, node_metrics[2], hybrid_metrics[2]]
    })
    metrics_df.to_csv(os.path.join(PROCESSED_DIR, 'metrics_table.csv'), index=False)
    print("\nMetrics Table Output:")
    print(metrics_df)
    
    # Output 2: Generate R0 Extracted Physics
    # Run the hybrid model parameter extractor across the full timeline physically
    full_data = np.concatenate([train_data, test_data], axis=0)
    full_t = np.arange(len(full_data))
    
    df_params = analyse_learned_params(
        hybrid_model, 
        torch.tensor(full_data, dtype=torch.float32), 
        full_t
    )
    df_params.to_csv(os.path.join(PROCESSED_DIR, 'learned_params.csv'), index=False)
    df_params.to_json(os.path.join(PROCESSED_DIR, 'learned_params.json'), orient='records')
    
    with open(os.path.join(MODELS_DIR, 'sir_result.pkl'), 'rb') as f:
        sir_params = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, 'sir_params.json'), 'w') as f:
        json.dump(sir_params, f)
    
    # Output 3: R0 Trajectory Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df_params['week'], df_params['R0'], 'b-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Epidemic Threshold')
    plt.axvline(x=len(train_data), color='k', linestyle=':', label='Train/Test Split')
    plt.title('Inferred Reproduction Number $R_0(t)$ (Hybrid UDE)', fontsize=14)
    plt.xlabel('Week')
    plt.ylabel('$R_0$')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'r0_trajectory.png'))
    plt.close()

    # Output 4: Master 3-Model Comparison Plot
    plt.figure(figsize=(14, 7))
    t_train = np.arange(len(train_data))
    t_test = np.arange(len(train_data), len(train_data) + len(test_data))
    
    # Observations
    plt.scatter(t_train, train_data[:, 1], color='gray', s=10, label='Train I (Observed)', alpha=0.5)
    plt.scatter(t_test, test_data[:, 1], color='black', s=25, label='Test I (Observed)')
    
    # SIR Line
    plt.plot(t_test, sir_test_pred[:, 1], 'r--', label='Classical SIR Forecast')
    
    # Node Line + Band
    plt.plot(t_test, node_mean[:, 1], 'b-', label='Latent Neural ODE')
    plt.fill_between(t_test, node_lower[:, 1], node_upper[:, 1], color='blue', alpha=0.15)
    
    # Hybrid Line + Band
    plt.plot(t_test, hybrid_mean[:, 1], 'g-', label='Hybrid Universal DE')
    plt.fill_between(t_test, hybrid_lower[:, 1], hybrid_upper[:, 1], color='green', alpha=0.15)
    
    plt.axvline(x=len(train_data), color='k', linestyle=':')
    max_val = max(train_data[:, 1].max(), test_data[:, 1].max())
    plt.ylim(-0.0001, max_val * 3.5)
    plt.title('Out-of-Distribution Epidemic Forecasting Comparison', fontsize=16)
    plt.xlabel('Weeks Since Onset')
    plt.ylabel('Infected Fraction (I)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'comparison.png'))
    plt.close()
    
    print("\nSuccessfully generated all required artifacts in data/processed/:")
    print(" - comparison.png")
    print(" - r0_trajectory.png")
    print(" - metrics_table.csv")
    print(" - learned_params.csv")

if __name__ == '__main__':
    main()

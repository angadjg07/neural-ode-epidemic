import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(__file__))
from node import LatentNeuralODE
from hybrid import HybridNeuralODE, analyse_learned_params
from sir import simulate_sir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def compute_metrics(y_true, y_pred):
    """Computes MAE, RMSE and Peak Error between two trajectories for Compartment I."""
    I_true = y_true[:, 1]
    I_pred = y_pred[:, 1]
    
    rmse = np.sqrt(mean_squared_error(I_true, I_pred))
    mae = mean_absolute_error(I_true, I_pred)
    peak_true = np.argmax(I_true)
    peak_pred = np.argmax(I_pred)
    peak_error = np.abs(peak_true - peak_pred)
    
    return rmse, mae, peak_error

def generate_forecast_bands(model, x_context, forecast_horizon, n_samples=10, noise_std=0.02):
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
            pred = model(noisy_x, forecast_horizon=forecast_horizon)
            predictions.append(pred.squeeze(0).numpy())
            
    predictions = np.array(predictions) # shape: (n_samples, horizon, 3)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # 90% CI is ~1.645 * std
    lower_bound = np.clip(mean_pred - 1.645 * std_pred, 0, 1)
    upper_bound = np.clip(mean_pred + 1.645 * std_pred, 0, 1)
    
    return mean_pred, lower_bound, upper_bound

def evaluate_sir(train_data, test_data):
    """Evaluates the classical math baseline"""
    with open(os.path.join(MODELS_DIR, 'sir_result.pkl'), 'rb') as f:
        sir_params = pickle.load(f)
        
    beta, gamma = sir_params['beta'], sir_params['gamma']
    y0 = train_data[0]
    total_steps = len(train_data) + len(test_data)
    t_full = np.arange(total_steps)
    y_pred_full = simulate_sir(y0, t_full, beta, gamma)
    
    # Extract only the test portion forecast
    test_pred = y_pred_full[len(train_data):]
    rmse, mae, peak = compute_metrics(test_data, test_pred)
    
    return y_pred_full, rmse, mae, peak

def evaluate_neural_model(model_cls, weights_name, train_data, test_data):
    """Initializes and evaluates a specific neural architecture."""
    model = model_cls(seq_length=5, hidden_dim=32)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, weights_name), map_location='cpu'))
    
    # Context is the last `seq_length` items of the train_data
    # This simulates actual forecasting where we only know the "present"
    context = torch.tensor(train_data[-5:], dtype=torch.float32).unsqueeze(0)
    
    mean_pred, lower, upper = generate_forecast_bands(
        model, context, forecast_horizon=len(test_data), n_samples=15
    )
    
    # Compute metrics exactly identically on the test horizon
    rmse, mae, peak = compute_metrics(test_data, mean_pred)
    
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
        'RMSE': [sir_rmse, node_metrics[0], hybrid_metrics[0]],
        'MAE': [sir_mae, node_metrics[1], hybrid_metrics[1]],
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

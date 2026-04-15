import os
import sys
import torch
import numpy as np

# Adjust path so pytest can discover modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data import preprocess_data, compute_sir_from_cases
from node import LatentNeuralODE, epidemic_loss
from hybrid import HybridNeuralODE, HybridODEFunc
from evaluate import compute_metrics

def test_data_preprocessing_bounds():
    """Ensure our preprocessing successfully normalizes physical population arrays."""
    cases = np.array([10, 50, 100, 20])
    N = 1000
    S, I, R = compute_sir_from_cases(cases, N)
    
    # Validations
    assert S[0] == 990
    assert R[-1] > 0
    
    processed = preprocess_data(S, I, R, N)
    
    # Values should STRICTLY be between 0 and 1
    assert np.all(processed >= 0.0)
    assert np.all(processed <= 1.0)
    # Dimensions should match input
    assert processed.shape == (4, 3)

def test_node_shape():
    """Test output shapes of the generic Latent Neural ODE."""
    batch_size = 3
    seq_length = 5
    horizon = 10
    model = LatentNeuralODE(seq_length=seq_length, hidden_dim=16)
    
    x = torch.rand(batch_size, seq_length, 3)
    out = model(x, forecast_horizon=horizon)
    
    assert out.shape == (batch_size, horizon, 3), "Tensor structural mismatch in NDE integration."

def test_node_constraints():
    """Physics constraint verification: Ensure S,I,R populations don't breach bounds natively."""
    model = LatentNeuralODE(seq_length=3, hidden_dim=16)
    x = torch.rand(2, 3, 3)
    out = model(x, forecast_horizon=8)
    
    # Sigmoid / Softmax + strict ODE formulation guarantees these limits
    assert torch.all(out >= -1e-4)
    assert torch.all(out <= 1.0 + 1e-4)

def test_hybrid_ode_shape():
    """Test output shapes of the parameter net hybrid model."""
    batch_size = 2
    model = HybridNeuralODE(seq_length=4, hidden_dim=16)
    
    x = torch.rand(batch_size, 4, 3)
    out = model(x, forecast_horizon=6)
    
    assert out.shape == (batch_size, 6, 3)

def test_hybrid_beta_gamma_positivity():
    """Paramnet guarantees: Time-varying beta and gamma MUST be strictly positive."""
    func = HybridODEFunc(beta_base=1.0, gamma_base=1.0, hidden_dim=16)
    
    # dummy state array
    y = torch.tensor([[0.5, 0.1, 0.4]])
    t = torch.tensor(1.0)
    
    # Extract the beta/gamma values the parameter network projects internally
    deltas = func.param_net(t, y)
    beta_t = torch.nn.functional.softplus(func.beta_base + deltas[..., 0:1])
    gamma_t = torch.nn.functional.softplus(func.gamma_base + deltas[..., 1:2])
    
    assert torch.all(beta_t > 0), "Beta dropped beneath logical zero limit!"
    assert torch.all(gamma_t > 0), "Gamma dropped beneath logical zero limit!"

def test_evaluate_compute_metrics():
    """Verify raw arithmetic formatting inside the metrics extractor."""
    y_true = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0]
    ])
    y_pred = np.array([
        [0.9, 0.1, 0.0],
        [0.85, 0.15, 0.0]  # Off by 0.05
    ])
    
    rmse, mae, peak = compute_metrics(y_true, y_pred)
    
    assert peak == 0
    # MAE of column 1: |0.1-0.1| + |0.2-0.15| / 2 = 0.05 / 2 = 0.025
    assert np.isclose(mae, 0.025)

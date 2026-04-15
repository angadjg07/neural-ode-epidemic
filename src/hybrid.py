import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import os
import pickle
import pandas as pd
import numpy as np
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class ParameterNet(nn.Module):
    """
    Predicts corrections to beta and gamma using [S, I, R] and seasonal time encoding.
    """
    def __init__(self, hidden_dim=32):
        super(ParameterNet, self).__init__()
        # Input: S, I, R (3) + sin(t), cos(t) (2) = 5
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # output: delta_beta, delta_gamma
        )
        
    def forward(self, t, y):
        # Time encoding (assuming t is in weeks, 52 weeks a year)
        # Usually, torchdiffeq calls with scalar t
        t_val = t.item() if isinstance(t, torch.Tensor) and t.numel() == 1 else t
        
        if isinstance(t_val, float) or (isinstance(t_val, torch.Tensor) and t_val.numel() == 1):
            sin_t = math.sin(2 * math.pi * t_val / 52.0)
            cos_t = math.cos(2 * math.pi * t_val / 52.0)
            batch_size = y.shape[0] if y.dim() > 1 else 1
            time_feat = torch.tensor([sin_t, cos_t], device=y.device, dtype=y.dtype)
            if y.dim() > 1:
                time_feat = time_feat.unsqueeze(0).repeat(batch_size, 1)
        else:
            # Handles if t is an array or vector (used post-training for analysis)
            sin_t = torch.sin(2 * math.pi * t / 52.0).unsqueeze(-1)
            cos_t = torch.cos(2 * math.pi * t / 52.0).unsqueeze(-1)
            time_feat = torch.cat([sin_t, cos_t], dim=-1)
            
        time_feat = time_feat.to(y.device)
        
        # Concat [S, I, R] with [sin_t, cos_t]
        x = torch.cat([y, time_feat], dim=-1)
        
        # Return [delta_beta, delta_gamma]
        return self.net(x)

class HybridODEFunc(nn.Module):
    def __init__(self, beta_base, gamma_base, hidden_dim=32):
        super(HybridODEFunc, self).__init__()
        self.beta_base = nn.Parameter(torch.tensor([float(beta_base)]))
        self.gamma_base = nn.Parameter(torch.tensor([float(gamma_base)]))
        
        self.param_net = ParameterNet(hidden_dim=hidden_dim)
        
    def forward(self, t, y):
        # y shape: (batch_size, 3) representing S, I, R
        S = y[..., 0:1]
        I = y[..., 1:2]
        
        # Get parameter corrections
        deltas = self.param_net(t, y)
        delta_beta = deltas[..., 0:1]
        delta_gamma = deltas[..., 1:2]
        
        # Compute dynamic beta and gamma
        # Use softplus instead of absolute value to ensure continuous gradients physically bounds to >0
        beta_t = nn.functional.softplus(self.beta_base + delta_beta)
        gamma_t = nn.functional.softplus(self.gamma_base + delta_gamma)
        
        # Classical SIR conceptually embedded directly as physical layers
        dSdt = -beta_t * S * I
        dIdt = beta_t * S * I - gamma_t * I
        dRdt = gamma_t * I
        
        # Concatenate derivatives
        dy = torch.cat([dSdt, dIdt, dRdt], dim=-1)
        return dy

class HybridNeuralODE(nn.Module):
    def __init__(self, seq_length=10, hidden_dim=64):
        super(HybridNeuralODE, self).__init__()
        self.seq_length = seq_length
        self.encoder = nn.GRU(input_size=3, hidden_size=hidden_dim, batch_first=True)
        
        self.y0_net = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1) # Perfect topological conservation guarantees
        )
        
        # Load baseline parameters from physical fit exactly 
        with open(os.path.join(MODELS_DIR, 'sir_result.pkl'), 'rb') as f:
            sir_result = pickle.load(f)
            
        beta_0 = sir_result['beta']
        gamma_0 = sir_result['gamma']
            
        self.ode_func = HybridODEFunc(beta_base=beta_0, gamma_base=gamma_0, hidden_dim=hidden_dim)
        
    def forward(self, x, forecast_horizon=None):
        if forecast_horizon is None:
            forecast_horizon = self.seq_length
            
        _, h_n = self.encoder(x)
        h_n = h_n.squeeze(0)
        y0 = self.y0_net(h_n)
        
        t = torch.linspace(0., forecast_horizon - 1, forecast_horizon, device=x.device)
        pred_y = odeint(self.ode_func, y0, t, method='dopri5')
        pred_y = pred_y.permute(1, 0, 2)
        
        return pred_y

def analyse_learned_params(model, y_trajectory, t_array):
    """
    Extracts time-varying beta(t), gamma(t), and R0(t) from the hybrid model.
    Returns a DataFrame holding the interpretation of neural mechanics.
    """
    model.eval()
    with torch.no_grad():
        beta_base = model.ode_func.beta_base.item()
        gamma_base = model.ode_func.gamma_base.item()
        
        t_tensor = torch.tensor(t_array, dtype=torch.float32, device=y_trajectory.device)
        betas = []
        gammas = []
        r0s = []
        
        t_global = 0 # Assume we start at week 0 of year
        for idx in range(len(t_array)):
            t_val = t_tensor[idx] + t_global # In real deployments, track global time!
            y_val = y_trajectory[idx:idx+1]
            
            deltas = model.ode_func.param_net(t_val, y_val)
            db = deltas[0, 0].item()
            dg = deltas[0, 1].item()
            
            beta_t = nn.functional.softplus(torch.tensor(beta_base + db)).item()
            gamma_t = nn.functional.softplus(torch.tensor(gamma_base + dg)).item()
            
            betas.append(beta_t)
            gammas.append(gamma_t)
            r0s.append(beta_t / gamma_t if gamma_t > 0 else 0)
            
    df = pd.DataFrame({
        'week': np.arange(len(t_array)),
        'beta': betas,
        'gamma': gammas,
        'R0': r0s
    })
    return df

def main():
    print("Testing Hybrid Neural ODE Components...\n")
    batch_size = 4
    seq_length = 5
    forecast_horizon = 10
    
    x = torch.rand(batch_size, seq_length, 3)
    
    if not os.path.exists(os.path.join(MODELS_DIR, 'sir_result.pkl')):
        print("ERROR: sir_result.pkl not found. Please run sir.py first.")
        return
        
    model = HybridNeuralODE(seq_length=seq_length, hidden_dim=32)
    print("Hybrid UDE Model instaniated from baselines.")
    
    pred_y = model(x, forecast_horizon=forecast_horizon)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (expected 4, 10, 3): {pred_y.shape}")
    
    y_target = pred_y[0] 
    t_arr = np.arange(forecast_horizon)
    df = analyse_learned_params(model, y_target, t_arr)
    
    print("\nExtracted Learned Parameters Table (Sample over 10 weeks):")
    print(df.head())

if __name__ == '__main__':
    main()

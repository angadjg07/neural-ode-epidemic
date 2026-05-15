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
    Predicts corrections to beta and gamma using [S, I, R] autonomously without broken relative time embeddings.
    """
    def __init__(self, hidden_dim=64, latent_dim=16):
        super(ParameterNet, self).__init__()
        # Input: S, I, R (3) + sin(t), cos(t) (2) + Latent Context (latent_dim)
        in_feats = 5 + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2) # output: delta_beta, delta_gamma
        )
        
    def forward(self, t, y, t_start=None, e_c=None):
        if t_start is not None:
            t_val = t_start.unsqueeze(-1) + t # shape (batch_size, 1)
        else:
            t_val = t
            
        sin_t = torch.sin(2 * math.pi * t_val / 52.0)
        cos_t = torch.cos(2 * math.pi * t_val / 52.0)
        
        # Ensure sin_t matches y's dimensions
        if sin_t.dim() < y.dim():
            sin_t = sin_t.expand(*y.shape[:-1], 1)
            cos_t = cos_t.expand(*y.shape[:-1], 1)
            
        time_feat = torch.cat([sin_t, cos_t], dim=-1).to(y.device)
        
        if e_c is not None:
            # Expand e_c if needed
            if e_c.dim() < y.dim():
                expanded_e_c = e_c.unsqueeze(1).expand(*y.shape[:-1], e_c.shape[-1])
            else:
                expanded_e_c = e_c.expand(*y.shape[:-1], e_c.shape[-1])
            x = torch.cat([y, time_feat, expanded_e_c], dim=-1)
        else:
            # Fallback if no latent space provided
            zero_e_c = torch.zeros(*y.shape[:-1], 16, device=y.device)
            x = torch.cat([y, time_feat, zero_e_c], dim=-1)
            
        # Return [delta_beta, delta_gamma]
        return self.net(x)

class HybridODEFunc(nn.Module):
    def __init__(self, beta_base, gamma_base, hidden_dim=32, latent_dim=16):
        super(HybridODEFunc, self).__init__()
        self.beta_base = nn.Parameter(torch.tensor([float(beta_base)]))
        self.gamma_base = nn.Parameter(torch.tensor([float(gamma_base)]))
        
        self.param_net = ParameterNet(hidden_dim=hidden_dim, latent_dim=latent_dim)
        
    def forward(self, t, y):
        # y shape: (batch_size, 3) representing S, I, R
        S = y[..., 0:1]
        I = y[..., 1:2]
        
        # Get parameter corrections
        t_start = getattr(self, 't_start', None)
        e_c = getattr(self, 'e_c', None)
        deltas = self.param_net(t, y, t_start=t_start, e_c=e_c)
        delta_beta = deltas[..., 0:1]
        delta_gamma = deltas[..., 1:2]
        
        # Compute dynamic beta and gamma
        # Use torch.clamp to strictly physically bound parameters (0 to 5) directly 
        # WITHOUT artificially wrapping the analytical baseline inside exponential limits identically at scale.
        beta_t = torch.clamp(self.beta_base + delta_beta, min=0.0, max=5.0)
        gamma_t = torch.clamp(self.gamma_base + delta_gamma, min=0.5, max=2.0)
        
        # Classical SIR conceptually embedded directly as physical layers
        dSdt = -beta_t * S * I
        dIdt = beta_t * S * I - gamma_t * I
        dRdt = gamma_t * I
        
        # Concatenate derivatives
        dy = torch.cat([dSdt, dIdt, dRdt], dim=-1)
        return dy

class HybridNeuralODE(nn.Module):
    def __init__(self, seq_length=10, hidden_dim=64, latent_dim=16):
        super(HybridNeuralODE, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        # PROTOTYPE FACTOR B: Removing the Recurrent GRU
        # Deep learning literature indicates that sparse epidemic datasets (10e-4) often 
        # suffer from gradient vanishing across recursive timestamps. 
        # We explicitly flatten the local sequential memory into a Dense MLP mapping instead.
        self.flattened_input_dim = seq_length * 3
        self.encoder = nn.Sequential(
            nn.Linear(self.flattened_input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
        
        # Load baseline parameters from physical fit exactly 
        with open(os.path.join(MODELS_DIR, 'sir_result.pkl'), 'rb') as f:
            sir_result = pickle.load(f)
            
        beta_0 = sir_result['beta']
        gamma_0 = sir_result['gamma']
            
        self.ode_func = HybridODEFunc(beta_base=beta_0, gamma_base=gamma_0, hidden_dim=hidden_dim, latent_dim=latent_dim)
        
    def forward(self, x, forecast_horizon=None, t_global=None):
        if forecast_horizon is None:
            forecast_horizon = self.seq_length
            
        # Encode Context Memory (Dense Factor B)
        x_flat = x.reshape(x.size(0), -1) # Flatten (batch_size, seq_length * 3)
        e_c = self.encoder(x_flat) # shape (batch_size, latent_dim)
        self.ode_func.e_c = e_c
            
        # Initial condition is identically the last observation in the context window
        y0 = x[:, -1, :]
        
        if t_global is not None:
            # t_global[0] corresponds to the first pred point.
            # y0 corresponds to t_global[0] - 1
            self.ode_func.t_start = t_global[:, 0] - 1.0
        else:
            self.ode_func.t_start = torch.zeros(x.shape[0], device=x.device)
            
        t = torch.arange(0., forecast_horizon + 1, device=x.device)
        pred_y = odeint(self.ode_func, y0, t, method='rk4', options={'step_size': 0.25})
        
        pred_y = pred_y[1:].permute(1, 0, 2)
        
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
        # Build dynamic representations simulating strict rolling sequences 
        for idx in range(len(t_array)):
            t_val = t_tensor[idx] + 0 
            y_val = torch.tensor(y_trajectory[idx:idx+1], dtype=torch.float32, device=y_trajectory.device)
            
            # Form context vector dynamically
            if idx >= 5:
                context_slice = y_trajectory[idx-5:idx]
                context_tensor = torch.tensor(context_slice, dtype=torch.float32, device=y_trajectory.device).unsqueeze(0)
                e_c_val = model.encoder(context_tensor.reshape(1, -1)) 
            else:
                # If no historical padding exists, pad to zero initially!
                context_tensor = torch.zeros((1, 5, 3), dtype=torch.float32, device=y_trajectory.device)
                if idx > 0:
                    context_tensor[0, -idx:, :] = torch.tensor(y_trajectory[0:idx], dtype=torch.float32, device=y_trajectory.device)
                e_c_val = model.encoder(context_tensor.reshape(1, -1))
                
            deltas = model.ode_func.param_net(t_val, y_val, t_start=None, e_c=e_c_val)
            db = deltas[0, 0].item()
            dg = deltas[0, 1].item()
            
            beta_t = max(0.0, min(5.0, beta_base + db))
            gamma_t = max(0.5, min(2.0, gamma_base + dg))
            
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

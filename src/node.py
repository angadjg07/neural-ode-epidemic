import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import os

class ODEFunc(nn.Module):
    """
    Component 1: The ODE function (the neural network).
    Parameterizes dY/dt where Y = [S, I, R].
    """
    def __init__(self, hidden_dim=64):
        super(ODEFunc, self).__init__()
        # A small MLP
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # Only predict dS and dR independently
        )
    
    def forward(self, t, y):
        # y shape: (batch_size, 3)
        out = self.net(y)
        
        # Epidemiological constraints (Physics-Informed)
        # dS must always be non-positive (susceptibles only decrease)
        # We multiply by positive S so that if S hits 0, dS becomes 0.
        dS = -torch.relu(out[:, 0:1]) * torch.relu(y[:, 0:1])
        
        # dR must always be non-negative (recovered never decrease)
        # Recovered comes from Infected, so it stops if I hits 0.
        dR = torch.relu(out[:, 1:2]) * torch.relu(y[:, 1:2])
        
        # dS + dI + dR must sum to zero (population is conserved)
        # Therefore: dI = -(dS + dR)
        dI = -(dS + dR)
        
        # Concatenate derivatives back to [dS, dI, dR]
        dy = torch.cat([dS, dI, dR], dim=-1)
        return dy

class LatentNeuralODE(nn.Module):
    """
    Component 3: The full model with encoder and ODE solver.
    """
    def __init__(self, seq_length=10, hidden_dim=64):
        super(LatentNeuralODE, self).__init__()
        self.seq_length = seq_length
        
        # Encoder: reads context window
        self.encoder = nn.GRU(input_size=3, hidden_size=hidden_dim, batch_first=True)
        # Project hidden state from GRU to initial condition y0 = [S0, I0, R0]
        self.y0_net = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1) # Hard constraint to keep initial states strictly summing to 1
        )
        
        # ODE Function
        self.ode_func = ODEFunc(hidden_dim=hidden_dim)
        
    def forward(self, x, forecast_horizon=None):
        """
        x: (batch_size, seq_length, 3)
        """
        if forecast_horizon is None:
            forecast_horizon = self.seq_length
            
        # Encode context window
        _, h_n = self.encoder(x)
        # h_n shape: (1, batch_size, hidden_dim). Squeeze out the layer dim.
        h_n = h_n.squeeze(0)
        
        # Map to initial condition
        y0 = self.y0_net(h_n)
        
        # Time steps to integrate over
        # We start at 0, go up to forecast_horizon
        t = torch.linspace(0., forecast_horizon - 1, forecast_horizon, device=x.device)
        
        # Integrate via COMPONENT 2 (torchdiffeq wrapper using adjoint method)
        # odeint output shape: (forecast_horizon, batch_size, 3)
        pred_y = odeint(self.ode_func, y0, t, method='dopri5')
        
        # Permute to (batch_size, forecast_horizon, 3)
        pred_y = pred_y.permute(1, 0, 2)
        
        return pred_y

def epidemic_loss(pred_y, true_y):
    """
    Loss function that combines MSE with soft physical physics constraints.
    """
    # Base MSE loss
    mse = nn.MSELoss()(pred_y, true_y)
    
    # Soft penalize values outside [0, 1] bounds
    # Since y0 is sigmoid, violating bounds primarily comes from the ODE integration
    penalty_upper = torch.relu(pred_y - 1.0).mean()
    penalty_lower = torch.relu(-pred_y).mean()
    
    return mse + 10.0 * (penalty_upper + penalty_lower)


def main():
    print("Testing Neural ODE Components...\n")
    
    batch_size = 16
    seq_length = 5
    forecast_horizon = 10
    
    # Dummy input
    x = torch.rand(batch_size, seq_length, 3)
    
    # Instantiate model
    model = LatentNeuralODE(seq_length=seq_length, hidden_dim=32)
    print(f"Model instaniated.")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Forward pass
    pred_y = model(x, forecast_horizon=forecast_horizon)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (expected {batch_size}, {forecast_horizon}, 3): {pred_y.shape}")
    
    # Verify range constraints
    min_val = pred_y.min().item()
    max_val = pred_y.max().item()
    print(f"\nValue bounds - Min: {min_val:.4f}, Max: {max_val:.4f}")
    if min_val >= -1e-4 and max_val <= 1.0 + 1e-4:
        print("PASS: Values are strictly contained near [0, 1].")
    else:
        print("WARNING: Values escaped [0, 1] bounds.")
        
    # Verify conservation constraint explicitly
    # Sum over dim=-1 should be 1.0 (or identical to the initial condition sum)
    # Actually, the derivatives sum to 0, meaning the sum is constant over time.
    y0_sum = pred_y[:, 0, :].sum(dim=-1)
    y_last_sum = pred_y[:, -1, :].sum(dim=-1)
    diff = torch.abs(y0_sum - y_last_sum).max().item()
    print(f"Max population drift over integration: {diff:.8f}")
    
    # Test loss and backward pass
    true_y = torch.rand_like(pred_y) # dummy target
    loss = epidemic_loss(pred_y, true_y)
    print(f"\nCalculated Loss: {loss.item():.4f}")
    
    print("Executing backward pass (Adjoint via torchdiffeq)...")
    loss.backward()
    print("Backward pass executed successfully. Gradients computed.")

if __name__ == '__main__':
    main()

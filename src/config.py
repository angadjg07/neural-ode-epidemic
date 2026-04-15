from dataclasses import dataclass

@dataclass
class Config:
    """Central configuration for Neural ODE Epidemic Simulator."""
    
    # Data params
    batch_size: int = 32         # Number of samples per batch
    seq_length: int = 10         # Length of sequence for training
    
    # Model params
    hidden_dim: int = 64         # Dimension of hidden layers
    latent_dim: int = 3          # S, I, R compartments
    
    # Training params
    learning_rate: float = 1e-3  # Optimizer learning rate
    epochs: int = 100            # Number of training epochs
    
    # ODE Solver params
    solver: str = 'dopri5'       # ODE solver method
    rtol: float = 1e-7           # Relative tolerance for ODE solver
    atol: float = 1e-9           # Absolute tolerance for ODE solver

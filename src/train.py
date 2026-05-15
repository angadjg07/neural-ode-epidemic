import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))

from node import LatentNeuralODE, epidemic_loss
from hybrid import HybridNeuralODE
from data import EpidemicDataset
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

class Trainer:
    def __init__(self, model, device, epochs, run_name):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.run_name = run_name
        self.best_loss = float('inf')
        self.patience = 50
        self.patience_counter = 0
        
        # Optimizer with Cosine Annealing Learning Rate
        # Justification: Adjoint solvers struggle with static LRs, decaying them 
        # smoothly allows tighter loss boundaries without ODE numerical overshooting.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
    def train(self, train_loader, val_loader):
        train_losses = []
        val_losses = []
        
        print(f"Starting training on {self.device} for {self.epochs} epochs...")
        # Expected Time: CPU takes roughly 5-10 minutes for 500 epochs depending on batch size.
        
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch_x, batch_t in train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_t = batch_t.to(self.device).float()
                
                # Context is first 5 timesteps, we forecast the remainder to evaluate loss
                context_length = 5
                
                # Protect smaller sets
                if batch_x.size(1) <= context_length:
                    continue
                    
                x_context = batch_x[:, :context_length, :]
                true_y = batch_x[:, context_length:, :]
                t_global = batch_t[:, context_length:] # Global indices for the predicted window
                forecast_horizon = true_y.size(1)
                
                self.optimizer.zero_grad()
                pred_y = self.model(x_context, forecast_horizon=forecast_horizon, t_global=t_global)
                
                loss = epidemic_loss(pred_y, true_y)
                if torch.isnan(loss).any():
                    print("Early exit: NaN loss detected! Numerical instability in ODE.")
                    return train_losses, val_losses
                    
                loss.backward()
                
                # Gradient Clipping is essential: Solvers explode if state changes abruptly!
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / max(len(train_loader), 1)
            train_losses.append(avg_train_loss)
            
            # Validation Pass
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, batch_t in val_loader:
                    batch_x = batch_x.to(self.device).float()
                    batch_t = batch_t.to(self.device).float()
                    if batch_x.size(1) <= context_length: continue
                    x_context = batch_x[:, :context_length, :]
                    true_y = batch_x[:, context_length:, :]
                    t_global = batch_t[:, context_length:]
                    pred_y = self.model(x_context, forecast_horizon=true_y.size(1), t_global=t_global)
                    total_val_loss += epidemic_loss(pred_y, true_y).item()
            
            avg_val_loss = total_val_loss / max(len(val_loader), 1)
            val_losses.append(avg_val_loss)
            
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                torch.save(self.model.state_dict(), os.path.join(MODELS_DIR, f"{self.run_name}_best.pt"))
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            self.scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d}/{self.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {self.scheduler.get_last_lr()[0]:.6f}")
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1} (patience={self.patience})")
                break
                
        self.plot_curves(train_losses, val_losses)
        return train_losses, val_losses
        
    def plot_curves(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.title(f'Training Curves - {self.run_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(alpha=0.3)
        plt_path = os.path.join(PROCESSED_DIR, f"{self.run_name}_curve.png")
        plt.savefig(plt_path)
        print(f"Saved learning curves to {plt_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['latent', 'hybrid'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--run_name', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    torch.manual_seed(42)
    
    print("Loading datasets...")
    train_data = torch.load(os.path.join(PROCESSED_DIR, 'train_data.pt'))
    test_data = torch.load(os.path.join(PROCESSED_DIR, 'test_data.pt'))
    
    seq_len = 45 # Provide 5 weeks context, 40 weeks prediction
    
    train_dataset = EpidemicDataset(train_data, seq_length=seq_len)
    test_dataset = EpidemicDataset(test_data, seq_length=seq_len, start_idx=len(train_data))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print(f"Initializing {args.model.upper()} model...")
    if args.model == 'latent':
        model = LatentNeuralODE(seq_length=5, hidden_dim=64)
    else:
        model = HybridNeuralODE(seq_length=5, hidden_dim=64)
        
    trainer = Trainer(model, device, args.epochs, args.run_name)
    trainer.train(train_loader, test_loader)
    print("Training finished.")
    
if __name__ == '__main__':
    main()

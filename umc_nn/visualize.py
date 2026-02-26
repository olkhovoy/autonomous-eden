import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from core import UMC_Cell
from train import generate_synthetic_data, UMC_Network
import os
from tqdm import tqdm

class Baseline_Network(nn.Module):
    """
    A standard recurrent neural network without UMC architectural constraints.
    No spectral normalization, no macro-state downward causation, no KM-iteration.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Standard RNN with Tanh (similar non-linearity but unconstrained)
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.readout = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x_seq):
        out, h_n = self.rnn(x_seq)
        # Readout from the last step
        pred = self.readout(out[:, -1, :])
        return pred, h_n

def train_model(model, dataloader, epochs, device, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    with tqdm(total=epochs, desc=f"Training {model.__class__.__name__}") as pbar:
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs, _ = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            pbar.set_postfix({'loss': f"{total_loss/len(dataloader):.4f}"})
            pbar.update(1)
    return model

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters (matched to the successful run)
    input_dim = 32
    num_nodes = 4
    hidden_dim = 128
    output_dim = 32
    seq_len = 15
    batch_size = 256
    epochs = 150 # slightly fewer epochs for quick visualization
    
    # 1. Generate Data
    X_train, Y_train = generate_synthetic_data(5000, seq_len, input_dim)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Models
    umc_model = UMC_Network(input_dim, num_nodes, hidden_dim, output_dim, k_contractive=0.9990).to(device)
    # Baseline hidden dim is equal to parameter count roughly
    baseline_model = Baseline_Network(input_dim, int(hidden_dim * 2.633), output_dim).to(device)
    
    # 3. Train Models
    print("\n--- Training Models ---")
    train_model(umc_model, dataloader, epochs, device, lr=0.0052)
    train_model(baseline_model, dataloader, epochs, device)
    
    # 4. Evaluation and Plotting
    umc_model.eval()
    baseline_model.eval()
    
    with torch.no_grad():
        # Get a test batch
        X_test, Y_test = generate_synthetic_data(50, seq_len, input_dim)
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        
        umc_pred, _ = umc_model(X_test)
        base_pred, _ = baseline_model(X_test)
        
        Y_test_np = Y_test.cpu().numpy()
        umc_pred_np = umc_pred.cpu().numpy()
        base_pred_np = base_pred.cpu().numpy()
        
        umc_error = np.abs(Y_test_np - umc_pred_np)
        base_error = np.abs(Y_test_np - base_pred_np)
        
        # --- PLOT 1: NC2 (Integration) - Task Accuracy ---
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('NC2 (Unitary Integration): Coupled Map Lattice Prediction', fontsize=16)
        
        im0 = axs[0, 0].imshow(Y_test_np, aspect='auto', cmap='viridis')
        axs[0, 0].set_title('Ground Truth (Target State)')
        axs[0, 0].set_ylabel('Sample Index')
        
        im1 = axs[0, 1].imshow(umc_pred_np, aspect='auto', cmap='viridis')
        axs[0, 1].set_title('UMC_Cell Prediction')
        
        im2 = axs[0, 2].imshow(base_pred_np, aspect='auto', cmap='viridis')
        axs[0, 2].set_title('Baseline RNN Prediction')
        
        # Errors
        vmax_err = max(umc_error.max(), base_error.max())
        
        im3 = axs[1, 1].imshow(umc_error, aspect='auto', cmap='magma', vmin=0, vmax=vmax_err)
        axs[1, 1].set_title(f'UMC Error (Mean: {umc_error.mean():.4f})')
        fig.colorbar(im3, ax=axs[1, 1])
        
        im4 = axs[1, 2].imshow(base_error, aspect='auto', cmap='magma', vmin=0, vmax=vmax_err)
        axs[1, 2].set_title(f'Baseline Error (Mean: {base_error.mean():.4f})')
        fig.colorbar(im4, ax=axs[1, 2])
        
        axs[1, 0].axis('off') # Hide empty subplot
        plt.tight_layout()
        plt.savefig('visualization_nc2_integration.png', dpi=150)
        plt.close()
        
        # --- PLOT 2: NC4 (Contractive Stability) - The Recursive Self ---
        print("\n--- Testing Zero-Input Convergence (NC4) ---")
        steps = 50
        
        # UMC Trajectory
        h_umc = umc_model.cell.init_hidden(1, device)
        zero_input = torch.zeros(1, input_dim).to(device)
        umc_traj = []
        for _ in range(steps):
            h_umc, macro = umc_model.cell(zero_input, h_umc)
            umc_traj.append(macro.norm().item())
            
        # Baseline Trajectory
        h_base = torch.zeros(1, 1, int(hidden_dim * 2.633)).to(device) # (num_layers, batch, hidden_size)
        base_traj = []
        for _ in range(steps):
            # RNN takes (batch, seq_len, input_size)
            out, h_base = baseline_model.rnn(zero_input.unsqueeze(1), h_base)
            base_traj.append(h_base.norm().item())
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(umc_traj, label='UMC_Cell (Spectral Norm + KM Iteration)', linewidth=2.5, color='blue')
        ax.plot(base_traj, label='Standard RNN (Unconstrained)', linewidth=2.5, color='red', alpha=0.7)
        
        ax.set_title('NC4 (Fixed-Point Stability): "The Recursive Self" Formation\nNorm of internal state under sensory deprivation (zero input)', fontsize=14)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('L2 Norm of Internal State', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Annotations
        ax.annotate('Perfect Stable Attractor\n(The Observer)', xy=(40, umc_traj[-1]), xytext=(30, umc_traj[-1] + (max(base_traj)-min(base_traj))*0.2),
                    arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=11, color='blue')
        
        if abs(base_traj[-1] - base_traj[-2]) > 0.01:
            ax.annotate('Limit Cycle / Oscillation\n(No Stable Self)', xy=(40, base_traj[-1]), xytext=(25, base_traj[-1] - (max(base_traj)-min(base_traj))*0.2),
                        arrowprops=dict(facecolor='red', shrink=0.05), fontsize=11, color='red')
                        
        plt.tight_layout()
        plt.savefig('visualization_nc4_stability.png', dpi=150)
        plt.close()
        
        print("Visualizations saved as 'visualization_nc2_integration.png' and 'visualization_nc4_stability.png'")

if __name__ == "__main__":
    visualize()

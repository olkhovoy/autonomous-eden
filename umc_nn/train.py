import torch
import torch.nn as nn
import torch.optim as optim
from core import UMC_Cell
from tqdm import tqdm

def generate_synthetic_data(num_samples, seq_len, input_dim):
    # Task: Coupled Map Lattice (CML) - Chaotic and structurally irreducible (NC2)
    # Each dimension's next state depends non-linearly on itself and its neighbors.
    # To predict the final state, the network MUST integrate information 
    # across dimensions over time. It cannot be factorized into independent parts.
    
    # Initialize random starting states in [-1, 1]
    X = torch.rand(num_samples, seq_len + 1, input_dim) * 2 - 1
    
    s = 0.4 # Coupling constant
    for t in range(seq_len):
        x_t = X[:, t, :]
        # Non-linear local dynamics: Logistic map f(x) = 1 - 2x^2
        f_x = 1 - 2 * x_t**2
        
        # Coupled spatial dynamics (periodic boundary conditions)
        f_x_shifted_left = torch.roll(f_x, shifts=1, dims=1)
        f_x_shifted_right = torch.roll(f_x, shifts=-1, dims=1)
        
        X[:, t+1, :] = (1 - s) * f_x + (s / 2) * (f_x_shifted_left + f_x_shifted_right)
        
    # Input is the sequence from t=0 to seq_len-1
    # Target is the final state at t=seq_len
    inputs = X[:, :-1, :]
    targets = X[:, -1, :] 
    
    return inputs, targets

class UMC_Network(nn.Module):
    def __init__(self, input_dim, num_nodes, hidden_dim, output_dim, k_contractive=0.95):
        super().__init__()
        self.cell = UMC_Cell(input_dim, num_nodes, hidden_dim, k_contractive)
        self.readout = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x_seq):
        # x_seq: (B, T, D_in)
        batch_size, seq_len, _ = x_seq.size()
        h = self.cell.init_hidden(batch_size, x_seq.device)
        
        for t in range(seq_len):
            h, macro_state = self.cell(x_seq[:, t, :], h)
            
        # Readout from the final macro state (which encodes the whole sequence via Downward Causation)
        out = self.readout(macro_state)
        return out, h

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    input_dim = 32
    num_nodes = 16
    hidden_dim = 128
    output_dim = 32
    seq_len = 15
    batch_size = 256
    epochs = 200
    
    model = UMC_Network(input_dim, num_nodes, hidden_dim, output_dim, k_contractive=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Generate data
    X_train, Y_train = generate_synthetic_data(5000, seq_len, input_dim)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    with tqdm(total=epochs, desc="Training UMC Network") as pbar:
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs, final_h = model(x_batch)
                
                # Loss
                loss = criterion(outputs, y_batch)
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients early on
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
            pbar.update(1)
            
    print("Training complete.")
    
    # Test Fixed-Point Stability (NC4)
    model.eval()
    with torch.no_grad():
        print("\nTesting Fixed-Point Convergence (The Recursive Self)...")
        # Provide zero input for many steps to see if it converges
        h = model.cell.init_hidden(1, device)
        zero_input = torch.zeros(1, input_dim).to(device)
        
        trajectory = []
        for _ in range(50):
            h, macro = model.cell(zero_input, h)
            trajectory.append(macro.norm().item())
            
        print("Macro state norms over 50 steps with zero input:")
        print([f"{val:.4f}" for val in trajectory[:10]], "...", [f"{val:.4f}" for val in trajectory[-5:]])
        
        diff = abs(trajectory[-1] - trajectory[-2]) if len(trajectory)>1 else 0
        if diff < 1e-4:
            print("=> FIXED POINT REACHED: The system exhibits IsContractiveStable (NC4) property.")
        else:
            print("=> Fixed point not fully reached, but difference is:", diff)

if __name__ == "__main__":
    train()

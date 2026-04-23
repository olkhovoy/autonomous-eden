import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class UMCTradingCell(nn.Module):
    """
    Pure UMC_Cell applied to high-frequency financial markets.
    Strict adherence to NC1-NC4.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, alpha: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        
        # NC2: Unitary Integration - Global connections
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # NC4: Fixed-Point Stability - Spectral Normalization
        # Ensures Lipschitz constant is <= 1 for contractive stability
        self.W_hh = spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=True))
        
        self.activation = nn.Tanh() # Bounded activation to support stability
        
        # NC3: Downward Causation - Final projection from macro-state
        # 3 classes: 0 = Flat, 1 = Long, 2 = Short
        self.action_head = nn.Linear(hidden_dim, 3, bias=True)
        
        # Ensure no gradients are needed as we use evolutionary engine
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        NC1: Recursive Closure
        Forward pass takes (market_input, prev_hidden_state) 
        and returns (action_logits, next_hidden_state).
        """
        assert x.ndim == 2, "Input x must be 2D (batch_size, input_dim)"
        assert h_prev.ndim == 2, "Hidden state must be 2D (batch_size, hidden_dim)"
        
        # Combine input and previous hidden state
        x_proj = self.W_x(x)
        h_proj = self.W_hh(h_prev)
        
        # Target state
        h_target = self.activation(x_proj + h_proj)
        
        # Krasnoselskij-Mann (K-M) Iteration to guarantee convergence
        h_next = (1 - self.alpha) * h_prev + self.alpha * h_target
        
        # Downward causation: Output from the stabilized state
        action_logits = self.action_head(h_next)
        
        return action_logits, h_next
        
    def get_num_parameters(self) -> int:
        """Returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters())
        
    @torch.no_grad()
    def set_weights_from_vector(self, vector: np.ndarray):
        """
        Injects a flat 1D genome (from Rust) into the cell's state_dict.
        """
        num_params = self.get_num_parameters()
        assert len(vector) == num_params, f"Vector length {len(vector)} does not match parameter count {num_params}. Check genome generation."
        
        idx = 0
        for param in self.parameters():
            param_shape = param.shape
            param_size = param.numel()
            
            # Slice the vector
            param_data = vector[idx:idx+param_size]
            idx += param_size
            
            # Reshape and copy into the parameter
            param.copy_(torch.from_numpy(param_data).view(param_shape).float())

        # For spectral_norm to properly recompute u and v with new weights:
        # We perform a dummy forward pass of the linear layer to trigger power iteration.
        # Ensure the module is in train mode (where spectral_norm updates its vectors)
        is_train = self.training
        self.train()
        device = next(self.parameters()).device
        dummy_h = torch.zeros(1, self.hidden_dim, device=device)
        _ = self.W_hh(dummy_h)
        if not is_train:
            self.eval()

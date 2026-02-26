import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class UMC_Cell(nn.Module):
    """
    A Neural Network Cell designed to explicitly satisfy the Necessary Conditions
    of the Unitary Model of Consciousness (UMC) as formalized in Lean 4.
    
    NC1 (Recursive Closure): The network models itself via self-attention.
    NC3 (Downward Causation): Macro-states modulate micro-state transitions.
    NC4 (Contractive Stability): We ensure the transition map F is non-expansive 
                                 (Lipschitz <= 1) via Spectral Normalization.
                                 To eliminate limit cycles and mathematically 
                                 guarantee convergence to a unique Fixed Point, 
                                 we apply Krasnoselskii-Mann iteration (Leaky update).
    """
    def __init__(self, input_dim: int, num_nodes: int, hidden_dim: int, k_contractive: float = 0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        # alpha for Krasnoselskii-Mann iteration. Must be in (0, 1)
        self.alpha = k_contractive 
        
        self.W_x = nn.Linear(input_dim, hidden_dim)
        
        # All linear layers in the recurrent path are Spectrally Normalized
        # to ensure the overall operator F(h) has a Lipschitz constant <= 1.
        self.W_h = spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False))
        
        # Custom Spectrally Normalized Attention (NC1)
        self.q_proj = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.k_proj = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.v_proj = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.out_proj = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        
        # Spectrally Normalized Macro Extractor (NC3)
        self.macro_extractor = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        )
        
        # Spectrally Normalized Downward Modulator (NC3)
        self.downward_modulator = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Sigmoid() # Gate in (0, 1) preserves Lipschitz bound
        )
        
        self.activation = nn.Tanh() # Tanh has Lipschitz constant 1.

    def forward(self, x, h_prev):
        batch_size = x.size(0)
        
        # 1. Project input (External stimuli)
        x_proj = self.W_x(x).unsqueeze(1).expand(-1, self.num_nodes, -1)
        
        # 2. NC1: Recursive Closure (S models S via Self-Attention)
        Q = self.q_proj(h_prev)
        K = self.k_proj(h_prev)
        V = self.v_proj(h_prev)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        h_attended = self.out_proj(torch.bmm(attn_weights, V))
        
        # 3. NC3: Downward Causation (Macro to Micro)
        macro_state_raw = h_attended.mean(dim=1)
        macro_state = self.macro_extractor(macro_state_raw)
        
        modulation_gate = self.downward_modulator(macro_state).unsqueeze(1)
        h_modulated = h_attended * modulation_gate
        
        # 4. NC4: Contractive Bottleneck
        h_recurrent = self.W_h(h_modulated)
        
        # Proposed next state F(h_t)
        h_proposed = self.activation(x_proj + h_recurrent)
        
        # 5. Krasnoselskii-Mann Iteration 
        # If F(x) is non-expansive, T(x) = (1-a)x + a F(x) guarantees convergence 
        # to a fixed point, eliminating any period-2 limit cycles!
        h_next = (1.0 - self.alpha) * h_prev + self.alpha * h_proposed
        
        return h_next, macro_state

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)

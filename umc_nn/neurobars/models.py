import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    """
    Causal 1D convolution to ensure we don't look into the future.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=self.__padding, dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, x):
        result = super(CausalConv1d, self).forward(x)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, dilation=dilation)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        
        return out + residual

class NeurobarEncoder(nn.Module):
    """
    Temporal Convolutional Network (TCN) to compress a sequence of OHLCV bars 
    into a dense latent vector (The Neurobar).
    """
    def __init__(self, input_dim: int = 138, hidden_channels: int = 64, latent_dim: int = 32, num_layers: int = 4):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        
        # Build TCN with exponentially increasing dilation
        for i in range(num_layers):
            dilation_size = 2 ** i
            layers.append(ResidualBlock(in_channels, hidden_channels, dilation=dilation_size))
            in_channels = hidden_channels
            
        self.tcn = nn.Sequential(*layers)
        
        # Projection to latent space
        self.latent_proj = nn.Linear(hidden_channels, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (Batch, Seq_Len, Features)
        Returns: (Batch, Latent_Dim) - The Neurobar representing the sequence context.
        """
        # TCN expects (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)
        
        # Pass through TCN
        features = self.tcn(x)
        
        # Take the features at the last time step
        last_step_features = features[:, :, -1]
        
        # Project to latent space
        latent_vector = self.latent_proj(last_step_features)
        
        # Normalize the latent vector (important for UMC_Cell stability)
        latent_vector = F.layer_norm(latent_vector, latent_vector.size()[1:])
        
        return latent_vector

class NeurobarAutoencoder(nn.Module):
    """
    Wrapper for unsupervised training. 
    Predicts the NEXT bar (or reconstructs the sequence) from the latent vector.
    """
    def __init__(self, encoder: NeurobarEncoder, input_dim: int = 138):
        super().__init__()
        self.encoder = encoder
        # Simple decoder to predict the next bar features
        self.decoder = nn.Sequential(
            nn.Linear(encoder.latent_proj.out_features, 64),
            nn.GELU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        next_bar_pred = self.decoder(latent)
        return latent, next_bar_pred

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    market_dim: int = 5
    self_dim: int = 6
    action_dim: int = 5
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512
    use_flash: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_flash: bool) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash = use_flash
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = float(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dropout_p = self.attn_dropout if self.training else 0.0
        if self.use_flash and x.is_cuda:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attn = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=dropout_p, is_causal=True
                    )
            except Exception:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=True,
                ):
                    attn = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=dropout_p, is_causal=True
                    )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.resid_dropout(self.proj(attn))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            config.d_model, config.n_heads, config.dropout, config.use_flash
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class UnitaryTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.market_dim + config.self_dim, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.action_head = nn.Linear(config.d_model, config.action_dim)
        self.self_head = nn.Linear(
            config.d_model, config.action_dim * config.self_dim
        )
        self.price_head = nn.Linear(config.d_model, config.market_dim)

    def forward(
        self,
        market_input: torch.Tensor,
        prev_self_state: torch.Tensor | None,
        *,
        return_price: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if market_input.ndim != 3:
            raise ValueError("market_input must be (batch, seq_len, market_dim).")
        batch, seq_len, _ = market_input.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("seq_len exceeds max_seq_len in ModelConfig.")
        if prev_self_state is None:
            prev_self_state = market_input.new_zeros(batch, self.config.self_dim)
        self_seq = prev_self_state.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([market_input, self_seq], dim=-1)
        x = self.input_proj(x)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_emb(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last = x[:, -1, :]
        action_logits = self.action_head(last)
        new_self_state = self.self_head(last).view(
            batch, self.config.action_dim, self.config.self_dim
        )
        if return_price:
            price_pred = self.price_head(last)
            return action_logits, new_self_state, price_pred
        return action_logits, new_self_state

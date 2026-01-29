"""
Temporal Transformer for self-attention over latent states
Learns temporal dependencies and long-range interactions
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for self-attention over latent states
    Learns temporal dependencies and delays in brain dynamics
    """
    
    def __init__(
        self,
        d_latent: int,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            d_latent: Latent dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network (default: 4*d_latent)
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        self.d_latent = d_latent
        
        if dim_feedforward is None:
            dim_feedforward = 4 * d_latent
        
        self.pos_encoding = PositionalEncoding(d_latent, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.output_proj = nn.Linear(d_latent, d_latent)
        self.norm = nn.LayerNorm(d_latent)
        
    def forward(self, z_sequence: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_sequence: [B, T, d_latent] - Sequence of latent states
            mask: [T, T] - Optional attention mask (for causal attention)
        Returns:
            z_context: [B, T, d_latent] - Context-enriched latent states
        """
        z_encoded = self.pos_encoding(z_sequence)
        
        z_context = self.transformer(z_encoded, mask=mask)
        
        z_context = self.output_proj(z_context)
        z_context = self.norm(z_context)
        
        return z_context
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive modeling
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        Returns:
            mask: [seq_len, seq_len] - Upper triangular mask
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


if __name__ == "__main__":
    # Test module
    B, T, d_latent = 4, 100, 128
    
    # Create transformer
    transformer = TemporalTransformer(
        d_latent=d_latent,
        n_heads=8,
        n_layers=3
    )
    
    # Test forward pass
    z = torch.randn(B, T, d_latent)
    z_out = transformer(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {z_out.shape}")
    
    # Test with causal mask
    mask = transformer.create_causal_mask(T, z.device)
    z_causal = transformer(z, mask=mask)
    print(f"Causal output shape: {z_causal.shape}")
    
    print(f"Parameters: {sum(p.numel() for p in transformer.parameters()):,}")


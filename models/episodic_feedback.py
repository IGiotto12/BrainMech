"""
Episodic Feedback Transformer
Learns history-dependent modulation from recent events
"""

import torch
import torch.nn as nn


class EpisodicFeedbackTransformer(nn.Module):
    """
    Episodic Feedback Mini-Transformer
    Learns episodic memory features from recent brain states
    Used to condition control signals on recent history
    """
    
    def __init__(
        self,
        d_latent: int,
        d_episodic: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            d_latent: Latent dimension
            d_episodic: Episodic memory dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        self.d_latent = d_latent
        self.d_episodic = d_episodic
        
        # Input projection
        self.input_proj = nn.Linear(d_latent, d_episodic)
        
        # Mini-transformer for episodic processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_episodic,
            nhead=n_heads,
            dim_feedforward=2 * d_episodic,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Episodic summary projection
        self.summary_proj = nn.Sequential(
            nn.Linear(d_episodic, d_episodic),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_sequence: [B, T, d_latent] - Sequence of latent states
        Returns:
            episodic_summary: [B, d_episodic] - Episodic memory features
        """
        # Project to episodic space
        z_episodic = self.input_proj(z_sequence)  # [B, T, d_episodic]
        
        # Apply transformer to capture episodic patterns
        z_processed = self.transformer(z_episodic)  # [B, T, d_episodic]
        
        # Global average pooling to get episodic summary
        episodic_summary = torch.mean(z_processed, dim=1)  # [B, d_episodic]
        
        # Final projection
        episodic_summary = self.summary_proj(episodic_summary)
        
        return episodic_summary


if __name__ == "__main__":
    # Test module
    B, T, d_latent = 4, 100, 128
    d_episodic = 64
    
    # Create module
    episodic = EpisodicFeedbackTransformer(
        d_latent=d_latent,
        d_episodic=d_episodic,
        n_heads=4,
        n_layers=2
    )
    
    # Test forward pass
    z = torch.randn(B, T, d_latent)
    summary = episodic(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Episodic summary shape: {summary.shape}")
    print(f"Parameters: {sum(p.numel() for p in episodic.parameters()):,}")


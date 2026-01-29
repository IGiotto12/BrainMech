"""
Control Fusion Module
Fuses episodic summary and task context to generate control signals
"""

import torch
import torch.nn as nn


class ControlFusion(nn.Module):
    """
    Fusion module for episodic summary + task context → control signal
    Generates bounded control signals for Koopman operator
    """
    
    def __init__(
        self,
        d_episodic: int,
        d_context: int,
        d_control: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_episodic: Episodic memory dimension
            d_context: Task context dimension
            d_control: Control signal dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_control = d_control
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(d_episodic + d_context, d_control * 2),
            nn.GELU(),
            nn.LayerNorm(d_control * 2),
            nn.Dropout(dropout),
            nn.Linear(d_control * 2, d_control),
            nn.Tanh()  # Bounded control signal in [-1, 1]
        )
        
    def forward(
        self,
        episodic_summary: torch.Tensor,
        context_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            episodic_summary: [B, d_episodic] - Episodic memory features
            context_embedding: [B, T, d_context] - Task context embeddings
        Returns:
            control_signal: [B, T, d_control] - Control signal c_t
        """
        B, T, d_context = context_embedding.shape
        
        # Expand episodic summary to match temporal dimension
        episodic_expanded = episodic_summary.unsqueeze(1).expand(B, T, -1)
        
        # Concatenate episodic memory and task context
        combined = torch.cat([episodic_expanded, context_embedding], dim=-1)
        
        # Generate control signals
        control_signal = self.fusion(combined)
        
        return control_signal


if __name__ == "__main__":
    # Test module
    B, T = 4, 100
    d_episodic, d_context, d_control = 64, 64, 32
    
    # Create module
    control_fusion = ControlFusion(
        d_episodic=d_episodic,
        d_context=d_context,
        d_control=d_control
    )
    
    # Test forward pass
    episodic = torch.randn(B, d_episodic)
    context = torch.randn(B, T, d_context)
    control = control_fusion(episodic, context)
    
    print(f"Episodic shape: {episodic.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Control signal shape: {control.shape}")
    print(f"Control range: [{control.min():.2f}, {control.max():.2f}]")
    print(f"Parameters: {sum(p.numel() for p in control_fusion.parameters()):,}")


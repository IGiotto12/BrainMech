"""
Cognitive State Readout for task labels and clinical scores
"""

import torch
import torch.nn as nn


class CognitiveStateReadout(nn.Module):
    """
    Cognitive State Readout for task labels, clinical scores, and predictions
    Provides flexible output head for various downstream tasks
    """
    
    def __init__(
        self,
        d_latent: int,
        n_classes: int,
        task_type: str = 'classification',
        dropout: float = 0.1
    ):
        """
        Args:
            d_latent: Latent state dimension
            n_classes: Number of output classes/dimensions
            task_type: 'classification' or 'regression'
            dropout: Dropout rate
        """
        super().__init__()
        self.task_type = task_type
        self.n_classes = n_classes
        
        # Readout layers
        self.readout = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.LayerNorm(d_latent // 2),
            nn.Dropout(dropout),
            nn.Linear(d_latent // 2, n_classes)
        )
        
        # Output activation based on task type
        if task_type == 'classification' and n_classes > 1:
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, z: torch.Tensor, pool: bool = True) -> torch.Tensor:
        """
        Args:
            z: [B, T, d_latent] or [B, d_latent] - Latent states
            pool: Whether to pool temporal dimension (if present)
        Returns:
            predictions: [B, n_classes] - Task predictions
        """
        # Handle temporal input
        if z.dim() == 3 and pool:  # [B, T, d_latent]
            z = torch.mean(z, dim=1)  # Global average pooling -> [B, d_latent]
        
        # Apply readout layers
        predictions = self.readout(z)
        predictions = self.output_activation(predictions)
        
        return predictions


if __name__ == "__main__":
    # Test module
    B, T, d_latent = 4, 100, 128
    
    # Test classification
    n_classes = 4
    readout_cls = CognitiveStateReadout(
        d_latent=d_latent,
        n_classes=n_classes,
        task_type='classification'
    )
    
    z = torch.randn(B, T, d_latent)
    pred_cls = readout_cls(z)
    print(f"Classification input: {z.shape}")
    print(f"Classification output: {pred_cls.shape}")
    print(f"Softmax check (sum): {pred_cls[0].sum():.4f}")
    
    # Test regression
    readout_reg = CognitiveStateReadout(
        d_latent=d_latent,
        n_classes=1,
        task_type='regression'
    )
    pred_reg = readout_reg(z)
    print(f"\nRegression output: {pred_reg.shape}")
    
    print(f"\nParameters: {sum(p.numel() for p in readout_cls.parameters()):,}")


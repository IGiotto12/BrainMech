"""
Regularizers for Koopman operator
Includes spectral and graph prior regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralRegularizer(nn.Module):
    """
    Spectral Regularization for Koopman Matrix
    Penalizes spectral radius exceeding target for stability
    """
    
    def __init__(self, spectral_radius: float = 0.95, weight: float = 1.0):
        """
        Args:
            spectral_radius: Target spectral radius
            weight: Regularization weight
        """
        super().__init__()
        self.spectral_radius = spectral_radius
        self.weight = weight
    
    def forward(self, K_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            K_matrix: [d, d] - Koopman matrix
        Returns:
            spectral_loss: scalar - Spectral regularization loss
        """
        try:
            eigenvals, _ = torch.linalg.eig(K_matrix)
            max_eigenval = torch.max(torch.abs(eigenvals))
        except:
            max_eigenval = torch.norm(K_matrix, p=2)
        
        spectral_loss = F.relu(max_eigenval - self.spectral_radius) ** 2
        
        return self.weight * spectral_loss


class GraphPriorRegularizer(nn.Module):
    """
    Graph Prior Regularization
    Aligns Koopman matrix structure with brain connectome topology
    """
    
    def __init__(
        self,
        fc_matrix: torch.Tensor,
        weight: float = 0.1,
        use_structure_only: bool = True
    ):
        """
        Args:
            fc_matrix: [N_ROI, N_ROI] - Functional connectivity matrix
            weight: Regularization weight
            use_structure_only: If True, only match sparsity pattern
        """
        super().__init__()
        self.register_buffer('fc_matrix', fc_matrix)
        self.weight = weight
        self.use_structure_only = use_structure_only
        
        if use_structure_only:
            threshold = torch.quantile(torch.abs(fc_matrix), 0.5)
            self.register_buffer('fc_mask', (torch.abs(fc_matrix) > threshold).float())
    
    def forward(self, K_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            K_matrix: [d_latent, d_latent] - Koopman matrix
        Returns:
            graph_loss: scalar - Graph prior loss
        """
        if self.use_structure_only and hasattr(self, 'fc_mask'):
            # Match sparsity structure
            if K_matrix.shape != self.fc_mask.shape:
                graph_loss = torch.norm(K_matrix, p='fro') * 0.01
            else:
                graph_loss = torch.sum((K_matrix ** 2) * (1 - self.fc_mask))
        else:
            if K_matrix.shape == self.fc_matrix.shape:
                graph_loss = F.mse_loss(K_matrix, self.fc_matrix)
            else:
                graph_loss = torch.norm(K_matrix, p='fro') * 0.01
        
        return self.weight * graph_loss


class ConsistencyRegularizer(nn.Module):
    """
    Consistency Regularization between Predicted and True Next States
    Ensures Koopman dynamics match the actual latent dynamics
    """
    
    def __init__(self, weight: float = 0.5):
        """
        Args:
            weight: Regularization weight
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_pred: [B, T, d] - Predicted latent states
            z_true: [B, T, d] - True latent states
        Returns:
            consistency_loss: scalar
        """
        consistency_loss = F.mse_loss(z_pred, z_true)
        return self.weight * consistency_loss


if __name__ == "__main__":
    # Test Spectral Regularizer
    d_latent = 128
    K = torch.randn(d_latent, d_latent) * 0.1
    
    spectral_reg = SpectralRegularizer(spectral_radius=0.95, weight=1.0)
    spectral_loss = spectral_reg(K)
    print(f"Spectral loss: {spectral_loss.item():.6f}")
    
    # Test Graph Prior Regularizer
    n_roi = 116
    fc_matrix = torch.randn(n_roi, n_roi)
    fc_matrix = (fc_matrix + fc_matrix.T) / 2
    
    graph_reg = GraphPriorRegularizer(fc_matrix, weight=0.1)
    K_small = torch.randn(n_roi, n_roi) * 0.1
    graph_loss = graph_reg(K_small)
    print(f"Graph prior loss: {graph_loss.item():.6f}")
    
    # Test Consistency Regularizer
    B, T, d = 4, 100, 128
    z_pred = torch.randn(B, T, d)
    z_true = torch.randn(B, T, d)
    
    consistency_reg = ConsistencyRegularizer(weight=0.5)
    consistency_loss = consistency_reg(z_pred, z_true)
    print(f"Consistency loss: {consistency_loss.item():.6f}")


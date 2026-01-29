"""
Module Probe Heads for Monitoring Module-Level Losses

These probe heads are used to monitor the quality of intermediate representations
from each module without affecting the main training objective.
All probe losses use detached features to ensure they don't backpropagate to the backbone.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ModuleProbeHeads(nn.Module):
    """
    Probe heads for monitoring module-level reconstruction/forecast losses.
    
    Each probe head is a simple linear layer that maps intermediate representations
    back to ROI space for comparison with ground truth.
    """
    
    def __init__(
        self,
        n_roi: int = 116,
        d_latent: int = 384,
        d_control: int = 192,
        d_episodic: int = 192,
        d_context: int = 192,
        use_episodic_feedback: bool = True,
        use_task_context: bool = True,
        use_control_fusion: bool = True,
    ):
        """
        Args:
            n_roi: Number of ROIs
            d_latent: Latent dimension
            d_control: Control dimension
            d_episodic: Episodic memory dimension
            d_context: Context dimension
            use_episodic_feedback: Whether episodic feedback is enabled
            use_task_context: Whether task context is enabled
            use_control_fusion: Whether control fusion is enabled
        """
        super().__init__()
        
        self.n_roi = n_roi
        self.d_latent = d_latent
        self.d_control = d_control
        self.d_episodic = d_episodic
        self.d_context = d_context
        
        # Probe heads for latent representations (all map to ROI space)
        self.head_spatial = nn.Linear(d_latent, n_roi)
        self.head_temporal = nn.Linear(d_latent, n_roi)
        self.head_fused = nn.Linear(d_latent, n_roi)
        self.head_koopman = nn.Linear(d_latent, n_roi)
        
        # Probe heads for control-related representations
        if use_control_fusion:
            self.head_control = nn.Linear(d_control, n_roi)
        else:
            self.head_control = None
            
        if use_task_context:
            self.head_context = nn.Linear(d_context, n_roi)
        else:
            self.head_context = None
            
        if use_episodic_feedback:
            self.head_episodic = nn.Linear(d_episodic, n_roi)
        else:
            self.head_episodic = None
        
        # Initialize probe heads with small weights
        for head in [self.head_spatial, self.head_temporal, self.head_fused, self.head_koopman]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
            
        if self.head_control is not None:
            nn.init.xavier_uniform_(self.head_control.weight, gain=0.1)
            nn.init.zeros_(self.head_control.bias)
            
        if self.head_context is not None:
            nn.init.xavier_uniform_(self.head_context.weight, gain=0.1)
            nn.init.zeros_(self.head_context.bias)
            
        if self.head_episodic is not None:
            nn.init.xavier_uniform_(self.head_episodic.weight, gain=0.1)
            nn.init.zeros_(self.head_episodic.bias)
    
    def forward(
        self,
        intermediates: Dict[str, torch.Tensor],
        x_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_forecast: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute probe losses for all modules.
        
        Args:
            intermediates: Dictionary of intermediate features from model forward
            x_target: [B, T, N_ROI] Target ROI signals
            mask: [B, T, 1] or [B, T, N_ROI] Mask for reconstruction (1 = masked, 0 = visible)
            mask_forecast: [B, T-1, 1] or [B, T-1, N_ROI] Mask for forecast
            
        Returns:
            Dictionary of probe losses (all detached, ready for logging)
        """
        losses = {}
        
        # Spatial encoder probe
        if 'z_spatial' in intermediates:
            z_spatial = intermediates['z_spatial'].detach()  # [B, T, d_latent]
            pred_spatial = self.head_spatial(z_spatial)  # [B, T, N_ROI]
            losses['spatial_recon'] = pred_spatial
        
        # Temporal transformer probe
        if 'z_temporal' in intermediates:
            z_temporal = intermediates['z_temporal'].detach()  # [B, T, d_latent]
            pred_temporal = self.head_temporal(z_temporal)  # [B, T, N_ROI]
            losses['temporal_recon'] = pred_temporal
        
        # Fused representation probe (after temporal, before koopman)
        if 'z_fused' in intermediates:
            z_fused = intermediates['z_fused'].detach()  # [B, T, d_latent]
            pred_fused = self.head_fused(z_fused)  # [B, T, N_ROI]
            losses['fused_recon'] = pred_fused
        
        # Control signal probe
        if 'control' in intermediates and self.head_control is not None:
            control = intermediates['control'].detach()  # [B, T, d_control]
            pred_control = self.head_control(control)  # [B, T, N_ROI]
            losses['control_recon'] = pred_control
        
        # Context embedding probe
        if 'context' in intermediates and self.head_context is not None:
            context = intermediates['context'].detach()  # [B, T, d_context]
            pred_context = self.head_context(context)  # [B, T, N_ROI]
            losses['context_recon'] = pred_context
        
        # Episodic memory probe (maps to time-averaged ROI)
        if 'episodic' in intermediates and self.head_episodic is not None:
            episodic = intermediates['episodic'].detach()  # [B, d_episodic]
            pred_episodic = self.head_episodic(episodic)  # [B, N_ROI]
            # Target is time-averaged x
            x_mean = x_target.mean(dim=1)  # [B, N_ROI]
            losses['episodic_recon'] = pred_episodic
            losses['episodic_target'] = x_mean  # Store for loss computation
        
        # Koopman forecast probe
        if 'z_koopman' in intermediates:
            z_koopman = intermediates['z_koopman'].detach()  # [B, T, d_latent]
            pred_koopman = self.head_koopman(z_koopman)  # [B, T, N_ROI]
            # For forecast: use z_koopman[:-1] to predict x[1:]
            pred_forecast = pred_koopman[:, :-1, :]  # [B, T-1, N_ROI]
            losses['koopman_forecast'] = pred_forecast
        
        return losses


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute masked MSE loss.
    
    Args:
        pred: [B, T, N_ROI] or [B, T-1, N_ROI] Predicted values
        target: [B, T, N_ROI] or [B, T-1, N_ROI] Target values
        mask: [B, T, 1] or [B, T, N_ROI] or [B, T-1, 1] or [B, T-1, N_ROI]
              Mask indicating which tokens to include (1 = include, 0 = exclude)
              If None, include all tokens
        reduction: 'mean' or 'sum'
        
    Returns:
        Scalar loss value
    """
    # Ensure pred and target have same shape
    if pred.shape != target.shape:
        # Handle forecast case: pred is [B, T-1, N_ROI], target should be [B, T-1, N_ROI]
        if pred.shape[1] == target.shape[1] - 1:
            target = target[:, 1:, :]
        else:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    # Compute squared error
    se = (pred - target) ** 2  # [B, T, N_ROI] or [B, T-1, N_ROI]
    
    # Apply mask if provided
    if mask is not None:
        # Handle mask shape: [B, T, 1] -> broadcast to [B, T, N_ROI]
        if mask.dim() == 3 and mask.shape[2] == 1:
            mask = mask.expand(-1, -1, pred.shape[2])
        # Ensure mask matches pred shape (for forecast case)
        if mask.shape[1] != pred.shape[1]:
            if mask.shape[1] == pred.shape[1] + 1:
                mask = mask[:, 1:, :]  # For forecast: mask[:, 1:]
            else:
                raise ValueError(f"Mask shape mismatch: mask {mask.shape} vs pred {pred.shape}")
        
        # Mask: 1 = include, 0 = exclude
        se = se * mask
        num_valid = mask.sum()
    else:
        num_valid = se.numel()
    
    # Reduce
    if reduction == 'mean':
        loss = se.sum() / (num_valid + 1e-8)
    elif reduction == 'sum':
        loss = se.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    return loss


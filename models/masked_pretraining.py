"""
Masked Autoencoder (MAE) Pre-training for Koopman Model
Randomly masks ROIs or timepoints and predicts them
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MaskedAutoencoderWrapper(nn.Module):
    """
    Wrapper that adds MAE pre-training capability to Koopman model
    
    Two masking strategies:
    1. Spatial masking: Mask random ROIs
    2. Temporal masking: Mask random timepoints
    """
    
    def __init__(
        self,
        base_model,
        mask_ratio: float = 0.5,
        mask_strategy: str = 'random',
        mask_type: str = 'spatial',
        use_random_masking: bool = True,
        min_mask_ratio: float = 0.1,
        max_mask_ratio: float = 0.8,
        use_curriculum: bool = True,
        # For combined masking
        spatial_mask_ratio: float = None,  # If None, uses mask_ratio
        temporal_mask_ratio: float = None  # If None, uses mask_ratio
    ):
        """
        Args:
            base_model: KoopmanBrainModel to wrap
            mask_ratio: Fixed ratio of input to mask (0.0-1.0) - used if use_random_masking=False
            mask_strategy: 'random', 'block', or 'structured'
            mask_type: 'spatial' (mask ROIs), 'temporal' (mask timepoints), or 'both' (mask both)
            use_random_masking: If True, use random masking ratio instead of fixed
            min_mask_ratio: Minimum masking ratio for random masking
            max_mask_ratio: Maximum masking ratio for random masking
            use_curriculum: If True, use curriculum learning (start easy, end hard)
            spatial_mask_ratio: Masking ratio for spatial dimension (only used when mask_type='both')
            temporal_mask_ratio: Masking ratio for temporal dimension (only used when mask_type='both')
        """
        super().__init__()
        self.base_model = base_model
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.mask_type = mask_type
        self.use_random_masking = use_random_masking
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.use_curriculum = use_curriculum
        self.current_epoch = 0
        
        # For combined masking: separate ratios for spatial and temporal
        self.spatial_mask_ratio = spatial_mask_ratio if spatial_mask_ratio is not None else mask_ratio
        self.temporal_mask_ratio = temporal_mask_ratio if temporal_mask_ratio is not None else mask_ratio
    
    def get_adaptive_mask_ratio(self, epoch: int = None, max_epochs: int = None) -> float:
        """
        Get adaptive masking ratio based on current epoch and strategy
        
        Args:
            epoch: Current epoch (for curriculum learning)
            max_epochs: Total number of epochs
        Returns:
            mask_ratio: Current masking ratio to use
        """
        if not self.use_random_masking:
            return self.mask_ratio
        
        if self.use_curriculum and epoch is not None and max_epochs is not None:
            # Curriculum learning: start easy, end hard
            progress = min(epoch / max_epochs, 1.0)
            mask_ratio = self.min_mask_ratio + (self.max_mask_ratio - self.min_mask_ratio) * progress
        else:
            # Random masking: uniform distribution between min and max
            mask_ratio = torch.rand(1).item() * (self.max_mask_ratio - self.min_mask_ratio) + self.min_mask_ratio
        
        return mask_ratio
    
    def create_random_mask(
        self, 
        shape: Tuple[int, ...], 
        epoch: int = None, 
        max_epochs: int = None,
        custom_ratio: float = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Create random mask tensor with adaptive masking ratio
        
        Args:
            shape: Shape of input [B, T, N_ROI] or [B, T] or [B, N_ROI]
            epoch: Current epoch (for curriculum learning)
            max_epochs: Total number of epochs
            custom_ratio: Override mask ratio (used for combined masking)
        Returns:
            mask: Binary mask (1 = keep, 0 = mask)
            mask_ratio: Actual masking ratio used
        """
        if custom_ratio is not None:
            # Use custom ratio (for combined masking)
            if self.use_random_masking:
                if self.use_curriculum and epoch is not None and max_epochs is not None:
                    progress = min(epoch / max_epochs, 1.0)
                    mask_ratio = self.min_mask_ratio + (custom_ratio - self.min_mask_ratio) * progress
                else:
                    mask_ratio = torch.rand(1).item() * (custom_ratio - self.min_mask_ratio) + self.min_mask_ratio
            else:
                mask_ratio = custom_ratio
        else:
            mask_ratio = self.get_adaptive_mask_ratio(epoch, max_epochs)
        
        mask = torch.rand(shape) > mask_ratio
        return mask.float(), mask_ratio
    
    def create_block_mask(
        self, 
        B: int, 
        T: int, 
        N: int, 
        epoch: int = None, 
        max_epochs: int = None,
        mask_dim: str = None,
        custom_ratio: float = None
    ) -> torch.Tensor:
        """
        Create block/patch-based mask (like vision MAE) with adaptive masking ratio
        
        Args:
            B, T, N: Batch, Time, ROI dimensions
            epoch: Current epoch (for curriculum learning)
            max_epochs: Total number of epochs
            mask_dim: 'spatial' or 'temporal' (overrides self.mask_type for combined masking)
            custom_ratio: Override mask ratio (used for combined masking)
        Returns:
            mask: [B, T, N] binary mask
        """
        if custom_ratio is not None:
            if self.use_random_masking:
                if self.use_curriculum and epoch is not None and max_epochs is not None:
                    progress = min(epoch / max_epochs, 1.0)
                    mask_ratio = self.min_mask_ratio + (custom_ratio - self.min_mask_ratio) * progress
                else:
                    mask_ratio = torch.rand(1).item() * (custom_ratio - self.min_mask_ratio) + self.min_mask_ratio
            else:
                mask_ratio = custom_ratio
        else:
            mask_ratio = self.get_adaptive_mask_ratio(epoch, max_epochs)
        
        # Use mask_dim if provided (for combined masking), else use self.mask_type
        current_mask_type = mask_dim if mask_dim is not None else self.mask_type
        
        if current_mask_type == 'temporal':
            # Mask contiguous time blocks
            block_size = max(5, int(T * 0.1))  # 10% of sequence
            num_blocks = int(T * mask_ratio / block_size)
            
            mask = torch.ones(B, T, N)
            for b in range(B):
                for _ in range(num_blocks):
                    start = np.random.randint(0, max(1, T - block_size))
                    end = min(start + block_size, T)
                    mask[b, start:end, :] = 0
        
        elif current_mask_type == 'spatial':
            # Mask groups of ROIs
            num_masked_rois = int(N * mask_ratio)
            mask = torch.ones(B, T, N)
            for b in range(B):
                if num_masked_rois > 0 and num_masked_rois < N:
                    masked_indices = np.random.choice(N, num_masked_rois, replace=False)
                    mask[b, :, masked_indices] = 0
        
        return mask
    
    def create_structured_mask(
        self, 
        B: int, 
        T: int, 
        N: int,
        mask_dim: str = None,
        custom_ratio: float = None
    ) -> torch.Tensor:
        """
        Create structured mask (e.g., mask entire brain networks or time blocks)
        
        Args:
            B, T, N: Batch, Time, ROI dimensions
            mask_dim: 'spatial' or 'temporal' (overrides self.mask_type for combined masking)
            custom_ratio: Override mask ratio (used for combined masking)
        Returns:
            mask: [B, T, N] binary mask
        """
        # Use mask_dim if provided (for combined masking), else use self.mask_type
        current_mask_type = mask_dim if mask_dim is not None else self.mask_type
        mask_ratio = custom_ratio if custom_ratio is not None else self.mask_ratio
        
        mask = torch.ones(B, T, N)
        
        if current_mask_type == 'spatial':
            # Mask groups of adjacent ROIs (representing networks)
            network_size = max(8, int(N * 0.1))  # ~10 ROIs per network
            num_networks = max(1, int(N * mask_ratio / network_size))
            
            for b in range(B):
                for _ in range(num_networks):
                    start = np.random.randint(0, max(1, N - network_size))
                    end = min(start + network_size, N)
                    mask[b, :, start:end] = 0
        
        elif current_mask_type == 'temporal':
            # Mask contiguous time blocks
            block_size = max(5, int(T * 0.1))  # ~10% of sequence
            num_blocks = max(1, int(T * mask_ratio / block_size))
            
            for b in range(B):
                for _ in range(num_blocks):
                    start = np.random.randint(0, max(1, T - block_size))
                    end = min(start + block_size, T)
                    mask[b, start:end, :] = 0
        
        return mask
    
    def apply_mask(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mask to input
        
        Args:
            x: [B, T, N_ROI] - Input
            mask: [B, T, N_ROI] or [B, T] or [B, N_ROI] - Binary mask
        Returns:
            x_masked: Masked input
            mask: Full mask
        """
        # Expand mask if needed
        if mask.dim() == 2 and x.dim() == 3:
            if mask.shape[1] == x.shape[1]:  # Temporal mask [B, T]
                mask = mask.unsqueeze(-1).expand_as(x)
            elif mask.shape[1] == x.shape[2]:  # Spatial mask [B, N_ROI]
                mask = mask.unsqueeze(1).expand_as(x)
        
        # Apply mask (set masked positions to 0)
        x_masked = x * mask
        
        return x_masked, mask
    
    def forward(
        self,
        x: torch.Tensor,
        task_vectors: Optional[torch.Tensor] = None,
        fc_matrix: Optional[torch.Tensor] = None,
        use_masking: bool = True,
        epoch: int = None,
        max_epochs: int = None,
        return_intermediates: bool = False,  # Support return_intermediates for shape debug
        **kwargs  # Accept extra kwargs but filter them out (e.g., 'age' from metadata)
    ) -> dict:
        """
        Forward with optional masking and adaptive masking ratio
        
        Args:
            x: [B, T, N_ROI] - Input
            task_vectors: Optional task vectors
            fc_matrix: [B, N_ROI, N_ROI] or [N_ROI, N_ROI] - Optional FC matrices
            use_masking: Whether to apply MAE masking
            epoch: Current epoch (for curriculum learning)
            max_epochs: Total number of epochs
        
        Returns:
            outputs: Dict with model outputs + mask info
        """
        B, T, N = x.shape
        mask_ratio = None
        
        if use_masking and self.training:
            # Create mask with adaptive ratio
            if self.mask_type == 'both':
                # Combined spatial and temporal masking
                if self.mask_strategy == 'random':
                    # Create spatial mask
                    mask_spatial, ratio_spatial = self.create_random_mask(
                        (B, N), epoch, max_epochs, custom_ratio=self.spatial_mask_ratio
                    )
                    mask_spatial = mask_spatial.to(x.device).unsqueeze(1).expand(B, T, N)
                    
                    # Create temporal mask
                    mask_temporal, ratio_temporal = self.create_random_mask(
                        (B, T), epoch, max_epochs, custom_ratio=self.temporal_mask_ratio
                    )
                    mask_temporal = mask_temporal.to(x.device).unsqueeze(-1).expand(B, T, N)
                    
                    # Combine: mask where EITHER spatial OR temporal mask is 0 (OR operation)
                    # In mask: 1 = keep, 0 = mask
                    # So combined mask = spatial_mask AND temporal_mask (both must be 1 to keep)
                    mask = mask_spatial * mask_temporal
                    mask_ratio = 1.0 - mask.mean().item()  # Actual masked ratio
                
                elif self.mask_strategy == 'block':
                    # Block masking: create separate spatial and temporal block masks
                    mask_spatial = self.create_block_mask(B, T, N, epoch, max_epochs, mask_dim='spatial', custom_ratio=self.spatial_mask_ratio)
                    mask_temporal = self.create_block_mask(B, T, N, epoch, max_epochs, mask_dim='temporal', custom_ratio=self.temporal_mask_ratio)
                    mask_spatial = mask_spatial.to(x.device)
                    mask_temporal = mask_temporal.to(x.device)
                    mask = mask_spatial * mask_temporal
                    mask_ratio = 1.0 - mask.mean().item()
                
                else:  # structured
                    # Structured: create both and combine
                    mask_spatial = self.create_structured_mask(B, T, N, mask_dim='spatial').to(x.device)
                    mask_temporal = self.create_structured_mask(B, T, N, mask_dim='temporal').to(x.device)
                    mask = mask_spatial * mask_temporal
                    mask_ratio = 1.0 - mask.mean().item()
            
            elif self.mask_strategy == 'random':
                if self.mask_type == 'spatial':
                    mask, mask_ratio = self.create_random_mask((B, N), epoch, max_epochs)
                    mask = mask.to(x.device)
                    mask = mask.unsqueeze(1).expand(B, T, N)
                else:  # temporal
                    mask, mask_ratio = self.create_random_mask((B, T), epoch, max_epochs)
                    mask = mask.to(x.device)
                    mask = mask.unsqueeze(-1).expand(B, T, N)
            
            elif self.mask_strategy == 'block':
                mask, mask_ratio = self.create_block_mask(B, T, N, epoch, max_epochs)
                mask = mask.to(x.device)
            
            elif self.mask_strategy == 'structured':
                mask = self.create_structured_mask(B, T, N).to(x.device)
                mask_ratio = self.get_adaptive_mask_ratio(epoch, max_epochs)
            
            # Apply mask
            x_masked, mask = self.apply_mask(x, mask)
        else:
            x_masked = x
            mask = torch.ones_like(x)
            mask_ratio = 0.0
        
        # Forward through base model
        outputs = self.base_model(
            x_masked, 
            task_vectors=task_vectors, 
            fc_matrix=fc_matrix,
            return_intermediates=return_intermediates  # Pass through for shape debug and probe losses
        )
        
        # Add mask information
        outputs['mask'] = mask
        outputs['x_original'] = x
        outputs['x_masked'] = x_masked
        outputs['mask_ratio'] = mask_ratio
        
        return outputs
    
    def compute_mae_loss(
        self,
        outputs: dict,
        use_masked_loss: bool = True
    ) -> dict:
        """
        Compute composite MSE+MAE loss
        
        Args:
            outputs: Model outputs (must include 'mask', 'x_original')
            use_masked_loss: If True, compute loss only on masked regions
        
        Returns:
            losses: Dict of loss components
        """
        x_original = outputs['x_original']
        x_recon = outputs['x_recon']
        mask = outputs['mask']
        
        if use_masked_loss and self.training:
            # Compute loss only on MASKED regions (like MAE)
            masked_regions = (1 - mask)  # Invert mask (1 where masked)
            
            # MSE on masked parts
            mse_masked = torch.sum(
                ((x_recon - x_original) ** 2) * masked_regions
            ) / (masked_regions.sum() + 1e-8)
            
            # MAE on masked parts
            mae_masked = torch.sum(
                torch.abs(x_recon - x_original) * masked_regions
            ) / (masked_regions.sum() + 1e-8)
            
            # Composite loss on masked regions (MSE + MAE)
            masked_recon_loss = mse_masked + mae_masked
            
            # MSE on unmasked parts
            mse_unmasked = torch.sum(
                ((x_recon - x_original) ** 2) * mask
            ) / (mask.sum() + 1e-8)
            
            # MAE on unmasked parts
            mae_unmasked = torch.sum(
                torch.abs(x_recon - x_original) * mask
            ) / (mask.sum() + 1e-8)
            
            # Composite loss on unmasked regions (MSE + MAE)
            unmasked_recon_loss = mse_unmasked + mae_unmasked
            
            # Combine (weighted: focus more on masked regions)
            recon_loss = masked_recon_loss * 2.0 + unmasked_recon_loss * 1.0
            recon_mse = mse_masked * 2.0 + mse_unmasked * 1.0
            recon_mae = mae_masked * 2.0 + mae_unmasked * 1.0
            
            # Raw metrics (unweighted, for consistent reporting)
            recon_mse_raw = torch.mean((x_recon - x_original) ** 2)
            recon_mae_raw = torch.mean(torch.abs(x_recon - x_original))
        else:
            # Standard composite loss (no masking) - MSE + MAE
            recon_mse = torch.mean((x_recon - x_original) ** 2)
            recon_mae = torch.mean(torch.abs(x_recon - x_original))
            recon_loss = recon_mse + recon_mae
            
            # Raw metrics (same as weighted in this case)
            recon_mse_raw = recon_mse
            recon_mae_raw = recon_mae
        
        # Forecast loss (composite MSE + MAE)
        x_forecast_target = x_original[:, 1:, :]
        forecast_mse = torch.mean((outputs['x_forecast'] - x_forecast_target) ** 2)
        forecast_mae = torch.mean(torch.abs(outputs['x_forecast'] - x_forecast_target))
        forecast_loss = forecast_mse + forecast_mae
        
        # Consistency loss
        z_true_next = outputs['z_latent'][:, 1:, :]
        z_pred_next = outputs['z_next'][:, :-1, :]
        consistency_loss = torch.mean((z_pred_next - z_true_next) ** 2) * 0.5
        
        # Spectral loss
        # Handle both V1 (KoopmanLinearCore with .K) and V2 (DPLRKoopmanCore with get_K_matrix())
        koopman_core = self.base_model.koopman_core
        if hasattr(koopman_core, 'K'):
            # V1 model: direct K attribute
            K_matrix = koopman_core.K
        elif hasattr(koopman_core, 'get_K_matrix'):
            # V2 model: DPLR decomposition, need to compute full matrix
            K_matrix = koopman_core.get_K_matrix()
        else:
            # Fallback: skip spectral loss if K matrix not accessible
            K_matrix = None
        
        if K_matrix is not None:
            spectral_loss = self.base_model.spectral_regularizer(K_matrix)
        else:
            # If spectral regularizer not available, use zero loss
            spectral_loss = torch.tensor(0.0, device=outputs['x_recon'].device)
        
        # Total
        total_loss = recon_loss + forecast_loss + consistency_loss + spectral_loss
        
        losses = {
            'recon_loss': recon_loss,
            'recon_mse': recon_mse,
            'recon_mae': recon_mae,
            'recon_mse_raw': recon_mse_raw.detach(),  # Raw metric (not in loss)
            'recon_mae_raw': recon_mae_raw.detach(),  # Raw metric (not in loss)
            'forecast_loss': forecast_loss,
            'forecast_mse': forecast_mse,
            'forecast_mae': forecast_mae,
            'consistency_loss': consistency_loss,
            'spectral_loss': spectral_loss,
            'total_loss': total_loss,
            'mask_ratio': mask.mean().item() if self.training else 1.0
        }
        
        return losses


if __name__ == "__main__":
    from koopman_brain_model_v2 import KoopmanBrainModel
    
    # Create base model
    base_model = KoopmanBrainModel(n_roi=116, d_latent=128)
    
    # Wrap with MAE
    mae_model = MaskedAutoencoderWrapper(
        base_model=base_model,
        mask_ratio=0.5,
        mask_strategy='random',
        mask_type='spatial'
    )
    
    # Test
    x = torch.randn(4, 100, 116)
    mae_model.train()
    
    outputs = mae_model(x, use_masking=True)
    print(f"Input shape: {x.shape}")
    print(f"Masked input shape: {outputs['x_masked'].shape}")
    print(f"Reconstruction shape: {outputs['x_recon'].shape}")
    print(f"Mask ratio: {outputs['mask'].mean():.2f}")
    
    # Compute MAE loss
    losses = mae_model.compute_mae_loss(outputs)
    print(f"\nLosses:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v:.4f}")


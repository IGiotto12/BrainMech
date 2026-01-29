"""
BrainMech Brain Model
Integrates spatiotemporal features, structured Koopman operators, and stochastic dynamics

Key Contributions:
1. Spatiotemporal Features
   - Graph spatial encoder
   - Temporal encoder
   - Episodic feedback
   - Task context encoder
   - Control fusion module
   - Koopman core
   - Stochastic decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .graph_spatial_encoder import GraphSpatialEncoder
from .temporal_encoder import TemporalTransformer
from .episodic_feedback import EpisodicFeedbackTransformer
from .task_context import TaskContextEncoder
from .control_fusion import ControlFusion
from .koopman_DPLR_core import DPLRKoopmanCore
from .stochastic_decoder import StochasticDecoder
from .regularizers import SpectralRegularizer, ConsistencyRegularizer


class BrainMechBrainModel(nn.Module):
    """
    BrainMech Brain Model with Spatiotemporal Features, Structured Koopman Operators, and Stochastic Dynamics
    
    Architecture Flow:
    1. Graph Spatial Encoder (ChebNet/GAT + DropEdge)
    2. Temporal Transformer (long-range dependencies)
    3. Episodic Feedback + Task Context → Control
    4. Structured Koopman SDE (structured + stochastic)
    5. Stochastic Decoder
    """
    
    def __init__(
        self,
        n_roi: int = 116,
        d_latent: int = 128,
        d_control: int = 64,
        d_episodic: int = 64,
        d_context: int = 64,
        d_task: int = 32,
        n_classes: int = 4,
        task_type: str = 'classification',
        # Graph spatial encoder params
        graph_type: str = 'cheb',
        n_graph_layers: int = 3,
        cheb_k: int = 3,
        drop_edge_p: float = 0.1,
        # Temporal encoder params
        n_temporal_layers: int = 3,
        n_heads: int = 8,
        # Structured Koopman operator params
        koopman_rank: int = 16,
        spectral_radius: float = 0.95,
        use_sde: bool = True,
        diffusion_mode: str = 'learned',
        init_noise_std: float = 0.01,
        # Decoder params
        uncertainty_mode: str = 'heteroscedastic',
        # Optional FC matrix
        fc_matrix: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
        # Control system flags
        use_episodic_feedback: bool = True,
        use_task_context: bool = True,
        use_control_fusion: bool = True,
        use_temporal_encoder: bool = True,
        use_spatial_encoder: bool = True,
        # Age prediction (auxiliary task for pretraining)
        use_age_prediction: bool = False,
        # Stochastic decoder params
        detach_sigma_head: bool = False,
        sigma_calibration: str = 'none',
        sigma_max_time_steps: int = 512,
    ):
        """
        Args:
            n_roi: Number of ROIs
            d_latent: Latent state dimension
            d_control: Control signal dimension
            d_episodic: Episodic memory dimension
            d_context: Task context dimension
            d_task: Task vector dimension
            n_classes: Number of output classes
            task_type: 'classification' or 'regression'
            graph_type: 'cheb' or 'gat' for graph encoder
            n_graph_layers: Number of graph conv layers
            cheb_k: Chebyshev polynomial order
            drop_edge_p: DropEdge probability
            n_temporal_layers: Number of transformer layers
            n_heads: Number of attention heads
            koopman_rank: Rank for DPLR decomposition
            spectral_radius: Target spectral radius
            use_sde: Whether to use stochastic dynamics
            diffusion_mode: 'learned', 'fixed', or 'none'
            init_noise_std: Initial noise standard deviation for fixed diffusion mode
            uncertainty_mode: 'heteroscedastic', 'homoscedastic', or 'none'
            fc_matrix: [N_ROI, N_ROI] FC adjacency (optional)
            dropout: Dropout rate
            use_episodic_feedback: Enable episodic feedback module (default: True)
            use_task_context: Enable task context encoder (default: True)
            use_control_fusion: Enable control fusion module (default: True)
            use_temporal_transformer: Enable temporal transformer (default: True)
            use_cognitive_readout: Enable cognitive readout head (default: True)
            use_spatial_encoder: Enable graph spatial encoder (default: True)
            temporal_model: 'transformer' or 'mamba' for temporal modeling (default: 'transformer')
            use_dual_path_mamba: If True, use TemporalMambaBi with separate causal/bi paths (default: False)
            
        Ablation Study Usage:
            To disable specific modules for ablation studies, set the corresponding flag to False:
            
            # Example: Train without episodic feedback
            model = KoopmanBrainModelV2(
                n_roi=116,
                use_episodic_feedback=False,  # Disable episodic feedback
                ...
            )
            
            # Example: Train without control system (episodic + task context + fusion)
            model = KoopmanBrainModelV2(
                n_roi=116,
                use_episodic_feedback=False,
                use_task_context=False,
                use_control_fusion=False,  # All control components disabled
                ...
            )
            
            # Example: Train with only basic encoder-decoder (no temporal, no control)
            model = KoopmanBrainModelV2(
                n_roi=116,
                use_temporal_transformer=False,
                use_episodic_feedback=False,
                use_task_context=False,
                use_control_fusion=False,
                ...
            )
        """
        super().__init__()
        
        # Dimension validation
        assert d_latent % n_heads == 0, f"d_latent ({d_latent}) must be divisible by n_heads ({n_heads})"
        assert koopman_rank <= d_latent, f"koopman_rank ({koopman_rank}) must be <= d_latent ({d_latent})"
        assert d_control <= d_latent, f"d_control ({d_control}) should be <= d_latent ({d_latent})"
        
        # Store config
        self.n_roi = n_roi
        self.d_latent = d_latent
        self.d_control = d_control
        self.d_episodic = d_episodic
        self.d_context = d_context
        self.n_classes = n_classes
        self.use_sde = use_sde
        self.uncertainty_mode = uncertainty_mode
        
        # Ablation flags
        self.use_episodic_feedback = use_episodic_feedback
        self.use_task_context = use_task_context
        self.use_control_fusion = use_control_fusion
        self.use_temporal_encoder = use_temporal_encoder
        self.use_spatial_encoder = use_spatial_encoder
        
        # 1. Graph spatial encoder
        if self.use_spatial_encoder:
            self.spatial_encoder = GraphSpatialEncoderReal(
                n_roi=n_roi,
                d_latent=d_latent,
                n_layers=n_graph_layers,
                graph_type=graph_type,
                cheb_k=cheb_k,
                n_heads=n_heads,
                dropout=dropout,
                drop_edge_p=drop_edge_p,
                fc_matrix=fc_matrix,
                use_temporal_encoding=True
            )
        else:
            # Simple linear projection as baseline (for ablation)
            self.spatial_encoder = nn.Linear(n_roi, d_latent)
        
        # 2. Temporal encoder
        if self.use_temporal_encoder:
                self.temporal_model_instance = TemporalTransformer(
                    d_latent=d_latent,
                    n_heads=n_heads,
                    n_layers=n_temporal_layers,
                    dropout=dropout
                )
        else:
            # Identity (no temporal processing) (for ablation)
            self.temporal_model_instance = nn.Identity()
        
        # 3. Episodic feedback module
        if self.use_episodic_feedback:
            self.episodic_feedback = EpisodicFeedbackTransformer(
                d_latent=d_latent,
                d_episodic=d_episodic,
                n_heads=n_heads // 2,
                dropout=dropout
            )
        else:
            self.episodic_feedback = None
        
        # 4. Task context encoder
        if self.use_task_context:
            self.task_context = TaskContextEncoder(
                d_task=d_task,
                d_context=d_context,
                dropout=dropout
            )
        else:
            self.task_context = None
        
        # 5. Control fusion module
        if self.use_control_fusion:
            self.control_fusion = ControlFusion(
                d_episodic=d_episodic,
                d_context=d_context,
                d_control=d_control,
                dropout=dropout
            )
        else:
            self.control_fusion = None
        
        # 6. Structured Koopman operator
        self.koopman_core = DPLRKoopmanCore(
            d_latent=d_latent,
            d_control=d_control,
            rank=koopman_rank,
            spectral_radius=spectral_radius,
            use_sde=use_sde,
            diffusion_mode=diffusion_mode,
            init_noise_std=init_noise_std
        )
        
        # 7. Stochastic decoder
        self.decoder = StochasticDecoder(
            d_latent=d_latent,
            n_roi=n_roi,
            hidden_dim=256,
            dropout=dropout,
            uncertainty_mode=uncertainty_mode,
            detach_sigma_head=detach_sigma_head,
            sigma_calibration=sigma_calibration,
            max_time_steps=sigma_max_time_steps,
        )
        
        # Regularizers
        self.spectral_regularizer = SpectralRegularizer(spectral_radius, weight=0.1)
        self.consistency_regularizer = ConsistencyRegularizer(weight=0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        task_vectors: Optional[torch.Tensor] = None,
        fc_matrix: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True,
        return_components: bool = False,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with spatiotemporal features, structured Koopman operators, and stochastic dynamics
        
        Args:
            x: [B, T, N_ROI] - ROI time series
            task_vectors: [B, T, d_task] - Task vectors (optional)
            fc_matrix: [B, N_ROI, N_ROI] or [N_ROI, N_ROI] - Optional FC matrices
            return_uncertainty: Whether to return uncertainty estimates
            return_components: Whether to return intermediate components
        
        Returns:
            Dictionary containing predictions and uncertainties
        """
        B, T, N = x.shape
        
        # 1. Graph spatial encoding
        if self.use_spatial_encoder:
            z_spatial = self.spatial_encoder(x, fc_matrix=fc_matrix)  # [B, T, d_latent]
        else:
            # Simple linear projection (for ablation)
            x_flat = x.reshape(B * T, N)
            z_flat = self.spatial_encoder(x_flat)
            z_spatial = z_flat.reshape(B, T, self.d_latent)
        
        # 2. Temporal encoder
        z_in = z_spatial  # Input to temporal model
        
        z_temporal = self.temporal_model_instance(z_in)  # [B, T, d_latent]
        
        # 3. Episodic feedback module
        if self.use_episodic_feedback:
            episodic_summary = self.episodic_feedback(z_temporal)  # [B, d_episodic]
        else:
            episodic_summary = torch.zeros(B, self.d_episodic, device=x.device, dtype=x.dtype)
        
        if self.use_task_context and task_vectors is not None:
            context_embedding = self.task_context(task_vectors)  # [B, T, d_context]
        else:
            context_embedding = torch.zeros(
                B, T, self.d_context,
                device=x.device, dtype=x.dtype
            )
        
        if self.use_control_fusion:
            control_signal = self.control_fusion(episodic_summary, context_embedding)  # [B, T, d_control]
        else:
            # Zero control signal (no control input)
            control_signal = torch.zeros(B, T, self.d_control, device=x.device, dtype=x.dtype)
        
        # 4. Structured Koopman operator
        z_next, sigma_z = self.koopman_core(
            z_temporal, 
            control_signal, 
            return_uncertainty=return_uncertainty
        )  # [B, T, d_latent], [B, T, d_latent]
        
        # 5. Stochastic decoder
        mu_recon, sigma_recon = self.decoder(z_temporal, return_uncertainty=return_uncertainty)
        
        # or prediction objectives
        mu_forecast, sigma_forecast = self.decoder(z_next[:, :-1, :], return_uncertainty=return_uncertainty)
        
        # 6. Age predictions (auxiliary task for pretraining)
        if self.use_age_prediction:
            age_pred, age_unc = self.age_predictor(
                z_temporal,
                pool=True,
                mc_uncertainty=return_uncertainty
            )
            age_pred = age_pred.squeeze(-1)  # [B, 1] -> [B]
        else:
            age_pred = None
            age_unc = None
        
        # Prepare outputs
        outputs = {
            # Predictions (means)
            'x_recon': mu_recon,           # [B, T, N_ROI]
            'x_forecast': mu_forecast,     # [B, T-1, N_ROI]
            'z_latent': z_temporal,        # [B, T, d_latent] (legacy key, kept for compatibility)
            'z_temporal': z_temporal,      # [B, T, d_latent] (explicit key for cycle loss)
            'z_next': z_next,              # [B, T, d_latent]
            'age_predictions': age_pred,    # [B] (age regression)
            'control_signal': control_signal,  # [B, T, d_control]
        }
        
        # Uncertainties
        if return_uncertainty:
            outputs.update({
                'sigma_recon': sigma_recon,              # [B, T, N_ROI]
                'sigma_forecast': sigma_forecast,        # [B, T-1, N_ROI]
                'sigma_z': sigma_z,                      # [B, T, d_latent]
                'age_uncertainty': age_unc,              # [B] (age uncertainty)
            })
        
        # Components for analysis (legacy, kept for backward compatibility)
        if return_components:
            components = {
                'z_spatial': z_spatial,
                'z_temporal': z_temporal,
                'episodic_summary': episodic_summary,
                'context_embedding': context_embedding,
                'control_signal': control_signal,
            }
            outputs['components'] = components
        
        # Intermediates for module-level probe losses
        if return_intermediates:
            intermediates = {
                'z_spatial': z_spatial,  # [B, T, d_latent]
                'z_temporal': z_temporal,  # [B, T, d_latent]
                'z_fused': z_temporal,  # [B, T, d_latent] (after temporal, before koopman)
                'episodic': episodic_summary,  # [B, d_episodic]
                'context': context_embedding,  # [B, T, d_context]
                'control': control_signal,  # [B, T, d_control]
                'z_koopman': z_next,  # [B, T, d_latent] (koopman next state)
                'mu_recon': mu_recon,  # [B, T, N_ROI]
                'mu_forecast': mu_forecast,  # [B, T-1, N_ROI]
            }
            outputs['intermediates'] = intermediates
        
        return outputs
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        x_target: torch.Tensor,
        cognitive_target: Optional[torch.Tensor] = None,
        age_target: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses with uncertainty-aware objectives
        
        Args:
            outputs: Model outputs from forward()
            x_target: [B, T, N_ROI] Target ROI signals
            loss_weights: Dict of loss weights
        
        Returns:
            Dictionary of loss components
        """
        if loss_weights is None:
            loss_weights = {
                'recon': 1.0,
                'forecast': 0.0, # Default to reconstruction only
                'consistency': 0.5,
                'spectral': 0.1,
                'age': 0.0, # Default to reconstruction only
            }
        
        # Training losses (weighted, included in total_loss)
        # 1. Reconstruction loss (NLL if uncertainty available)
        recon_weight = loss_weights.get('recon', 0.0)
        if recon_weight > 0:
            if 'sigma_recon' in outputs and outputs['sigma_recon'] is not None:
                # Gaussian NLL: -log p(x|z) = 0.5*log(2πσ²) + (x-μ)²/(2σ²)
                # Clamp sigma for numerical stability
                sigma = outputs['sigma_recon'].clamp(min=1e-3, max=10.0)
                mu = outputs['x_recon']
                nll = 0.5 * torch.log(2 * torch.pi * sigma**2 + 1e-8) + \
                      0.5 * (x_target - mu)**2 / (sigma**2 + 1e-8)
                # Raw, unweighted NLL for logging
                losses['recon_nll_raw'] = nll.mean().detach()
                losses['recon_loss'] = nll.mean() * recon_weight
            else:
                # Standard MSE (weighted) when no sigma
                mse = F.mse_loss(outputs['x_recon'], x_target)
                losses['recon_loss'] = mse * recon_weight
        # If recon_weight == 0, we still keep raw MSE metrics above, but do not add recon_loss
        
        # 2. Forecast loss (NLL if uncertainty available)
        forecast_weight = loss_weights.get('forecast', 0.0)
        if forecast_weight > 0 and 'x_forecast' in outputs and outputs['x_forecast'] is not None:
            x_forecast_target = x_target[:, 1:, :]
            if 'sigma_forecast' in outputs and outputs['sigma_forecast'] is not None:
                sigma = outputs['sigma_forecast'].clamp(min=1e-3, max=10.0)
                mu = outputs['x_forecast']
                nll = 0.5 * torch.log(2 * torch.pi * sigma**2 + 1e-8) + \
                      0.5 * (x_forecast_target - mu)**2 / (sigma**2 + 1e-8)
                losses['forecast_nll_raw'] = nll.mean().detach()
                losses['forecast_loss'] = nll.mean() * forecast_weight
            else:
                losses['forecast_loss'] = F.mse_loss(outputs['x_forecast'], x_forecast_target) * forecast_weight
        
        # 3. Age prediction loss (auxiliary task for pretraining)
        age_weight = loss_weights.get('age', 0.0)
        if age_weight > 0 and age_target is not None:
            # Age is regression, use MSE loss
            age_pred = outputs['age_predictions']  # [B]
            losses['age_loss'] = F.mse_loss(age_pred, age_target) * age_weight
            losses['age_mse_raw'] = F.mse_loss(age_pred, age_target).detach()
        
        # 4. Consistency regularization (latent dynamics)
        consistency_weight = loss_weights.get('consistency', 0.0)
        if consistency_weight > 0:
            z_true_next = outputs['z_latent'][:, 1:, :]
            z_pred_next = outputs['z_next'][:, :-1, :]
            losses['consistency_loss'] = self.consistency_regularizer(z_pred_next, z_true_next) * consistency_weight
        
        # 5. Spectral regularization on K matrix
        if loss_weights.get('spectral', 0.0) > 0:
            K_matrix = self.koopman_core.get_K_matrix()
            losses['spectral_loss'] = self.spectral_regularizer(K_matrix) * loss_weights.get('spectral', 0.1)
        
        # Total loss: only sum training losses (exclude raw metrics)
        TRAIN_LOSS_KEYS = [
            'recon_loss', 'forecast_loss', 'cognitive_loss', 'consistency_loss', 'spectral_loss'
        ]
        train_loss_terms = [losses[k] for k in TRAIN_LOSS_KEYS if k in losses]
        losses['total_loss'] = sum(train_loss_terms) if train_loss_terms else torch.tensor(0.0, device=x_target.device)
        
        return losses
    
    def multistep_rollout(
        self,
        x: torch.Tensor,
        task_vectors: Optional[torch.Tensor] = None,
        fc_matrix: Optional[torch.Tensor] = None,
        horizon: int = 8,
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-step rollout for long-horizon prediction
        
        Args:
            x: [B, T, N_ROI] Initial observations
            task_vectors: [B, T+horizon, d_task] Extended task vectors
            fc_matrix: [B, N_ROI, N_ROI] or [N_ROI, N_ROI] - Optional FC matrices
            horizon: Number of steps to predict ahead
            return_uncertainty: Whether to compute uncertainty
        
        Returns:
            Dictionary with rollout predictions
        """
        B, T, N = x.shape
        
        # Initial encoding
        if self.use_spatial_encoder:
            z_spatial = self.spatial_encoder(x, fc_matrix=fc_matrix)
        else:
            x_flat = x.reshape(B * T, N)
            z_flat = self.spatial_encoder(x_flat)
            z_spatial = z_flat.reshape(B, T, self.d_latent)
        
        # Temporal modeling
        if self.is_dual_path:
            z_temporal, _ = self.temporal_model_instance(z_spatial)
        else:
            z_temporal = self.temporal_model_instance(z_spatial)
        
        # Get episodic summary (fixed for entire rollout)
        if self.use_episodic_feedback:
            episodic_summary = self.episodic_feedback(z_temporal)
        else:
            episodic_summary = torch.zeros(B, self.d_episodic, device=x.device, dtype=x.dtype)
        
        # Prepare extended context
        T_extended = T + horizon
        if self.use_task_context and task_vectors is not None:
            context_embedding = self.task_context(task_vectors)
        else:
            context_embedding = torch.zeros(
                B, T_extended, self.d_context,
                device=x.device, dtype=x.dtype
            )
        
        # Generate extended control signals
        if self.use_control_fusion:
            control_extended = self.control_fusion(episodic_summary, context_embedding)
        else:
            control_extended = torch.zeros(B, T_extended, self.d_control, device=x.device, dtype=x.dtype)
        
        # Rollout from last observed state
        z_0 = z_temporal[:, -1, :]  # [B, d_latent]
        controls_future = control_extended[:, T:, :]  # [B, horizon, d_control]
        
        z_rollout, sigma_rollout = self.koopman_core.rollout(
            z_0, 
            controls_future, 
            return_trajectory=True,
            return_uncertainty=return_uncertainty
        )  # [B, horizon+1, d_latent]
        
        # Decode to ROI space
        z_rollout_decode = z_rollout[:, 1:, :]  # Skip initial state
        mu_rollout, sigma_x_rollout = self.decoder(z_rollout_decode, return_uncertainty=return_uncertainty)
        
        return {
            'z_rollout': z_rollout,           # [B, horizon+1, d_latent]
            'x_rollout': mu_rollout,          # [B, horizon, N_ROI]
            'sigma_z_rollout': sigma_rollout, # [B, horizon+1, d_latent] or None
            'sigma_x_rollout': sigma_x_rollout,  # [B, horizon, N_ROI] or None
        }
    
    def get_spectral_radius(self) -> float:
        """Get current spectral radius of Koopman operator"""
        return self.koopman_core.get_spectral_radius()
    
    def get_param_stats(self) -> dict:
        """Get parameter statistics including reduction factors"""
        koopman_stats = self.koopman_core.get_num_params()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            **koopman_stats,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


if __name__ == "__main__":
    # Test V2 model
    print("Koopman Brain Model V2 - Structured & Stochastic")
    print("=" * 60)
    
    # Model parameters
    n_roi = 116
    batch_size = 4
    seq_len = 100
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, n_roi)
    task_vectors = torch.randn(batch_size, seq_len, 32)
    cognitive_labels = torch.randint(0, 4, (batch_size,))
    
    # Create dummy FC matrix
    fc_matrix = torch.randn(n_roi, n_roi)
    fc_matrix = (fc_matrix + fc_matrix.T) / 2
    fc_matrix = torch.abs(fc_matrix)
    
    # Create model
    model = BrainMechBrainModel(
        n_roi=n_roi,
        d_latent=128,
        d_control=64,
        koopman_rank=16,
        graph_type='cheb',
        use_sde=True,
        diffusion_mode='learned',
        uncertainty_mode='heteroscedastic',
        fc_matrix=fc_matrix
    )
    
    # Forward pass
    print("\n1. Forward Pass")
    outputs = model(x, task_vectors, return_uncertainty=True, return_components=True)
    
    print(f"  Input: {x.shape}")
    print(f"  Outputs:")
    print(f"    x_recon: {outputs['x_recon'].shape}")
    print(f"    x_forecast: {outputs['x_forecast'].shape}")
    print(f"    sigma_recon: {outputs['sigma_recon'].shape if outputs['sigma_recon'] is not None else None}")
    print(f"    sigma_forecast: {outputs['sigma_forecast'].shape if outputs['sigma_forecast'] is not None else None}")
    print(f"    z_latent: {outputs['z_latent'].shape}")
    print(f"    sigma_z: {outputs['sigma_z'].shape if outputs['sigma_z'] is not None else None}")
    print(f"    cognitive_predictions: {outputs['cognitive_predictions'].shape}")
    print(f"    cognitive_uncertainty: {outputs['cognitive_uncertainty'].shape if outputs['cognitive_uncertainty'] is not None else None}")
    
    # Compute losses
    print("\n2. Losses")
    losses = model.compute_losses(outputs, x, cognitive_labels)
    for key, value in losses.items():
        print(f"    {key}: {value.item():.6f}")
    
    # Parameter analysis
    print("\n3. Parameter Analysis")
    stats = model.get_param_stats()
    print(f"  Koopman K reduction: {stats['k_reduction']:.2f}x")
    print(f"  Koopman C reduction: {stats['c_reduction']:.2f}x")
    print(f"  Total parameters: {stats['total_params']:,}")
    print(f"  Trainable parameters: {stats['trainable_params']:,}")
    
    # Spectral radius
    print(f"\n4. Spectral radius: {model.get_spectral_radius():.4f}")
    
    # Multi-step rollout
    print("\n5. Multi-step Rollout")
    task_extended = torch.randn(batch_size, seq_len + 8, 32)
    rollout_outputs = model.multistep_rollout(x, task_extended, horizon=8, return_uncertainty=True)
    print(f"  x_rollout: {rollout_outputs['x_rollout'].shape}")
    print(f"  sigma_x_rollout: {rollout_outputs['sigma_x_rollout'].shape if rollout_outputs['sigma_x_rollout'] is not None else None}")


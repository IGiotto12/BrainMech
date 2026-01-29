"""
Training Script for Koopman Brain Model V2
Features:
- Multi-step rollout curriculum (h=1→4→8)
- Risk-aware training (uncertainty-weighted losses)
- Structured parameter reduction tracking
- Calibrated uncertainty metrics
- Multi-GPU training with DDP
- MAE (Masked Autoencoder) pre-training support
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from typing import Dict, Optional, Tuple
from models.brainmech import BrainMech
from models.masked_pretraining import MaskedAutoencoderWrapper
from models.module_probes import ModuleProbeHeads, masked_mse
from utils.data_loader import HCPABOLDDataset, HCPABOLDWithFCDataset, UKBiobankBOLDDataset, UKBiobankBOLDWithFCDataset
from training.training_utils import (
    filter_kwargs_by_signature, parse_training_log, create_epoch_snapshot,
    log_tensor_stats, CollapseGuard
)


class CombinedDatasetWithMetadata(Dataset):
    """
    Wrapper around ConcatDataset that properly exposes metadata for subject-wise splitting.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.concat_dataset = ConcatDataset(datasets)
        # Combine metadata from all datasets
        self.metadata = []
        for dataset in datasets:
            if hasattr(dataset, 'metadata'):
                self.metadata.extend(dataset.metadata)
            else:
                # If dataset doesn't have metadata, create placeholder entries
                for i in range(len(dataset)):
                    self.metadata.append({'subject': f'unknown_{id(dataset)}_{i}'})
    
    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, idx):
        return self.concat_dataset[idx]
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

# Import downstream evaluation helpers
try:
    from training.evaluate_downstream_during_training import evaluate_age_sex_during_training
except ImportError:
    evaluate_age_sex_during_training = None

# Import disease evaluation
try:
    from eval.run_all_evaluations import run_disease_evaluation
    DISEASE_DATASETS = ['ABIDE', 'ADNI', 'PPMI', 'NEUROCON', 'TAOWU']
except ImportError:
    run_disease_evaluation = None
    DISEASE_DATASETS = []


def safe_barrier():
    """
    Safely call dist.barrier() without timeout parameter.
    
    Note: dist.barrier() does NOT accept a timeout parameter.
    The timeout is set during dist.init_process_group() and applies automatically.
    """
    dist.barrier()


def setup_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Set the CUDA device BEFORE initializing process group
        # This ensures the correct device is used when CUDA_VISIBLE_DEVICES is set
        # Note: When CUDA_VISIBLE_DEVICES=4,5,6,7 is set, PyTorch remaps those to logical indices 0,1,2,3
        # So local_rank=0 means physical GPU 4, local_rank=1 means physical GPU 5, etc.
        torch.cuda.set_device(local_rank)
        
        # Set NCCL timeout (in seconds, default 30 minutes for data loading)
        # Convert to timedelta for PyTorch
        from datetime import timedelta
        timeout_seconds = int(os.environ.get('NCCL_TIMEOUT', '1800'))
        timeout = timedelta(seconds=timeout_seconds)
        
        # Initialize process group
        # The device is already set via torch.cuda.set_device() above
        # NCCL warning about "guessing device ID" is informational - the device mapping is correct
        dist.init_process_group(
            backend='nccl', 
            init_method='env://',
            timeout=timeout
        )
        safe_barrier()
        
        return rank, world_size, local_rank
    else:
        return None, 1, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class CurriculumScheduler:
    """Curriculum scheduler for multi-step rollout training"""
    
    def __init__(self, start_horizon=1, max_horizon=8, warmup_epochs=10, increase_every=5):
        self.start_horizon = start_horizon
        self.max_horizon = max_horizon
        self.warmup_epochs = warmup_epochs
        self.increase_every = increase_every
    
    def get_horizon(self, epoch: int) -> int:
        """Get rollout horizon for current epoch"""
        if epoch < self.warmup_epochs:
            return self.start_horizon
        
        # Gradually increase: 1 → 2 → 4 → 8
        progress = (epoch - self.warmup_epochs) // self.increase_every
        horizon = min(self.start_horizon * (2 ** progress), self.max_horizon)
        return int(horizon)


def compute_calibration_metrics(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics for uncertainty estimates
    
    Args:
        mu: [N, ...] Predicted means
        sigma: [N, ...] Predicted std deviations
        target: [N, ...] Ground truth
        n_bins: Number of bins for calibration
    
    Returns:
        dict with calibration metrics
    """
    with torch.no_grad():
        # Flatten
        mu_flat = mu.reshape(-1)
        sigma_flat = sigma.reshape(-1)
        target_flat = target.reshape(-1)
        
        # Absolute errors
        errors = torch.abs(target_flat - mu_flat)
        
        # Expected calibration error (ECE)
        # Bin predictions by confidence
        confidences = 1.0 / (sigma_flat + 1e-8)
        
        # Sort by confidence
        sorted_idx = torch.argsort(confidences)
        errors_sorted = errors[sorted_idx]
        sigma_sorted = sigma_flat[sorted_idx]
        
        # Bin into quantiles
        n = len(errors_sorted)
        bin_size = n // n_bins
        
        ece = 0.0
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else n
            
            if end > start:
                bin_errors = errors_sorted[start:end]
                bin_sigmas = sigma_sorted[start:end]
                
                # Expected error in this bin
                expected_error = bin_sigmas.mean()
                actual_error = bin_errors.mean()
                
                # Contribution to ECE
                weight = (end - start) / n
                ece += weight * torch.abs(actual_error - expected_error)
        
        # Mean squared error
        mse = (errors ** 2).mean()
        
        # Negative log-likelihood
        nll = (0.5 * torch.log(2 * np.pi * sigma_flat ** 2 + 1e-8) + 
               0.5 * errors ** 2 / (sigma_flat ** 2 + 1e-8)).mean()
        
        return {
            'ece': ece.item(),
            'mse': mse.item(),
            'nll': nll.item(),
            'mean_sigma': sigma_flat.mean().item()
        }


def compute_probe_losses(
    probe_heads,
    intermediates: Dict[str, torch.Tensor],
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mask_forecast: Optional[torch.Tensor] = None,
    use_mae: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Compute probe losses for all modules with consistent mask and normalization.
    
    Args:
        probe_heads: ModuleProbeHeads instance
        intermediates: Dictionary of intermediate features from model forward
        x: [B, T, N_ROI] Target ROI signals
        mask: [B, T, 1] or [B, T, N_ROI] Mask for reconstruction (1 = include, 0 = exclude)
        mask_forecast: [B, T-1, 1] or [B, T-1, N_ROI] Mask for forecast
        use_mae: Whether using MAE mode
        
    Returns:
        Tuple of (probe_losses_dict, probe_counts_dict)
        - probe_losses_dict: {module_name: scalar_loss}
        - probe_counts_dict: {module_name: valid_token_count}
    """
    probe_losses_dict = {}
    probe_counts_dict = {}
    
    if probe_heads is None or intermediates is None:
        return probe_losses_dict, probe_counts_dict
    
    # Get mask: if None, create all-ones mask (include all tokens)
    if mask is None:
        mask = torch.ones_like(x)
    # Ensure mask is [B, T, N_ROI] shape
    if mask.dim() == 3 and mask.shape[2] == 1:
        mask = mask.expand(-1, -1, x.shape[2])
    
    # For forecast: mask[:, 1:]
    if mask.shape[1] > 1:
        mask_forecast = mask[:, 1:, :]
    else:
        mask_forecast = None
    
    # Compute probe predictions (all detached)
    probe_preds = probe_heads(intermediates, x, mask=mask, mask_forecast=mask_forecast)
    
    # Compute losses with consistent mask and normalization
    # Spatial recon
    if 'spatial_recon' in probe_preds:
        loss, count = _compute_masked_loss(probe_preds['spatial_recon'], x, mask)
        probe_losses_dict['spatial_recon'] = loss
        probe_counts_dict['spatial_recon'] = count
    
    # Temporal recon
    if 'temporal_recon' in probe_preds:
        loss, count = _compute_masked_loss(probe_preds['temporal_recon'], x, mask)
        probe_losses_dict['temporal_recon'] = loss
        probe_counts_dict['temporal_recon'] = count
    
    # Fused recon
    if 'fused_recon' in probe_preds:
        loss, count = _compute_masked_loss(probe_preds['fused_recon'], x, mask)
        probe_losses_dict['fused_recon'] = loss
        probe_counts_dict['fused_recon'] = count
    
    # Control recon
    if 'control_recon' in probe_preds:
        loss, count = _compute_masked_loss(probe_preds['control_recon'], x, mask)
        probe_losses_dict['control_recon'] = loss
        probe_counts_dict['control_recon'] = count
    
    # Context recon
    if 'context_recon' in probe_preds:
        loss, count = _compute_masked_loss(probe_preds['context_recon'], x, mask)
        probe_losses_dict['context_recon'] = loss
        probe_counts_dict['context_recon'] = count
    
    # Episodic recon (time-averaged target)
    if 'episodic_recon' in probe_preds and 'episodic_target' in probe_preds:
        pred_episodic = probe_preds['episodic_recon']  # [B, N_ROI]
        target_episodic = probe_preds['episodic_target']  # [B, N_ROI]
        loss = F.mse_loss(pred_episodic, target_episodic)
        count = pred_episodic.numel()  # B * N_ROI
        probe_losses_dict['episodic_recon'] = loss
        probe_counts_dict['episodic_recon'] = count
    
    # Koopman forecast
    if 'koopman_forecast' in probe_preds:
        pred_forecast = probe_preds['koopman_forecast']  # [B, T-1, N_ROI]
        target_forecast = x[:, 1:, :]  # [B, T-1, N_ROI]
        loss, count = _compute_masked_loss(pred_forecast, target_forecast, mask_forecast)
        probe_losses_dict['koopman_forecast'] = loss
        probe_counts_dict['koopman_forecast'] = count
    
    return probe_losses_dict, probe_counts_dict


def _compute_masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, int]:
    """
    Compute masked MSE loss and return loss + valid token count.
    
    Returns:
        Tuple of (loss_scalar, valid_token_count)
    """
    # Ensure shapes match
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    # Compute squared error
    se = (pred - target) ** 2  # [B, T, N_ROI]
    
    # Apply mask if provided
    if mask is not None:
        # Ensure mask matches pred shape
        if mask.shape != pred.shape:
            if mask.dim() == 3 and mask.shape[2] == 1:
                mask = mask.expand(-1, -1, pred.shape[2])
            elif mask.shape[1] != pred.shape[1]:
                # Handle forecast case: mask might be [B, T, N_ROI] but pred is [B, T-1, N_ROI]
                if mask.shape[1] == pred.shape[1] + 1:
                    mask = mask[:, 1:, :]
                else:
                    raise ValueError(f"Mask shape mismatch: mask {mask.shape} vs pred {pred.shape}")
        
        # Mask: 1 = include, 0 = exclude
        se = se * mask
        num_valid = mask.sum().item()
    else:
        num_valid = se.numel()
    
    # Compute mean over valid tokens
    if num_valid > 0:
        loss = se.sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss, int(num_valid)


def train_epoch(
    model, 
    train_loader, 
    optimizer, 
    device, 
    loss_weights,
    rollout_horizon=1,
    risk_aware=True,
    use_mae=False,
    epoch=1,
    max_epochs=50,
    rank=0,
    is_distributed=False,
    probe_heads=None,
    probe_optimizer=None,
    enable_module_probes=False,
    probe_weight=1.0,
    writer=None,
    global_step=0,
    shape_debug_once=True,
    shape_debug_printed_ref=None,  # Reference to global flag (list to allow modification)
    use_age_context=False,
    age_dict=None,
    age_min=8.0,
    age_max=100.0,
    task_dim=32
):
    """Train for one epoch with optional multi-step rollout and MAE"""
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_forecast = 0
    total_rollout = 0
    total_mae = 0
    # Initialize raw metrics accumulators
    total_recon_mse_raw = 0.0
    total_recon_mae_raw = 0.0
    total_recon_nll_raw = 0.0
    
    # Probe loss accumulators
    probe_loss_accum = {} if enable_module_probes else None
    
    # Shape debug flag (only print once across all epochs)
    if shape_debug_printed_ref is not None:
        shape_debug_printed = shape_debug_printed_ref[0]  # Use global flag
    else:
        shape_debug_printed = not shape_debug_once  # Fallback: if disabled, mark as already printed
    
    pbar = tqdm(
        train_loader, 
        desc="Training", 
        disable=(rank != 0 and is_distributed),
        mininterval=0.5,  # 最小更新间隔（秒），减少刷新频率
        maxinterval=2.0,  # 最大更新间隔（秒）
        smoothing=0.1,    # 平滑系数，减少数值跳动
        dynamic_ncols=True,  # 动态调整列宽以适应终端
        leave=True        # 完成后保留进度条
    )
    for batch_idx, (x, metadata) in enumerate(pbar):
        # Clear cache before processing batch to reduce fragmentation
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()
        
        x = x.to(device, non_blocking=True)
        B, T, N = x.shape
        
        # Extract FC matrix from metadata if available
        fc_matrix = None
        if isinstance(metadata, dict) and 'fc_matrix' in metadata:
            fc_value = metadata['fc_matrix']
            if isinstance(fc_value, torch.Tensor):
                fc_matrix = fc_value.to(device, non_blocking=True)  # [B, N, N]
            elif isinstance(fc_value, list) and all(isinstance(v, torch.Tensor) for v in fc_value):
                fc_matrix = torch.stack(fc_value, dim=0).to(device, non_blocking=True)
        
        # Clear cache before forward pass to reduce fragmentation (especially for large batches)
        if batch_idx % 3 == 0:
            torch.cuda.empty_cache()
        
        # Extract age and create task_vectors if age context is enabled
        task_vectors = None
        if use_age_context and args.use_task_context:
            # Extract age from metadata or age_dict
            ages = []
            
            # Handle metadata: could be list of dicts (one per sample) or single dict
            if isinstance(metadata, list):
                # List of metadata dicts (one per sample in batch)
                for i in range(B):
                    meta_item = metadata[i] if i < len(metadata) else {}
                    age = None
                    
                    if isinstance(meta_item, dict):
                        # Check metadata for age
                        if 'age' in meta_item:
                            age = float(meta_item['age'])
                        # Otherwise, try to get from subject ID
                        elif 'subject' in meta_item:
                            subj_id = str(meta_item['subject']).strip()
                            age = age_dict.get(subj_id)
                    
                    # Default age if not found (use middle of range)
                    if age is None:
                        age = (age_min + age_max) / 2.0
                    
                    ages.append(age)
            else:
                # Single metadata dict (same for all samples in batch)
                meta_item = metadata if isinstance(metadata, dict) else {}
                age = None
                
                if isinstance(meta_item, dict):
                    # Check metadata for age
                    if 'age' in meta_item:
                        age = float(meta_item['age'])
                    # Otherwise, try to get from subject ID
                    elif 'subject' in meta_item:
                        subj_id = str(meta_item['subject']).strip()
                        age = age_dict.get(subj_id)
                
                # Default age if not found (use middle of range)
                if age is None:
                    age = (age_min + age_max) / 2.0
                
                # Use same age for all samples in batch
                ages = [age] * B
            
            # Normalize age to [0, 1] range
            age_tensor = torch.tensor(ages, dtype=torch.float32, device=device)  # [B]
            age_normalized = (age_tensor - age_min) / (age_max - age_min + 1e-8)  # [B]
            age_normalized = torch.clamp(age_normalized, 0.0, 1.0)
            
            # Create task_vectors [B, T, d_task]
            # Age is the first feature, rest are zeros (or can be extended with other features)
            task_vectors = torch.zeros(B, T, args.task_dim, device=device, dtype=torch.float32)
            task_vectors[:, :, 0] = age_normalized.unsqueeze(1).expand(-1, T)  # [B, T]
        
        # Shape debug: print once on first batch (rank 0 only, across all epochs)
        # Don't set shape_debug_printed yet - wait until after intermediate shapes are printed
        should_print_shape_debug = not shape_debug_printed and (rank == 0 or not is_distributed) and batch_idx == 0
        if should_print_shape_debug:
            print(f"\n【关键张量Shape调试 (Key Tensor Shapes) - 仅打印一次】")
            print(f"  x (input): {x.shape}  # [B, T, n_roi]")
            if fc_matrix is not None:
                print(f"  fc_matrix: {fc_matrix.shape}  # [B, n_roi, n_roi]")
            # Get world_size and effective batch
            if is_distributed:
                world_size = dist.get_world_size()
                effective_batch = B * world_size
                print(f"  Per-GPU batch: {B}, World size: {world_size}, Effective batch: {effective_batch}")
            else:
                print(f"  Batch size: {B} (single GPU)")
            # Note: shape_debug_printed will be set AFTER intermediate shapes are printed
        
        optimizer.zero_grad()
        
        # === MAE pre-training (if enabled) ===
        # Enable return_intermediates for shape debug or probe losses
        # Use should_print_shape_debug flag (computed above) to determine if we need intermediates
        need_intermediates_mae = enable_module_probes or should_print_shape_debug
        
        if use_mae:
            if is_distributed:
                outputs = model(
                    x, task_vectors=task_vectors, fc_matrix=fc_matrix, 
                    use_masking=True, epoch=epoch, max_epochs=max_epochs,
                    return_intermediates=need_intermediates_mae
                )
                losses = model.module.compute_mae_loss(outputs, use_masked_loss=True)
                
                # Shape debug: print intermediate shapes once (MAE mode)
                if should_print_shape_debug and 'intermediates' in outputs:
                    intermediates = outputs['intermediates']
                    print(f"  z_spatial: {intermediates.get('z_spatial', torch.tensor([])).shape if isinstance(intermediates.get('z_spatial'), torch.Tensor) else 'N/A'}")
                    z_wavelet = intermediates.get('z_wavelet')
                    if z_wavelet is not None and isinstance(z_wavelet, torch.Tensor):
                        print(f"  z_wavelet: {z_wavelet.shape}")
                    z_temporal = intermediates.get('z_temporal')
                    print(f"  z_temporal: {z_temporal.shape if isinstance(z_temporal, torch.Tensor) else 'N/A'}")
                    z_fused = intermediates.get('z_fused')
                    print(f"  z_fused: {z_fused.shape if isinstance(z_fused, torch.Tensor) else 'N/A'}")
                    control = intermediates.get('control')
                    print(f"  control (b): {control.shape if isinstance(control, torch.Tensor) else 'N/A'}")
                    z_koopman = intermediates.get('z_koopman')
                    print(f"  z_koopman: {z_koopman.shape if isinstance(z_koopman, torch.Tensor) else 'N/A'}")
                    mu_recon = outputs.get('x_recon')
                    print(f"  mu_recon: {mu_recon.shape if isinstance(mu_recon, torch.Tensor) else 'N/A'}")
                    print()  # Empty line for readability
                    # Mark as printed after intermediate shapes
                    shape_debug_printed = True
                    if shape_debug_printed_ref is not None:
                        shape_debug_printed_ref[0] = True
            else:
                outputs = model(
                    x, task_vectors=task_vectors, fc_matrix=fc_matrix, 
                    use_masking=True, epoch=epoch, max_epochs=max_epochs,
                    return_intermediates=need_intermediates_mae
                )
                losses = model.compute_mae_loss(outputs, use_masked_loss=True)
                
                # Shape debug: print intermediate shapes once (MAE mode)
                if should_print_shape_debug and 'intermediates' in outputs:
                    intermediates = outputs['intermediates']
                    print(f"  z_spatial: {intermediates.get('z_spatial', torch.tensor([])).shape if isinstance(intermediates.get('z_spatial'), torch.Tensor) else 'N/A'}")
                    z_wavelet = intermediates.get('z_wavelet')
                    if z_wavelet is not None and isinstance(z_wavelet, torch.Tensor):
                        print(f"  z_wavelet: {z_wavelet.shape}")
                    z_temporal = intermediates.get('z_temporal')
                    print(f"  z_temporal: {z_temporal.shape if isinstance(z_temporal, torch.Tensor) else 'N/A'}")
                    z_fused = intermediates.get('z_fused')
                    print(f"  z_fused: {z_fused.shape if isinstance(z_fused, torch.Tensor) else 'N/A'}")
                    control = intermediates.get('control')
                    print(f"  control (b): {control.shape if isinstance(control, torch.Tensor) else 'N/A'}")
                    z_koopman = intermediates.get('z_koopman')
                    print(f"  z_koopman: {z_koopman.shape if isinstance(z_koopman, torch.Tensor) else 'N/A'}")
                    mu_recon = outputs.get('x_recon')
                    print(f"  mu_recon: {mu_recon.shape if isinstance(mu_recon, torch.Tensor) else 'N/A'}")
                    print()  # Empty line for readability
                    # Mark as printed after intermediate shapes
                    shape_debug_printed = True
                    if shape_debug_printed_ref is not None:
                        shape_debug_printed_ref[0] = True
            
            loss = losses['total_loss']
            total_mae += losses.get('recon_mse', torch.tensor(0.0)).item()
        else:
            # === Standard 1-step prediction ===
            # Enable return_intermediates for shape debug or probe losses
            # Use should_print_shape_debug flag (computed above) to determine if we need intermediates
            need_intermediates = enable_module_probes or should_print_shape_debug
            
            if is_distributed:
                outputs = model.module.base_model(
                    x, task_vectors=task_vectors, fc_matrix=fc_matrix, 
                    return_uncertainty=True, 
                    return_intermediates=need_intermediates
                )
                losses = model.module.base_model.compute_losses(outputs, x, cognitive_target=None, loss_weights=loss_weights)
                
                # Shape debug: print intermediate shapes once (non-MAE mode)
                if should_print_shape_debug and 'intermediates' in outputs:
                    intermediates = outputs['intermediates']
                    print(f"  z_spatial: {intermediates.get('z_spatial', torch.tensor([])).shape if isinstance(intermediates.get('z_spatial'), torch.Tensor) else 'N/A'}")
                    z_wavelet = intermediates.get('z_wavelet')
                    if z_wavelet is not None and isinstance(z_wavelet, torch.Tensor):
                        print(f"  z_wavelet: {z_wavelet.shape}")
                    z_temporal = intermediates.get('z_temporal')
                    print(f"  z_temporal: {z_temporal.shape if isinstance(z_temporal, torch.Tensor) else 'N/A'}")
                    z_fused = intermediates.get('z_fused')
                    print(f"  z_fused: {z_fused.shape if isinstance(z_fused, torch.Tensor) else 'N/A'}")
                    control = intermediates.get('control')
                    print(f"  control (b): {control.shape if isinstance(control, torch.Tensor) else 'N/A'}")
                    z_koopman = intermediates.get('z_koopman')
                    print(f"  z_koopman: {z_koopman.shape if isinstance(z_koopman, torch.Tensor) else 'N/A'}")
                    mu_recon = outputs.get('x_recon')
                    print(f"  mu_recon: {mu_recon.shape if isinstance(mu_recon, torch.Tensor) else 'N/A'}")
                    print()  # Empty line for readability
                    # Mark as printed after intermediate shapes
                    shape_debug_printed = True
                    if shape_debug_printed_ref is not None:
                        shape_debug_printed_ref[0] = True
            else:
                outputs = model.base_model(
                    x, task_vectors=task_vectors, fc_matrix=fc_matrix, 
                    return_uncertainty=True,
                    return_intermediates=need_intermediates
                )
                losses = model.base_model.compute_losses(outputs, x, cognitive_target=None, loss_weights=loss_weights)
                
                # Shape debug: print intermediate shapes once (non-MAE mode)
                if should_print_shape_debug and 'intermediates' in outputs:
                    intermediates = outputs['intermediates']
                    print(f"  z_spatial: {intermediates.get('z_spatial', torch.tensor([])).shape if isinstance(intermediates.get('z_spatial'), torch.Tensor) else 'N/A'}")
                    z_wavelet = intermediates.get('z_wavelet')
                    if z_wavelet is not None and isinstance(z_wavelet, torch.Tensor):
                        print(f"  z_wavelet: {z_wavelet.shape}")
                    z_temporal = intermediates.get('z_temporal')
                    print(f"  z_temporal: {z_temporal.shape if isinstance(z_temporal, torch.Tensor) else 'N/A'}")
                    z_fused = intermediates.get('z_fused')
                    print(f"  z_fused: {z_fused.shape if isinstance(z_fused, torch.Tensor) else 'N/A'}")
                    control = intermediates.get('control')
                    print(f"  control (b): {control.shape if isinstance(control, torch.Tensor) else 'N/A'}")
                    z_koopman = intermediates.get('z_koopman')
                    print(f"  z_koopman: {z_koopman.shape if isinstance(z_koopman, torch.Tensor) else 'N/A'}")
                    mu_recon = outputs.get('x_recon')
                    print(f"  mu_recon: {mu_recon.shape if isinstance(mu_recon, torch.Tensor) else 'N/A'}")
                    print()  # Empty line for readability
                    # Mark as printed after intermediate shapes
                    shape_debug_printed = True
                    if shape_debug_printed_ref is not None:
                        shape_debug_printed_ref[0] = True
            
            loss = losses['total_loss']
            
            # === Multi-step rollout loss (curriculum) ===
            if rollout_horizon > 1 and T > rollout_horizon + 10:
                # Use first part for encoding, predict remainder
                T_enc = T - rollout_horizon
                x_enc = x[:, :T_enc, :]
                x_target_rollout = x[:, T_enc:, :]
                
                # Extend task_vectors for rollout (if age context is enabled)
                rollout_task_vectors = None
                if task_vectors is not None:
                    # Extend task_vectors to match extended time horizon [B, T_enc+horizon, d_task]
                    T_extended = T_enc + rollout_horizon
                    rollout_task_vectors = torch.zeros(B, T_extended, task_vectors.shape[2], device=device, dtype=torch.float32)
                    # Copy age from first timestep and repeat across extended horizon
                    rollout_task_vectors[:, :, 0] = task_vectors[:, 0, 0].unsqueeze(1).expand(-1, T_extended)
                
                # Rollout
                if is_distributed:
                    rollout_outputs = model.module.base_model.multistep_rollout(
                        x_enc, 
                        task_vectors=rollout_task_vectors, 
                        fc_matrix=fc_matrix,
                        horizon=rollout_horizon,
                        return_uncertainty=True
                    )
                else:
                    rollout_outputs = model.base_model.multistep_rollout(
                        x_enc, 
                        task_vectors=rollout_task_vectors, 
                        fc_matrix=fc_matrix,
                        horizon=rollout_horizon,
                        return_uncertainty=True
                    )
                
                # Rollout loss
                mu_rollout = rollout_outputs['x_rollout']
                sigma_rollout = rollout_outputs['sigma_x_rollout']
                
                if sigma_rollout is not None and risk_aware:
                    # Risk-aware loss: down-weight high-uncertainty predictions
                    nll = 0.5 * torch.log(2 * np.pi * sigma_rollout**2 + 1e-8) + \
                          0.5 * (x_target_rollout - mu_rollout)**2 / (sigma_rollout**2 + 1e-8)
                    rollout_loss = nll.mean()
                else:
                    rollout_loss = F.mse_loss(mu_rollout, x_target_rollout)
                
                # Add to total loss
                loss = loss + rollout_loss * loss_weights.get('rollout', 0.5)
                total_rollout += rollout_loss.item()
        
        # === Module probe losses (if enabled) ===
        probe_losses_dict = {}
        probe_counts_dict = {}
        probe_loss_total = None
        if enable_module_probes and probe_heads is not None:
            # Get intermediates from model (need to call with return_intermediates=True)
            if use_mae:
                if is_distributed:
                    base_outputs = model.module.base_model(
                        x if 'mask' not in outputs else outputs.get('x_original', x),
                        task_vectors=task_vectors,
                        fc_matrix=fc_matrix,
                        return_uncertainty=True,
                        return_intermediates=True
                    )
                else:
                    base_outputs = model.base_model(
                        x if 'mask' not in outputs else outputs.get('x_original', x),
                        task_vectors=task_vectors,
                        fc_matrix=fc_matrix,
                        return_uncertainty=True,
                        return_intermediates=True
                    )
            else:
                # For non-MAE mode, outputs already has intermediates if return_intermediates was True
                base_outputs = outputs
            
            if 'intermediates' in base_outputs:
                intermediates = base_outputs['intermediates']
                # Get mask from outputs (MAE) or create all-ones mask
                # MAE mask: 1 = masked (exclude), 0 = visible (include)
                # For probe: we want to compute loss on visible tokens, so invert mask
                if use_mae:
                    mask = outputs.get('mask', torch.ones_like(x))
                    # Invert: 1 -> 0 (exclude), 0 -> 1 (include)
                    mask = 1.0 - mask
                else:
                    mask = torch.ones_like(x)  # Include all tokens
                
                # Compute probe losses with unified function
                probe_losses_dict, probe_counts_dict = compute_probe_losses(
                    probe_heads, intermediates, x, mask=mask, use_mae=use_mae
                )
                
                # Total probe loss (for training probe heads)
                if probe_losses_dict:
                    probe_loss_total = sum(probe_losses_dict.values()) * probe_weight
                    
                    # Backward probe loss (separate optimizer)
                    if probe_optimizer is not None:
                        probe_optimizer.zero_grad()
                        probe_loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(probe_heads.parameters(), max_norm=1.0)
                        probe_optimizer.step()
                    
                    # Accumulate probe losses for logging (sum/count format)
                    # sum = sum(loss * count) for each batch, total_count = sum(count)
                    # final_mean = sum / total_count
                    if probe_loss_accum is not None:
                        for k, v in probe_losses_dict.items():
                            count = probe_counts_dict.get(k, 0)
                            if count > 0:  # Only accumulate if count > 0
                                if k not in probe_loss_accum:
                                    probe_loss_accum[k] = {'sum': 0.0, 'count': 0}
                                # loss is mean over valid tokens, so loss * count = total loss for this batch
                                probe_loss_accum[k]['sum'] += v.item() * count
                                probe_loss_accum[k]['count'] += count
        
        # Backward main loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Clear cache periodically to reduce memory fragmentation
        # Only clear every N batches to avoid performance impact
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Accumulate training losses
        total_loss += losses['total_loss'].item()
        total_recon += losses.get('recon_loss', torch.tensor(0.0)).item()
        total_forecast += losses.get('forecast_loss', torch.tensor(0.0)).item()
        
        # Accumulate raw metrics (for consistent reporting)
        if 'recon_mse_raw' in losses:
            total_recon_mse_raw += losses['recon_mse_raw'].item()
        if 'recon_mae_raw' in losses:
            total_recon_mae_raw += losses['recon_mae_raw'].item()
        if 'recon_nll_raw' in losses:
            total_recon_nll_raw += losses['recon_nll_raw'].item()
        
        if rank == 0 or not is_distributed:
            # Show raw MSE in progress bar for consistency
            recon_mse_display = losses.get('recon_mse_raw', torch.tensor(0.0)).item()
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'recon_mse': f"{recon_mse_display:.4f}",
                'horizon': rollout_horizon if not use_mae else 'MAE'
            })
            
            # Log probe losses to TensorBoard (if enabled and writer available)
            if writer is not None and enable_module_probes and probe_losses_dict:
                for k, v in probe_losses_dict.items():
                    if isinstance(v, torch.Tensor):
                        writer.add_scalar(f'loss/probe/train/{k}', v.item(), global_step)
                    else:
                        writer.add_scalar(f'loss/probe/train/{k}', v, global_step)
    
    n_batches = len(train_loader)
    
    # Average across GPUs if distributed
    metrics = {
        'total_loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,  # This is weighted loss (NLL or MSE)
        'forecast_loss': total_forecast / n_batches,
        'shape_debug_printed': shape_debug_printed,  # Return flag status
        'rollout_loss': total_rollout / n_batches if rollout_horizon > 1 else 0.0,
        'mae_loss': total_mae / n_batches if use_mae else 0.0,
        # Probe losses (will be populated if enabled)
        # Format: {module_name: {'mean': float, 'count': int}}
        'probe_losses': {
            k: {
                'mean': v['sum'] / (v['count'] + 1e-8) if v['count'] > 0 else None,
                'count': v['count']
            }
            for k, v in probe_loss_accum.items()
        } if probe_loss_accum else {},
        'recon_mse_raw': total_recon_mse_raw / n_batches,
        'recon_mae_raw': total_recon_mae_raw / n_batches,
        'recon_nll_raw': total_recon_nll_raw / n_batches if total_recon_nll_raw != 0.0 else 0.0,
    }
    
    if is_distributed:
        # Handle dict values separately (probe_losses)
        for key in list(metrics.keys()):
            if key == 'probe_losses':
                continue  # Skip dict, handle separately if needed
            if key == 'shape_debug_printed':
                continue  # Skip boolean flag
            tensor = torch.tensor(metrics[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            metrics[key] = tensor.item()
    
    return metrics


def validate_epoch(model, val_loader, device, loss_weights, rollout_horizon=8, use_mae=False, rank=0, is_distributed=False, probe_heads=None, enable_module_probes=False, epoch=1, max_epochs=50, use_age_context=False, age_dict=None, age_min=8.0, age_max=100.0, task_dim=32):
    """Validation with calibration metrics"""
    model.eval()
    if probe_heads is not None:
        probe_heads.eval()
    
    total_loss = 0
    total_recon = 0
    total_forecast = 0
    total_recon_nll_raw = 0.0
    
    # Probe loss accumulators
    probe_loss_accum = {} if enable_module_probes else None
    
    # For calibration
    all_mu_recon = []
    all_sigma_recon = []
    all_target = []
    
    # For rollout
    rollout_maes = []
    
    with torch.no_grad():
        pbar = tqdm(
            val_loader, 
            desc="Validation", 
            disable=(rank != 0 and is_distributed),
            mininterval=0.5,
            maxinterval=2.0,
            smoothing=0.1,
            dynamic_ncols=True,
            leave=True
        )
        for x, metadata in pbar:
            x = x.to(device, non_blocking=True)
            B, T, N = x.shape
            
            # Extract FC matrix from metadata if available
            fc_matrix = None
            if isinstance(metadata, dict) and 'fc_matrix' in metadata:
                fc_value = metadata['fc_matrix']
                if isinstance(fc_value, torch.Tensor):
                    fc_matrix = fc_value.to(device, non_blocking=True)  # [B, N, N]
                elif isinstance(fc_value, list) and all(isinstance(v, torch.Tensor) for v in fc_value):
                    fc_matrix = torch.stack(fc_value, dim=0).to(device, non_blocking=True)
            
            # Extract age and create task_vectors if age context is enabled
            task_vectors = None
            if use_age_context and task_dim > 0:
                # Extract age from metadata or age_dict (same logic as training)
                ages = []
                
                if isinstance(metadata, list):
                    for i in range(B):
                        meta_item = metadata[i] if i < len(metadata) else {}
                        age = None
                        if isinstance(meta_item, dict):
                            if 'age' in meta_item:
                                age = float(meta_item['age'])
                            elif 'subject' in meta_item:
                                subj_id = str(meta_item['subject']).strip()
                                age = age_dict.get(subj_id) if age_dict else None
                        if age is None:
                            age = (age_min + age_max) / 2.0
                        ages.append(age)
                else:
                    meta_item = metadata if isinstance(metadata, dict) else {}
                    age = None
                    if isinstance(meta_item, dict):
                        if 'age' in meta_item:
                            age = float(meta_item['age'])
                        elif 'subject' in meta_item:
                            subj_id = str(meta_item['subject']).strip()
                            age = age_dict.get(subj_id) if age_dict else None
                    if age is None:
                        age = (age_min + age_max) / 2.0
                    ages = [age] * B
                
                # Normalize age to [0, 1] and create task_vectors
                age_tensor = torch.tensor(ages, dtype=torch.float32, device=device)
                age_normalized = (age_tensor - age_min) / (age_max - age_min + 1e-8)
                age_normalized = torch.clamp(age_normalized, 0.0, 1.0)
                
                task_vectors = torch.zeros(B, T, task_dim, device=device, dtype=torch.float32)
                task_vectors[:, :, 0] = age_normalized.unsqueeze(1).expand(-1, T)
            
            if use_mae:
                # MAE validation (without masking, but need intermediates for probes)
                if is_distributed:
                    # Use masking during validation for fair comparison with training loss
                    # This ensures we're comparing losses computed on the same regions
                    outputs = model(x, task_vectors=task_vectors, fc_matrix=fc_matrix, use_masking=True, epoch=epoch, max_epochs=max_epochs)
                    losses = model.module.compute_mae_loss(outputs, use_masked_loss=True)
                    # Get intermediates for probes
                    if enable_module_probes:
                        base_outputs = model.module.base_model(
                            x, task_vectors=task_vectors, fc_matrix=fc_matrix,
                            return_uncertainty=True, return_intermediates=True
                        )
                        if 'intermediates' not in outputs:
                            outputs['intermediates'] = base_outputs.get('intermediates', {})
                else:
                    # Use masking during validation for fair comparison with training loss
                    # This ensures we're comparing losses computed on the same regions
                    outputs = model(x, task_vectors=task_vectors, fc_matrix=fc_matrix, use_masking=True, epoch=epoch, max_epochs=max_epochs)
                    losses = model.compute_mae_loss(outputs, use_masked_loss=True)
                    # Get intermediates for probes
                    if enable_module_probes:
                        base_outputs = model.base_model(
                            x, task_vectors=task_vectors, fc_matrix=fc_matrix,
                            return_uncertainty=True, return_intermediates=True
                        )
                        if 'intermediates' not in outputs:
                            outputs['intermediates'] = base_outputs.get('intermediates', {})
            else:
                # Standard forward (need return_intermediates for probes)
                if is_distributed:
                    outputs = model.module.base_model(
                        x, task_vectors=task_vectors, fc_matrix=fc_matrix,
                        return_uncertainty=True, return_intermediates=enable_module_probes
                    )
                    losses = model.module.base_model.compute_losses(outputs, x, cognitive_target=None, loss_weights=loss_weights)
                else:
                    outputs = model.base_model(
                        x, task_vectors=task_vectors, fc_matrix=fc_matrix,
                        return_uncertainty=True, return_intermediates=enable_module_probes
                    )
                    losses = model.base_model.compute_losses(outputs, x, cognitive_target=None, loss_weights=loss_weights)
            
            # === Module probe losses (if enabled) ===
            if enable_module_probes and probe_heads is not None and 'intermediates' in outputs:
                intermediates = outputs['intermediates']
                # For validation: use all tokens (no masking)
                mask = torch.ones_like(x)  # Include all tokens
                
                # Compute probe losses with unified function
                probe_losses_dict, probe_counts_dict = compute_probe_losses(
                    probe_heads, intermediates, x, mask=mask, use_mae=use_mae
                )
                
                # Accumulate probe losses for logging (sum/count format)
                # sum = sum(loss * count) for each batch, total_count = sum(count)
                # final_mean = sum / total_count
                if probe_loss_accum is not None:
                    for k, v in probe_losses_dict.items():
                        count = probe_counts_dict.get(k, 0)
                        if count > 0:  # Only accumulate if count > 0
                            if k not in probe_loss_accum:
                                probe_loss_accum[k] = {'sum': 0.0, 'count': 0}
                            # loss is mean over valid tokens, so loss * count = total loss for this batch
                            probe_loss_accum[k]['sum'] += v.item() * count
                            probe_loss_accum[k]['count'] += count
            
            total_loss += losses['total_loss'].item()
            total_recon += losses.get('recon_loss', losses.get('recon_mse', torch.tensor(0.0))).item()
            total_forecast += losses.get('forecast_loss', torch.tensor(0.0)).item()
            if 'recon_nll_raw' in losses:
                total_recon_nll_raw += losses['recon_nll_raw'].item()
            
            # Collect for calibration and R² (works for both MAE and non-MAE modes)
            # MAE wrapper returns x_recon, and base model also returns x_recon
            if 'x_recon' in outputs:
                all_mu_recon.append(outputs['x_recon'].cpu())
                if outputs.get('sigma_recon') is not None:
                    all_sigma_recon.append(outputs['sigma_recon'].cpu())
                # For MAE, use x_original if available, otherwise use x
                target_for_r2 = outputs.get('x_original', x).cpu()
                all_target.append(target_for_r2)
            
            # Multi-step rollout evaluation (only if not MAE mode)
            if not use_mae and T > rollout_horizon + 10:
                T_enc = T - rollout_horizon
                x_enc = x[:, :T_enc, :]
                x_target_rollout = x[:, T_enc:, :]
                
                # Extend task_vectors for rollout (if age context is enabled)
                rollout_task_vectors = None
                if task_vectors is not None:
                    # Extend task_vectors to match extended time horizon [B, T_enc+horizon, d_task]
                    T_extended = T_enc + rollout_horizon
                    rollout_task_vectors = torch.zeros(B, T_extended, task_vectors.shape[2], device=device, dtype=torch.float32)
                    # Copy age from first timestep and repeat across extended horizon
                    rollout_task_vectors[:, :, 0] = task_vectors[:, 0, 0].unsqueeze(1).expand(-1, T_extended)
                
                if is_distributed:
                    rollout_outputs = model.module.base_model.multistep_rollout(
                        x_enc, 
                        task_vectors=rollout_task_vectors, 
                        fc_matrix=fc_matrix,
                        horizon=rollout_horizon,
                        return_uncertainty=True
                    )
                else:
                    rollout_outputs = model.base_model.multistep_rollout(
                        x_enc, 
                        task_vectors=rollout_task_vectors, 
                        fc_matrix=fc_matrix,
                        horizon=rollout_horizon,
                        return_uncertainty=True
                    )
                
                mae = torch.abs(rollout_outputs['x_rollout'] - x_target_rollout).mean()
                rollout_maes.append(mae.item())
            
            if rank == 0 or not is_distributed:
                pbar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})
    
    n_batches = len(val_loader)
    
    # Average across GPUs if distributed
    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_forecast = total_forecast / n_batches
    
    if is_distributed:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_recon_tensor = torch.tensor(avg_recon, device=device)
        avg_forecast_tensor = torch.tensor(avg_forecast, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_recon_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_forecast_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
        avg_recon = avg_recon_tensor.item()
        avg_forecast = avg_forecast_tensor.item()
    
    # Compute calibration metrics and R² (only on rank 0)
    calibration = {}
    recon_r2 = 0.0
    if len(all_mu_recon) > 0 and (rank == 0 or not is_distributed):
        mu_cat = torch.cat(all_mu_recon, dim=0)
        target_cat = torch.cat(all_target, dim=0)
        
        # Assert shapes match (do not silently truncate - this would hide bugs)
        assert mu_cat.shape == target_cat.shape, (
            f"Shape mismatch in validation: pred={mu_cat.shape}, target={target_cat.shape}. "
            f"This indicates a bug in data collection or model output shapes."
        )
        
        target_flat = target_cat.reshape(-1)
        mu_flat = mu_cat.reshape(-1)
        
        # Compute R²
        sse = torch.sum((target_flat - mu_flat) ** 2)
        sst = torch.sum((target_flat - target_flat.mean()) ** 2)
        
        if sst > 1e-8:  # Avoid division by zero
            recon_r2 = 1.0 - (sse / sst)
            # Clamp R² to reasonable range (can be negative if model is worse than mean)
            recon_r2 = max(-10.0, min(1.0, recon_r2.item() if isinstance(recon_r2, torch.Tensor) else recon_r2))
        else:
            # If SST is too small, R² is undefined (target has no variance)
            recon_r2 = 0.0
            if rank == 0 or not is_distributed:
                print(f"  [Warning] SST too small for R² computation: {sst.item():.8f} (target may be constant)")
        
        # Compute calibration metrics (only if not MAE mode and sigma available)
        if not use_mae and len(all_sigma_recon) > 0:
            sigma_cat = torch.cat(all_sigma_recon, dim=0)
            if sigma_cat.shape == mu_cat.shape:
                calibration = compute_calibration_metrics(mu_cat, sigma_cat, target_cat)
                # Add simple sigma statistics for logging
                sigma_flat = sigma_cat.reshape(-1)
                calibration['sigma_mean'] = sigma_flat.mean().item()
                calibration['sigma_median'] = sigma_flat.median().item()
                calibration['sigma_min'] = sigma_flat.min().item()
                calibration['sigma_max'] = sigma_flat.max().item()
    
    # Initialize results dictionary
    results = {
        'total_loss': avg_loss,
        'recon_loss': avg_recon,  # Weighted loss (NLL or MSE)
        'forecast_loss': avg_forecast,
        'rollout_mae': np.mean(rollout_maes) if rollout_maes else 0.0,
        'recon_r2': recon_r2.item() if isinstance(recon_r2, torch.Tensor) else recon_r2,
        'recon_nll_raw': total_recon_nll_raw / n_batches if total_recon_nll_raw != 0.0 else 0.0,
        # Probe losses (will be populated if enabled)
        # Format: {module_name: {'mean': float, 'count': int}}
        'probe_losses': {
            k: {
                'mean': v['sum'] / (v['count'] + 1e-8) if v['count'] > 0 else None,
                'count': v['count']
            }
            for k, v in probe_loss_accum.items()
        } if probe_loss_accum else {},
        **calibration
    }
    
    # Add raw metrics (consistent with train_epoch)
    if len(all_mu_recon) > 0 and (rank == 0 or not is_distributed):
        mu_cat = torch.cat(all_mu_recon, dim=0)
        target_cat = torch.cat(all_target, dim=0)
        target_flat = target_cat.reshape(-1)
        mu_flat = mu_cat.reshape(-1)
        sse = torch.sum((target_flat - mu_flat) ** 2)
        sst = torch.sum((target_flat - target_flat.mean()) ** 2)
        
        # Raw MSE/MAE (same calculation as in compute_losses)
        n_samples = len(target_flat)
        results['recon_mse_raw'] = (sse / n_samples).item() if n_samples > 0 else 0.0
        results['recon_mae_raw'] = torch.mean(torch.abs(target_flat - mu_flat)).item()
        
        # Debug info for R² diagnostics
        results['r2_debug'] = {
            'sse': sse.item() if isinstance(sse, torch.Tensor) else sse,
            'sst': sst.item() if isinstance(sst, torch.Tensor) else sst,
            'target_mean': target_flat.mean().item(),
            'target_std': target_flat.std().item(),
            'pred_mean': mu_flat.mean().item(),
            'pred_std': mu_flat.std().item(),
            'mse': results['recon_mse_raw'],  # Use same value as recon_mse_raw
            'n_samples': n_samples
        }
    
    # Handle probe_losses DDP reduction if needed
    if is_distributed and 'probe_losses' in results:
        probe_losses = results.pop('probe_losses', {})
        # Re-initialize probe_losses in results before populating it
        results['probe_losses'] = {}
        if probe_losses:
            for module_name, info in probe_losses.items():
                if isinstance(info, dict) and 'mean' in info and 'count' in info:
                    mean_val = info.get('mean')
                    count_val = info.get('count', 0)
                    if mean_val is not None:
                        sum_val = mean_val * count_val
                        sum_tensor = torch.tensor(sum_val, device=device)
                        count_tensor = torch.tensor(float(count_val), device=device)
                        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                        results['probe_losses'][module_name] = {
                            'mean': (sum_tensor.item() / (count_tensor.item() + 1e-8)) if count_tensor.item() > 0 else None,
                            'count': int(count_tensor.item())
                        }
                    else:
                        results['probe_losses'][module_name] = {'mean': None, 'count': count_val}
    
    return results


def train(args):
    """Main training function with multi-GPU and MAE support"""
    
    # Setup distributed training
    ddp_info = setup_distributed()
    rank, world_size, local_rank = ddp_info
    
    is_distributed = rank is not None
    device = torch.device(f'cuda:{local_rank}' if is_distributed else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Detect dataset type from path or use provided dataset_type (do this early)
    dataset_name = args.dataset_type.upper() if args.dataset_type else ''
    if not dataset_name:
        # Check if COMBINED (has additional_data_dir)
        if args.additional_data_dir:
            dataset_name = 'COMBINED'
        # Try to detect from path
        elif args.npz_path:
            npz_name = Path(args.npz_path).stem.upper()
            if 'UKB' in npz_name or 'UKBIOBANK' in npz_name:
                dataset_name = 'UKB'
            else:
                dataset_name = 'HCPA'
        elif args.data_dir:
            data_dir_str = str(args.data_dir).upper()
            if 'UKB' in data_dir_str or 'UKBIOBANK' in data_dir_str:
                dataset_name = 'UKB'
            else:
                dataset_name = 'HCPA'
        else:
            dataset_name = 'HCPA'
    
    if rank == 0 or not is_distributed:
        print("Koopman Brain Model V2 - Training")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        if is_distributed:
            print(f"Multi-GPU Training: {world_size} GPUs")
            print(f"Effective batch size: {args.batch_size * world_size}")
        else:
            print("Single GPU Training")
        if args.use_mae:
            print("MAE Pre-training: Enabled")
        print(f"Device: {device}\n")
    
    # Output directory - store in results folder at project root
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    
    # Format date and time as YYYYMMDD_HHMMSS for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine model size based on latent_dim
    # Small: <= 256, Medium: 257-384, Large: > 384
    latent_dim = args.latent_dim
    if latent_dim <= 256:
        size_str = "small"
    elif latent_dim <= 384:
        size_str = "med"
    else:
        size_str = "large"
    
    # Keep dataset name as-is (HCPA, UKB)
    dataset_short = dataset_name.upper()
    
    # Create model name: HCPA-med-20241217_143052 or UKB-large-20241217_143052
    model_name = f"{dataset_short}-{size_str}-{timestamp}"
    
    # If output_dir is absolute, use it directly; otherwise use results folder
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir) / model_name
    else:
        # If output_dir is empty, use results root; otherwise use as subdirectory
        if args.output_dir:
            output_dir = results_dir / args.output_dir / model_name
        else:
            output_dir = results_dir / model_name
    
    if rank == 0 or not is_distributed:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}\n")
        
        # Create training log file (real-time updates)
        log_file_path = output_dir / 'training.log'
        log_file = open(log_file_path, 'w', encoding='utf-8')
        print(f"✓ Training log file: {log_file_path}")
        
        def log_print(*args, **kwargs):
            """Print to both console and log file with real-time flush"""
            # Handle both print() style and f-string style
            if args:
                # Convert all args to strings and join
                message = ' '.join(str(arg) for arg in args)
            else:
                message = ''
            
            # Print to console
            print(*args, **kwargs)
            
            # Write to log file (without color codes if any)
            log_file.write(message + '\n')
            log_file.flush()  # Real-time update
        
        # Create TensorBoard writer (if enabled)
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = output_dir / "tensorboard"
            tb_log_dir.mkdir(exist_ok=True)
            writer = SummaryWriter(str(tb_log_dir))
            log_print(f"✓ TensorBoard logs: {tb_log_dir}")
        except ImportError:
            writer = None
            log_print(f"⚠️  TensorBoard not available (pip install tensorboard)")
        
        # Save config (raw args)
        config = vars(args)
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    else:
        writer = None
        log_file = None
        log_print = print  # Fallback for non-rank-0 processes
    
    # Load data
    if rank == 0 or not is_distributed:
        print("Loading data...")
    
    # Use npz file if provided, otherwise use CSV files
    if args.npz_path:
        # Resolve npz path
        npz_path = Path(args.npz_path)
        if not npz_path.is_absolute():
            project_root = Path(__file__).parent.parent
            possible_paths = [
                npz_path,
                project_root / args.npz_path,
                Path.cwd() / args.npz_path,
            ]
            
            for path in possible_paths:
                if path.exists():
                    npz_path = path
                    break
            else:
                npz_path = project_root / args.npz_path
        
        if rank == 0 or not is_distributed:
            if not npz_path.exists():
                print(f"❌ Error: NPZ file does not exist: {npz_path}")
                cleanup_distributed()
                return
            print(f"✓ Using NPZ file: {npz_path}")
        
        # Create dataset from npz file (includes FC matrices) - HCPA only
        # Build kwargs conditionally to handle optional parameters
        npz_dataset_kwargs = {
            'npz_path': str(npz_path),
            'sequence_length': args.sequence_length,
            'stride': args.stride,
            'normalize': True,
            'task_filter': args.task_filter,
            'max_samples': args.max_files,  # Reuse max_files for max_samples
        }
        # Add fc_mode and hybrid_weight if they exist in args
        if hasattr(args, 'fc_mode') and args.fc_mode is not None:
            npz_dataset_kwargs['fc_mode'] = args.fc_mode
        if hasattr(args, 'hybrid_weight') and args.hybrid_weight is not None:
            npz_dataset_kwargs['hybrid_weight'] = args.hybrid_weight
        
        # Filter kwargs by signature (prevents TypeError)
        npz_dataset_kwargs = filter_kwargs_by_signature(
            HCPABOLDWithFCDataset, npz_dataset_kwargs,
            log_dropped=(rank == 0 or not is_distributed)
        )
        dataset = HCPABOLDWithFCDataset(**npz_dataset_kwargs)
        data_dir_path = npz_path.parent  # For FC path resolution later
    else:
        # Use CSV files (original method)
        if not args.data_dir and not args.npz_path:
            if rank == 0 or not is_distributed:
                print("❌ Error: Either --data_dir or --npz_path must be provided")
                print("   You can also use --config <config_name> to load a predefined configuration")
                print("   Available configs: small_model, large_model, sota_latent256, hcpa, ukb, etc.")
                print("   Example: python training/train_with_mae.py --config sota_latent256 --enable_module_probes")
                cleanup_distributed()
                return
        
        # Resolve data directory path (handle relative paths)
        data_dir_path = Path(args.data_dir)
        if not data_dir_path.is_absolute():
            project_root = Path(__file__).parent.parent
            possible_paths = [
                data_dir_path,
                project_root / args.data_dir,
                Path.cwd() / args.data_dir,
            ]
            
            for path in possible_paths:
                if path.exists():
                    data_dir_path = path
                    break
            else:
                data_dir_path = project_root / args.data_dir
        
        # Check if data directory exists
        if rank == 0 or not is_distributed:
            if not data_dir_path.exists():
                print(f"❌ Error: Data directory does not exist: {data_dir_path}")
                cleanup_distributed()
                return
            
            csv_files = list(data_dir_path.glob("*.csv"))
            if len(csv_files) == 0:
                print(f"⚠️  Warning: No CSV files found in {data_dir_path}")
                cleanup_distributed()
                return
            else:
                print(f"✓ Found {len(csv_files)} CSV files in {data_dir_path}")
        
        # Create dataset from CSV files
        if dataset_name == 'COMBINED':
            # Combined dataset: load both HCPA and UKB
            if not args.additional_data_dir:
                if rank == 0 or not is_distributed:
                    print("❌ Error: --additional_data_dir required for COMBINED dataset type")
                cleanup_distributed()
                return
            
            if rank == 0 or not is_distributed:
                print("Loading COMBINED dataset (HCPA + UKB)...")
            
            # Load primary dataset (HCPA)
            primary_data_dir_path = data_dir_path
            # For HCPA: try FC_RS first (resting state FC), then FC
            primary_fc_dir_rs = primary_data_dir_path.parent / 'FC_RS'
            primary_fc_dir = primary_data_dir_path.parent / 'FC'
            primary_fc_dir = primary_fc_dir_rs if primary_fc_dir_rs.exists() else primary_fc_dir
            if primary_fc_dir.exists():
                if rank == 0 or not is_distributed:
                    print(f"✓ Loading HCPA dataset with FC from: {primary_data_dir_path}")
                # Build kwargs conditionally to handle optional parameters
                hcpa_dataset_kwargs = {
                    'bold_dir': str(primary_data_dir_path),
                    'sequence_length': args.sequence_length,
                    'stride': args.stride,
                    'normalize': True,
                    'task_filter': args.task_filter,
                    'max_files': args.max_files,
                }
                # Try to add fc_mode and hybrid_weight if they exist in args
                # If dataset class doesn't support them, retry without them
                if hasattr(args, 'fc_mode') and args.fc_mode is not None:
                    hcpa_dataset_kwargs['fc_mode'] = args.fc_mode
                if hasattr(args, 'hybrid_weight') and args.hybrid_weight is not None:
                    hcpa_dataset_kwargs['hybrid_weight'] = args.hybrid_weight
                
                try:
                    dataset_hcpa = UKBiobankBOLDWithFCDataset(**hcpa_dataset_kwargs)
                except TypeError as e:
                    # If dataset class doesn't support fc_mode/hybrid_weight, retry without them
                    if 'fc_mode' in str(e) or 'hybrid_weight' in str(e) or 'unexpected keyword' in str(e):
                        if rank == 0 or not is_distributed:
                            print(f"⚠️  Dataset class doesn't support fc_mode/hybrid_weight, using defaults")
                        hcpa_dataset_kwargs.pop('fc_mode', None)
                        hcpa_dataset_kwargs.pop('hybrid_weight', None)
                        dataset_hcpa = UKBiobankBOLDWithFCDataset(**hcpa_dataset_kwargs)
                    else:
                        raise
            else:
                if rank == 0 or not is_distributed:
                    print(f"✓ Loading HCPA dataset (no FC) from: {primary_data_dir_path}")
                dataset_hcpa = HCPABOLDDataset(
                    data_dir=str(primary_data_dir_path),
                    sequence_length=args.sequence_length,
                    stride=args.stride,
                    normalize=True,
                    task_filter=args.task_filter,
                    max_files=args.max_files
                )
            
            # Load secondary dataset (UKB)
            secondary_data_dir_path = Path(args.additional_data_dir)
            if not secondary_data_dir_path.is_absolute():
                project_root = Path(__file__).parent.parent
                possible_paths = [
                    secondary_data_dir_path,
                    project_root / args.additional_data_dir,
                    Path.cwd() / args.additional_data_dir,
                ]
                for path in possible_paths:
                    if path.exists():
                        secondary_data_dir_path = path
                        break
                else:
                    secondary_data_dir_path = project_root / args.additional_data_dir
            
            if not secondary_data_dir_path.exists():
                if rank == 0 or not is_distributed:
                    print(f"❌ Error: Additional data directory does not exist: {secondary_data_dir_path}")
                cleanup_distributed()
                return
            
            # For UKB: use FC directory
            secondary_fc_dir = secondary_data_dir_path.parent / 'FC'
            if args.additional_fc_matrix_path:
                secondary_fc_dir = Path(args.additional_fc_matrix_path)
            
            if secondary_fc_dir.exists():
                if rank == 0 or not is_distributed:
                    print(f"✓ Loading UKB dataset with FC from: {secondary_data_dir_path}")
                # Build kwargs conditionally to handle optional parameters
                ukb_dataset_kwargs = {
                    'bold_dir': str(secondary_data_dir_path),
                    'fc_dir': str(secondary_fc_dir),  # Explicitly pass FC directory
                    'sequence_length': args.sequence_length,
                    'stride': args.stride,
                    'normalize': True,
                    'task_filter': args.task_filter,
                    'max_files': args.max_files,
                }
                # Add fc_mode and hybrid_weight if they exist in args
                if hasattr(args, 'fc_mode') and args.fc_mode is not None:
                    ukb_dataset_kwargs['fc_mode'] = args.fc_mode
                if hasattr(args, 'hybrid_weight') and args.hybrid_weight is not None:
                    ukb_dataset_kwargs['hybrid_weight'] = args.hybrid_weight
                
                # Filter kwargs by signature (prevents TypeError)
                ukb_dataset_kwargs = filter_kwargs_by_signature(
                    UKBiobankBOLDWithFCDataset, ukb_dataset_kwargs,
                    log_dropped=(rank == 0 or not is_distributed)
                )
                dataset_ukb = UKBiobankBOLDWithFCDataset(**ukb_dataset_kwargs)
            else:
                if rank == 0 or not is_distributed:
                    print(f"✓ Loading UKB dataset (no FC) from: {secondary_data_dir_path}")
                dataset_ukb = UKBiobankBOLDDataset(
                    data_dir=str(secondary_data_dir_path),
                    sequence_length=args.sequence_length,
                    stride=args.stride,
                    normalize=True,
                    task_filter=args.task_filter,
                    max_files=args.max_files
                )
            
            # Load third dataset (ABCD) if provided
            datasets_to_combine = [dataset_hcpa, dataset_ukb]
            dataset_names = ["HCPA", "UKB"]
            
            if args.third_data_dir:
                if rank == 0 or not is_distributed:
                    print("Loading third dataset (ABCD)...")
                
                third_data_dir_path = Path(args.third_data_dir)
                if not third_data_dir_path.is_absolute():
                    project_root = Path(__file__).parent.parent
                    possible_paths = [
                        third_data_dir_path,
                        project_root / args.third_data_dir,
                        Path.cwd() / args.third_data_dir,
                    ]
                    for path in possible_paths:
                        if path.exists():
                            third_data_dir_path = path
                            break
                    else:
                        third_data_dir_path = project_root / args.third_data_dir
                
                if not third_data_dir_path.exists():
                    if rank == 0 or not is_distributed:
                        print(f"⚠️  Warning: Third data directory does not exist: {third_data_dir_path}")
                        print(f"   Continuing with 2 datasets (HCPA + UKB)")
                else:
                    # For ABCD: FC directory is at abcd/FC (not abcd/AAL_116/FC)
                    third_fc_dir = third_data_dir_path.parent.parent / 'FC' if 'AAL_116' in str(third_data_dir_path) else third_data_dir_path.parent / 'FC'
                    if args.third_fc_matrix_path:
                        third_fc_dir = Path(args.third_fc_matrix_path)
                    
                    # Use UKBiobankBOLDWithFCDataset for ABCD (same structure as HCPA/UKB with bold_dir/fc_dir)
                    if third_fc_dir.exists():
                        if rank == 0 or not is_distributed:
                            print(f"✓ Loading ABCD dataset with FC from: {third_data_dir_path}")
                            print(f"  FC directory: {third_fc_dir}")
                        # Use separate max_files for third dataset if available (for balancing)
                        max_files_abcd = getattr(args, 'max_files_third_dataset', args.max_files)
                        abcd_dataset_kwargs = {
                            'bold_dir': str(third_data_dir_path),
                            'fc_dir': str(third_fc_dir),  # Explicitly pass FC directory
                            'sequence_length': args.sequence_length,
                            'stride': args.stride,
                            'normalize': True,
                            'task_filter': args.task_filter,
                            'max_files': max_files_abcd,  # Use separate limit for ABCD if specified
                        }
                        if hasattr(args, 'fc_mode') and args.fc_mode is not None:
                            abcd_dataset_kwargs['fc_mode'] = args.fc_mode
                        if hasattr(args, 'hybrid_weight') and args.hybrid_weight is not None:
                            abcd_dataset_kwargs['hybrid_weight'] = args.hybrid_weight
                        
                        try:
                            dataset_abcd = UKBiobankBOLDWithFCDataset(**abcd_dataset_kwargs)
                        except TypeError as e:
                            # If dataset class doesn't support fc_mode/hybrid_weight, retry without them
                            if 'fc_mode' in str(e) or 'hybrid_weight' in str(e) or 'unexpected keyword' in str(e):
                                if rank == 0 or not is_distributed:
                                    print(f"⚠️  Dataset class doesn't support fc_mode/hybrid_weight, using defaults")
                                abcd_dataset_kwargs.pop('fc_mode', None)
                                abcd_dataset_kwargs.pop('hybrid_weight', None)
                                dataset_abcd = UKBiobankBOLDWithFCDataset(**abcd_dataset_kwargs)
                            else:
                                raise
                    else:
                        if rank == 0 or not is_distributed:
                            print(f"✓ Loading ABCD dataset (no FC) from: {third_data_dir_path}")
                        # Use separate max_files for third dataset if available (for balancing)
                        max_files_abcd = getattr(args, 'max_files_third_dataset', args.max_files)
                        dataset_abcd = HCPABOLDDataset(
                            data_dir=str(third_data_dir_path),
                            sequence_length=args.sequence_length,
                            stride=args.stride,
                            normalize=True,
                            task_filter=args.task_filter,
                            max_files=max_files_abcd  # Use separate limit for ABCD if specified
                        )
                    
                    datasets_to_combine.append(dataset_abcd)
                    dataset_names.append("ABCD")
            
            # Combine datasets with metadata support
            dataset = CombinedDatasetWithMetadata(datasets_to_combine)
            if rank == 0 or not is_distributed:
                dataset_info = " + ".join([f"{len(d)} {name}" for d, name in zip(datasets_to_combine, dataset_names)])
                print(f"✓ Combined dataset: {dataset_info} = {len(dataset)} total samples")
            
        elif dataset_name == 'UKB':
            # Check if FC directory exists (../FC relative to BOLD directory)
            fc_dir = data_dir_path.parent / 'FC'
            if fc_dir.exists():
                if rank == 0 or not is_distributed:
                    print(f"✓ Found FC directory: {fc_dir}")
                # Build kwargs conditionally to handle optional parameters
                dataset_kwargs = {
                    'bold_dir': str(data_dir_path),
                    'sequence_length': args.sequence_length,
                    'stride': args.stride,
                    'normalize': True,
                    'task_filter': args.task_filter,
                    'max_files': args.max_files,
                }
                # Add fc_mode and hybrid_weight if they exist in args
                if hasattr(args, 'fc_mode') and args.fc_mode is not None:
                    dataset_kwargs['fc_mode'] = args.fc_mode
                if hasattr(args, 'hybrid_weight') and args.hybrid_weight is not None:
                    dataset_kwargs['hybrid_weight'] = args.hybrid_weight
                
                # Filter kwargs by signature (prevents TypeError)
                dataset_kwargs = filter_kwargs_by_signature(
                    UKBiobankBOLDWithFCDataset, dataset_kwargs,
                    log_dropped=(rank == 0 or not is_distributed)
                )
                dataset = UKBiobankBOLDWithFCDataset(**dataset_kwargs)
            else:
                if rank == 0 or not is_distributed:
                    print(f"⚠️  No FC directory found, will compute on-the-fly")
                dataset = UKBiobankBOLDDataset(
                    data_dir=str(data_dir_path),
                    sequence_length=args.sequence_length,
                    stride=args.stride,
                    normalize=True,
                    task_filter=args.task_filter,
                    max_files=args.max_files
                )
        else:  # HCPA
            # Check if FC directory exists (../FC relative to BOLD directory)
            fc_dir = data_dir_path.parent / 'FC'
            if fc_dir.exists():
                if rank == 0 or not is_distributed:
                    print(f"✓ Found FC directory: {fc_dir}")
                # Use UKBiobankBOLDWithFCDataset (same structure, supports task_filter)
                # Build kwargs conditionally to handle optional parameters
                dataset_kwargs = {
                    'bold_dir': str(data_dir_path),
                    'sequence_length': args.sequence_length,
                    'stride': args.stride,
                    'normalize': True,
                    'task_filter': args.task_filter,
                    'max_files': args.max_files,
                }
                # Add fc_mode and hybrid_weight if they exist in args
                if hasattr(args, 'fc_mode') and args.fc_mode is not None:
                    dataset_kwargs['fc_mode'] = args.fc_mode
                if hasattr(args, 'hybrid_weight') and args.hybrid_weight is not None:
                    dataset_kwargs['hybrid_weight'] = args.hybrid_weight
                
                # Filter kwargs by signature (prevents TypeError)
                dataset_kwargs = filter_kwargs_by_signature(
                    UKBiobankBOLDWithFCDataset, dataset_kwargs,
                    log_dropped=(rank == 0 or not is_distributed)
                )
                dataset = UKBiobankBOLDWithFCDataset(**dataset_kwargs)
            else:
                if rank == 0 or not is_distributed:
                    print(f"⚠️  No FC directory found, will compute on-the-fly")
                dataset = HCPABOLDDataset(
                    data_dir=str(data_dir_path),
                    sequence_length=args.sequence_length,
                    stride=args.stride,
                    normalize=True,
                    task_filter=args.task_filter,
                    max_files=args.max_files
                )
    
    # ---- Subject-wise split: ensure same subject never appears in both train and val ----
    def subject_wise_split(ds, train_split: float, seed: int = 42):
        """
        Split segments into train/val based on subject id.
        Expects ds.metadata[i]['subject'] to exist for each segment.
        """
        if is_distributed:
            print(f"[RANK {rank}] Starting subject_wise_split...", flush=True)
        
        if not hasattr(ds, "metadata"):
            raise RuntimeError("Dataset has no 'metadata' attribute; cannot perform subject-wise split.")
        
        meta = ds.metadata
        if is_distributed:
            print(f"[RANK {rank}] Checking metadata length: {len(meta)} vs dataset length: {len(ds)}", flush=True)
        
        if len(meta) != len(ds):
            raise RuntimeError(
                f"metadata length ({len(meta)}) does not match dataset length ({len(ds)}); "
                "cannot perform subject-wise split."
            )
        
        if is_distributed:
            print(f"[RANK {rank}] Extracting subject IDs from {len(meta)} metadata entries...", flush=True)
        
        subject_ids = [str(m.get("subject", "")) for m in meta]
        
        if is_distributed:
            print(f"[RANK {rank}] Checking for empty subject IDs...", flush=True)
        
        if any(sid == "" for sid in subject_ids):
            raise RuntimeError("Some segments have empty subject id; please ensure metadata['subject'] is set.")
        
        if is_distributed:
            print(f"[RANK {rank}] Finding unique subjects...", flush=True)
        
        unique_subj = sorted(set(subject_ids))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_subj)
        
        n_train_subj = int(len(unique_subj) * train_split)
        train_subj = set(unique_subj[:n_train_subj])
        val_subj = set(unique_subj[n_train_subj:])
        
        if is_distributed:
            print(f"[RANK {rank}] Assigning indices to train/val sets (this may take a moment for large datasets)...", flush=True)
        
        # Assign indices - use list comprehension with progress for very large datasets
        # For datasets with >50k segments, this can be slow
        if len(subject_ids) > 50000:
            # Use a more efficient approach: create a mapping first
            train_indices = []
            val_indices = []
            for i, sid in enumerate(subject_ids):
                if sid in train_subj:
                    train_indices.append(i)
                elif sid in val_subj:
                    val_indices.append(i)
                if (i + 1) % 10000 == 0 and is_distributed:
                    print(f"[RANK {rank}] Processed {i+1}/{len(subject_ids)} segments...", flush=True)
        else:
            train_indices = [i for i, sid in enumerate(subject_ids) if sid in train_subj]
            val_indices = [i for i, sid in enumerate(subject_ids) if sid in val_subj]
        
        if is_distributed:
            print(f"[RANK {rank}] Created train/val indices. Train: {len(train_indices)}, Val: {len(val_indices)}", flush=True)
        
        overlap = train_subj.intersection(val_subj)
        assert len(overlap) == 0, f"Subject overlap detected between train/val: {overlap}"
        
        if is_distributed:
            print(f"[RANK {rank}] Creating Subset objects...", flush=True)
        
        train_ds = Subset(ds, train_indices)
        val_ds = Subset(ds, val_indices)
        
        if is_distributed:
            print(f"[RANK {rank}] Subject-wise split completed!", flush=True)
        
        return train_ds, val_ds, train_subj, val_subj, train_indices, val_indices
    
    split_seed = getattr(args, 'split_seed', 42)
    
    # Add debug output from all ranks to identify stuck processes
    if is_distributed:
        print(f"[RANK {rank}] Starting subject-wise split...", flush=True)
    
    try:
        train_dataset, val_dataset, train_subj, val_subj, train_idx, val_idx = subject_wise_split(
            dataset, args.train_split, seed=split_seed
        )
        
        if is_distributed:
            print(f"[RANK {rank}] Subject-wise split completed. Train: {len(train_idx)}, Val: {len(val_idx)}", flush=True)
    except Exception as e:
        print(f"[RANK {rank}] ERROR during subject-wise split: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    if rank == 0 or not is_distributed:
        print(f"✓ Subject-wise split:")
        print(f"  Train subjects: {len(train_subj)}, Val subjects: {len(val_subj)}, Overlap: 0")
        print(f"  Train segments: {len(train_idx)}, Val segments: {len(val_idx)}")
        # Show a few subject ids for sanity check
        train_subj_list = sorted(list(train_subj))[:5]
        val_subj_list = sorted(list(val_subj))[:5]
        print(f"  Sample train subjects: {train_subj_list}")
        print(f"  Sample val subjects:   {val_subj_list}")
    
    # Synchronize all ranks after data loading (critical for multi-GPU)
    if is_distributed:
        print(f"[RANK {rank}] Reaching barrier after data loading...", flush=True)
        if rank == 0:
            print("  Synchronizing all ranks after data loading...")
        try:
            # Barrier uses the timeout set during process group initialization
            # Note: dist.barrier() does NOT accept timeout parameter - timeout is set during init_process_group
            safe_barrier()
        except Exception as e:
            print(f"[RANK {rank}] ERROR at barrier: {e}", flush=True)
            raise
        print(f"[RANK {rank}] Passed barrier", flush=True)
        if rank == 0:
            print("  ✓ All ranks synchronized")
    
    # Create distributed samplers if using DDP
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    # Create data loaders
    # For multi-GPU, reduce num_workers to avoid I/O contention and sync issues
    effective_num_workers = args.num_workers
    if is_distributed and args.num_workers > 2:
        effective_num_workers = 2  # Reduce workers for multi-GPU stability
        if rank == 0:
            print(f"  ⚠️  Reduced num_workers from {args.num_workers} to {effective_num_workers} for multi-GPU stability")
    
    def safe_collate(batch):
        """Collate batch while tolerating missing metadata keys across datasets."""
        if not batch:
            return batch
        xs, metas = zip(*batch)
        x_tensor = torch.stack(xs, dim=0)
        if not metas or not isinstance(metas[0], dict):
            return x_tensor, list(metas)

        keys = set()
        for meta in metas:
            if isinstance(meta, dict):
                keys.update(meta.keys())

        collated = {}
        for key in keys:
            values = [m.get(key) if isinstance(m, dict) else None for m in metas]
            if key == 'fc_matrix':
                if all(isinstance(v, torch.Tensor) for v in values if v is not None) and all(v is not None for v in values):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    # Skip fc_matrix if any sample lacks it
                    continue
            else:
                collated[key] = values

        return x_tensor, collated

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
        collate_fn=safe_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
        collate_fn=safe_collate
    )
    
    # Synchronize all ranks after DataLoader creation
    if is_distributed:
        print(f"[RANK {rank}] Reaching barrier after DataLoader creation...", flush=True)
        try:
            # Barrier uses the timeout set during process group initialization
            # Note: dist.barrier() does NOT accept timeout parameter - timeout is set during init_process_group
            safe_barrier()
        except Exception as e:
            print(f"[RANK {rank}] ERROR at DataLoader barrier: {e}", flush=True)
            raise
        print(f"[RANK {rank}] Passed DataLoader barrier", flush=True)
        if rank == 0:
            print("  ✓ DataLoaders created and synchronized")
    
    if rank == 0 or not is_distributed:
        print(f"Train: {len(train_loader)} batches per GPU, Val: {len(val_loader)} batches per GPU")
        if is_distributed:
            print(f"Total train batches: {len(train_loader) * world_size}\n")
        else:
            print()
    
    # Load FC matrix (respecting task filter)
    # Note: If using npz_path, FC matrices are already loaded per-sample in the dataset
    # This is for a global FC matrix (optional, for model initialization)
    fc_matrix = None
    if args.fc_matrix_path:
        if rank == 0 or not is_distributed:
            print(f"Loading FC matrix from {args.fc_matrix_path}...")
        
        # Resolve FC matrix path
        fc_path = Path(args.fc_matrix_path)
        if not fc_path.is_absolute():
            project_root = Path(__file__).parent.parent
            possible_fc_paths = [
                fc_path,
                project_root / args.fc_matrix_path,
                Path.cwd() / args.fc_matrix_path,
                data_dir_path.parent / args.fc_matrix_path,  # Relative to data directory parent
            ]
            
            for path in possible_fc_paths:
                if path.exists():
                    fc_path = path
                    break
            else:
                fc_path = project_root / args.fc_matrix_path
        
        # Check if FC matrix path matches task filter
        if args.task_filter and len(args.task_filter) > 0:
            fc_path_str = str(fc_path).upper()
            task_filter_upper = [t.upper() for t in args.task_filter]
            
            # Check if FC path contains any of the task names
            matches_task = any(task in fc_path_str for task in task_filter_upper)
            
            if not matches_task:
                # Try to find task-specific FC matrix
                if fc_path.is_file():
                    # If it's a file, check if we can find a task-specific version
                    fc_dir = fc_path.parent
                    fc_name = fc_path.stem
                    fc_ext = fc_path.suffix
                    
                    # Try to find task-specific FC file
                    task_fc_found = False
                    for task in args.task_filter:
                        task_fc_path = fc_dir / f"{fc_name}_{task}{fc_ext}"
                        if task_fc_path.exists():
                            fc_path = task_fc_path
                            task_fc_found = True
                            if rank == 0 or not is_distributed:
                                print(f"  → Found task-specific FC: {task_fc_path.name}")
                            break
                    
                    if not task_fc_found:
                        if rank == 0 or not is_distributed:
                            print(f"⚠️  Warning: FC matrix path '{fc_path.name}' doesn't match task filter {args.task_filter}")
                            print(f"   Looking for FC matrices matching: {args.task_filter}")
                            print(f"   Will try to load anyway, but verify this is correct...")
                elif fc_path.is_dir():
                    # If it's a directory, look for task-specific files
                    task_fc_files = []
                    for task in args.task_filter:
                        # Try common patterns
                        patterns = [
                            f"*{task}*.npy",
                            f"*{task.upper()}*.npy",
                            f"*{task.lower()}*.npy",
                            f"FC_{task}*.npy",
                            f"fc_{task}*.npy",
                        ]
                        for pattern in patterns:
                            matches = list(fc_path.glob(pattern))
                            if matches:
                                task_fc_files.extend(matches)
                                break
                    
                    if task_fc_files:
                        # Use the first matching file
                        fc_path = task_fc_files[0]
                        if rank == 0 or not is_distributed:
                            print(f"  → Found task-specific FC: {fc_path.name}")
                    elif rank == 0 or not is_distributed:
                        print(f"⚠️  Warning: No task-specific FC files found in {fc_path}")
                        print(f"   Looking for files matching: {args.task_filter}")
        
        if fc_path.exists() and fc_path.is_file():
            try:
                fc_data = np.load(str(fc_path))
                fc_matrix = torch.from_numpy(fc_data).float().to(device)
                if rank == 0 or not is_distributed:
                    print(f"✓ FC matrix loaded: {fc_matrix.shape}")
                    if args.task_filter:
                        print(f"  Task filter: {args.task_filter}")
                    print()
            except Exception as e:
                if rank == 0 or not is_distributed:
                    print(f"❌ Error loading FC matrix: {e}\n")
        else:
            if rank == 0 or not is_distributed:
                print(f"⚠️  Warning: FC matrix file not found: {fc_path}\n")
    
    # Create model
    if rank == 0 or not is_distributed:
        print("Creating Model V2...")
    
    # Mamba is deprecated - always use transformer
    if args.temporal_model == 'mamba':
        if rank == 0 or not is_distributed:
            print("⚠️  Mamba temporal model is deprecated. Using Transformer instead.")
        args.temporal_model = 'transformer'
    
    # Create base model
    task_type = getattr(args, 'task_type', 'classification')  # Default to classification if not specified
    base_model = BrainMech(
        n_roi=args.n_roi,
        d_latent=args.latent_dim,
        d_control=args.control_dim,
        d_episodic=args.episodic_dim,
        d_context=args.context_dim,
        d_task=args.task_dim,
        n_classes=args.n_classes,
        task_type=task_type,
        graph_type=args.graph_type,
        n_graph_layers=args.n_graph_layers,
        cheb_k=args.cheb_k,
        drop_edge_p=args.drop_edge_p,
        n_temporal_layers=args.n_temporal_layers,
        n_heads=args.n_heads,
        koopman_rank=args.koopman_rank,
        spectral_radius=args.target_spectral_radius,
        use_sde=args.use_sde,
        diffusion_mode=args.diffusion_mode,
        init_noise_std=args.init_noise_std,
        uncertainty_mode=args.uncertainty_mode,
        fc_matrix=fc_matrix,
        dropout=args.dropout,
        # Ablation study flags
        use_episodic_feedback=args.use_episodic_feedback,
        use_task_context=args.use_task_context,
        use_control_fusion=args.use_control_fusion,
        use_temporal_transformer=args.use_temporal_transformer,
        use_cognitive_readout=args.use_cognitive_readout,
        use_spatial_encoder=args.use_spatial_encoder,
        # Mamba architecture flags
        temporal_model=args.temporal_model,
        use_dual_path_mamba=args.use_dual_path_mamba,
        # Ablation flags
        temporal_fc_mode=getattr(args, 'temporal_fc_mode', False),
        # Decoder / sigma calibration flags
        detach_sigma_head=args.detach_sigma_head,
        sigma_calibration=args.sigma_calibration,
        sigma_max_time_steps=args.sigma_max_time_steps,
    ).to(device)
    
    # Wrap with MAE if enabled
    if args.use_mae:
        model = MaskedAutoencoderWrapper(
            base_model=base_model,
            mask_ratio=args.mask_ratio,
            mask_strategy=args.mask_strategy,
            mask_type=args.mask_type,
            use_random_masking=args.use_random_masking,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            use_curriculum=args.use_curriculum,
            spatial_mask_ratio=args.spatial_mask_ratio,
            temporal_mask_ratio=args.temporal_mask_ratio
        ).to(device)
    else:
        model = base_model
    
    # Check CUDA memory before DDP wrapping
    if is_distributed:
        torch.cuda.synchronize(device)
        if rank == 0:
            print(f"\n[DEBUG] CUDA memory before DDP wrapping:")
            for i in range(world_size):
                if i == 0:
                    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    print(f"  GPU {local_rank}: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    # Wrap with DDP if distributed
    if is_distributed:
        try:
            # Synchronize CUDA before DDP (error checking removed - not available in all PyTorch versions)
            torch.cuda.synchronize(device)
            
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
            
            # Synchronize CUDA after DDP
            torch.cuda.synchronize(device)
            
            # Synchronize all ranks after DDP
            safe_barrier()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"[RANK {rank}] CUDA OOM or error during DDP wrapping: {e}")
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"[RANK {rank}] Unexpected error during DDP wrapping: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        model_for_stats = model.module.base_model if args.use_mae else model.module
    else:
        model_for_stats = model.base_model if args.use_mae else model
    
    # Load age from demographics CSV if provided (for age context)
    age_dict = {}  # subject_id -> age mapping
    use_age_context = getattr(args, 'use_age_context', False)
    age_min = getattr(args, 'age_min', 8.0)
    age_max = getattr(args, 'age_max', 100.0)
    
    if use_age_context:
        demographics_path = getattr(args, 'demographics_csv', None)
        if demographics_path and Path(demographics_path).exists():
            if rank == 0 or not is_distributed:
                print(f"\nLoading age from demographics: {demographics_path}")
            try:
                import pandas as pd
                demo_df = pd.read_csv(demographics_path)
                
                # Try to find subject and age columns
                subject_col = None
                age_col = None
                
                for col in ['subject', 'Subject', 'subject_id', 'Subject', 'src_subject_id', 'Image Data ID']:
                    if col in demo_df.columns:
                        subject_col = col
                        break
                
                for col in ['age', 'Age', 'AGE', 'age_at_scan', 'Age_at_scan']:
                    if col in demo_df.columns:
                        age_col = col
                        break
                
                if subject_col and age_col:
                    for _, row in demo_df.iterrows():
                        subj_id = str(row[subject_col]).strip()
                        age_val = row[age_col]
                        if pd.notna(age_val):
                            age_dict[subj_id] = float(age_val)
                    
                    if rank == 0 or not is_distributed:
                        print(f"  ✓ Loaded {len(age_dict)} age entries")
                        if len(age_dict) > 0:
                            ages = list(age_dict.values())
                            print(f"  Age range: [{min(ages):.1f}, {max(ages):.1f}] years")
                            print(f"  Age will be included in task_vectors (normalized to [0, 1])")
                else:
                    if rank == 0 or not is_distributed:
                        print(f"  ⚠️  Could not find subject/age columns in demographics CSV")
            except Exception as e:
                if rank == 0 or not is_distributed:
                    print(f"  ⚠️  Error loading demographics: {e}")
        else:
            if rank == 0 or not is_distributed:
                print(f"  ⚠️  Demographics CSV not provided or not found")
    else:
        if rank == 0 or not is_distributed:
            print(f"\n  Age context disabled (use --use_age_context to enable)")
    
    # Print parameter stats
    stats = model_for_stats.get_param_stats()
    if rank == 0 or not is_distributed:
        print(f"✓ Model created")
        print(f"  Total parameters: {stats['total_params']:,}")
        print(f"  Koopman K reduction: {stats['k_reduction']:.2f}x")
        print(f"  Koopman C reduction: {stats['c_reduction']:.2f}x")
        print(f"  Overall core reduction: {stats['total_dense'] / stats['total_structured']:.2f}x")
        
        # Temporal backend visibility
        temporal_backend_fn = getattr(model_for_stats, "get_temporal_backend", None)
        if callable(temporal_backend_fn):
            backend_str = model_for_stats.get_temporal_backend()
        else:
            backend_str = args.temporal_model if args.use_temporal_transformer else "identity"
        print(f"  Temporal backend: {backend_str}")
        
        # Print key dimensions (A4 requirement)
        print(f"\n【关键维度信息 (Key Dimensions)】")
        print(f"  d_latent: {args.latent_dim}")
        print(f"  d_control: {args.control_dim}")
        print(f"  d_episodic: {args.episodic_dim}")
        print(f"  d_context: {args.context_dim}")
        print(f"  n_heads: {args.n_heads}")
        print(f"  koopman_rank: {args.koopman_rank}")
        print(f"  n_roi: {args.n_roi}")
        print(f"  Validation: d_latent % n_heads = {args.latent_dim % args.n_heads} (should be 0)")
        print(f"  Validation: koopman_rank <= d_latent = {args.koopman_rank <= args.latent_dim} (should be True)")
        
        # Print detailed module configuration
        print(f"\n【模块配置详情 (Module Configuration Details)】")
        print(f"{'='*70}")
        
        # 1. Spatial Encoder
        print(f"\n1. Spatial Encoder (Graph Neural Network)")
        if args.use_spatial_encoder:
            print(f"   Status: ✓ ENABLED")
            print(f"   Type: GraphSpatialEncoderReal")
            print(f"   Graph Mode: {args.graph_type.upper()} ({'Chebyshev Spectral' if args.graph_type == 'cheb' else 'Graph Attention'})")
            print(f"   Layers: {args.n_graph_layers}")
            if args.graph_type == 'cheb':
                print(f"   ChebK (polynomial order): {args.cheb_k}")
            elif args.graph_type == 'gat':
                print(f"   Attention Heads: {args.n_heads}")
            print(f"   DropEdge Probability: {args.drop_edge_p}")
            print(f"   FC Matrix: {'Provided' if fc_matrix is not None else 'Learnable'}")
        else:
            print(f"   Status: ✗ DISABLED (using Linear projection)")
        
        # 2. Temporal Model
        print(f"\n2. Temporal Model")
        if args.use_temporal_transformer:
            print(f"   Status: ✓ ENABLED")
            print(f"   Backend: {args.temporal_model.upper()}")
            if args.temporal_model == 'transformer':
                print(f"   Type: TemporalTransformer")
                print(f"   Layers: {args.n_temporal_layers}")
                print(f"   Attention Heads: {args.n_heads}")
            elif args.temporal_model == 'mamba':
                print(f"   Type: {'TemporalMambaBi (dual-path)' if args.use_dual_path_mamba else 'TemporalMamba (causal)'}")
                print(f"   Dual Path: {'Yes (causal + bidirectional)' if args.use_dual_path_mamba else 'No (causal only)'}")
            print(f"   Dropout: {args.dropout}")
        else:
            print(f"   Status: ✗ DISABLED (using Identity)")
        
        # 3. Control System
        print(f"\n3. Control System")
        print(f"   3.1 Episodic Feedback")
        if args.use_episodic_feedback:
            print(f"      Status: ✓ ENABLED")
            print(f"      Type: EpisodicFeedbackTransformer")
            print(f"      Dimension: {args.episodic_dim}")
            print(f"      Attention Heads: {args.n_heads // 2}")
        else:
            print(f"      Status: ✗ DISABLED")
        
        print(f"   3.2 Task Context Encoder")
        if args.use_task_context:
            print(f"      Status: ✓ ENABLED")
            print(f"      Type: TaskContextEncoder")
            print(f"      Task Dim: {args.task_dim}")
            print(f"      Context Dim: {args.context_dim}")
        else:
            print(f"      Status: ✗ DISABLED")
        
        print(f"   3.3 Control Fusion")
        if args.use_control_fusion:
            print(f"      Status: ✓ ENABLED")
            print(f"      Type: ControlFusion")
            print(f"      Input: episodic ({args.episodic_dim}) + context ({args.context_dim})")
            print(f"      Output: control ({args.control_dim})")
        else:
            print(f"      Status: ✗ DISABLED")
        
        # 4. Koopman Core
        print(f"\n4. Koopman Core")
        print(f"   Status: ✓ ENABLED (always)")
        print(f"   Type: DPLRKoopmanCore (Diagonal Plus Low-Rank)")
        print(f"   Latent Dim: {args.latent_dim}")
        print(f"   Control Dim: {args.control_dim}")
        print(f"   Rank: {args.koopman_rank} (low-rank decomposition)")
        print(f"   Spectral Radius: {args.target_spectral_radius}")
        print(f"   SDE (Stochastic): {'✓ ENABLED' if args.use_sde else '✗ DISABLED'}")
        if args.use_sde:
            print(f"   Diffusion Mode: {args.diffusion_mode}")
            if args.diffusion_mode == 'fixed':
                print(f"   Fixed Noise Std: {args.init_noise_std}")
            elif args.diffusion_mode == 'learned':
                print(f"   Learned State-Dependent Diffusion: σ(z,u) = softplus(MLP([z,u]))")
        
        # 5. Decoder
        print(f"\n5. Decoder")
        print(f"   Status: ✓ ENABLED (always)")
        print(f"   Type: StochasticDecoder")
        print(f"   Uncertainty Mode: {args.uncertainty_mode}")
        if args.uncertainty_mode == 'heteroscedastic':
            print(f"   Output: (μ_x, σ_x) - state-dependent uncertainty")
        elif args.uncertainty_mode == 'homoscedastic':
            print(f"   Output: (μ_x, σ) - global uncertainty")
        else:
            print(f"   Output: μ_x only (no uncertainty)")
        print(f"   Sigma Calibration: {args.sigma_calibration}")
        print(f"   Detach Sigma Head: {args.detach_sigma_head}")
        
        # 6. Cognitive Readout
        print(f"\n6. Cognitive Readout")
        if args.use_cognitive_readout:
            print(f"   Status: ✓ ENABLED")
            print(f"   Type: CognitiveStateReadoutStochastic")
            task_type = getattr(args, 'task_type', 'classification')  # Default to classification
            print(f"   Task Type: {task_type}")
            print(f"   Classes: {args.n_classes}")
            print(f"   MC Samples: 10 (for uncertainty)")
        else:
            print(f"   Status: ✗ DISABLED")
        
        # 7. MAE Wrapper (if enabled)
        if args.use_mae:
            print(f"\n7. MAE Pre-training Wrapper")
            print(f"   Status: ✓ ENABLED")
            print(f"   Type: MaskedAutoencoderWrapper")
            print(f"   Mask Strategy: {args.mask_strategy}")
            print(f"   Mask Type: {args.mask_type}")
            print(f"   Base Mask Ratio: {args.mask_ratio}")
            if args.fixed_mask_ratio is not None:
                print(f"   Fixed Mask Ratio: {args.fixed_mask_ratio} (curriculum disabled)")
            else:
                print(f"   Random Masking: {'Yes' if args.use_random_masking else 'No'}")
                if args.use_random_masking:
                    print(f"   Mask Range: [{args.min_mask_ratio}, {args.max_mask_ratio}]")
                print(f"   Curriculum Learning: {'Yes' if args.use_curriculum else 'No'}")
            if args.mask_type == 'both':
                if args.spatial_mask_ratio is not None:
                    print(f"   Spatial Mask Ratio: {args.spatial_mask_ratio}")
                if args.temporal_mask_ratio is not None:
                    print(f"   Temporal Mask Ratio: {args.temporal_mask_ratio}")
        
        # 8. Regularizers
        print(f"\n8. Regularizers")
        print(f"   8.1 Spectral Regularizer")
        print(f"      Status: ✓ ENABLED")
        print(f"      Target Spectral Radius: {args.target_spectral_radius}")
        print(f"      Weight: 0.1 (in loss)")
        
        print(f"   8.2 Consistency Regularizer")
        print(f"      Status: ✓ ENABLED")
        print(f"      Type: Latent dynamics consistency (z_pred vs z_true)")
        print(f"      Weight: 0.5 (in loss)")
        
        # 9. Loss Configuration
        print(f"\n9. Loss Configuration")
        print(f"   Recon Weight: {args.recon_weight}")
        print(f"   Forecast Weight: {args.forecast_weight}")
        print(f"   Cognitive Weight: {args.cognitive_weight}")
        print(f"   Consistency Weight: {args.consistency_weight}")
        print(f"   Spectral Weight: {args.spectral_weight}")
        print(f"   Rollout Weight: {args.rollout_weight}")
        if args.disable_wavelet:
            print(f"   Wavelet Weight: 0.0 (DISABLED)")
        if args.disable_koopman:
            print(f"   Koopman Weight: 0.0 (DISABLED)")
        if args.koopman_weight is not None:
            print(f"   Koopman Weight: {args.koopman_weight} (OVERRIDE)")
        
        print(f"{'='*70}\n")
        
        # Save extended run configuration (including backend and param stats)
        try:
            import subprocess
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_hash = None
        
        run_config = {
            **vars(args),
            "git_hash": git_hash,
            "temporal_backend": backend_str,
            "param_stats": stats,
            "split_info": {
                "split_type": "subject_wise",
                "train_split": float(args.train_split),
                "split_seed": getattr(args, "split_seed", 42),
                "n_train_subjects": len(train_subj) if "train_subj" in locals() else None,
                "n_val_subjects": len(val_subj) if "val_subj" in locals() else None,
                "n_train_segments": len(train_idx) if "train_idx" in locals() else None,
                "n_val_segments": len(val_idx) if "val_idx" in locals() else None,
            },
        }
        with open(output_dir / "run_config.json", "w") as f:
            json.dump(run_config, f, indent=2)
        
        # Print ablation study status
        ablation_status = []
        if not args.use_episodic_feedback:
            ablation_status.append("NO Episodic Feedback")
        if not args.use_task_context:
            ablation_status.append("NO Task Context")
        if not args.use_control_fusion:
            ablation_status.append("NO Control Fusion")
        if not args.use_temporal_transformer:
            ablation_status.append("NO Temporal Transformer")
        if not args.use_cognitive_readout:
            ablation_status.append("NO Cognitive Readout")
        if not args.use_spatial_encoder:
            ablation_status.append("NO Spatial Encoder (using Linear)")
        
        if ablation_status:
            print(f"  ⚠️  Ablation Study Mode:")
            for status in ablation_status:
                print(f"     - {status}")
        else:
            print(f"  ✓ Full Model (all modules enabled)")
        print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Module probe heads (for monitoring module-level losses)
    probe_heads = None
    probe_optimizer = None
    if args.enable_module_probes:
        if rank == 0 or not is_distributed:
            print(f"\n✓ Module probe heads enabled")
        probe_heads = ModuleProbeHeads(
            n_roi=args.n_roi,
            d_latent=args.latent_dim,
            d_control=args.control_dim,
            d_episodic=args.episodic_dim,
            d_context=args.context_dim,
            use_episodic_feedback=args.use_episodic_feedback,
            use_task_context=args.use_task_context,
            use_control_fusion=args.use_control_fusion,
        ).to(device)
        probe_optimizer = torch.optim.Adam(
            probe_heads.parameters(),
            lr=args.probe_lr,
            weight_decay=1e-5
        )
        if rank == 0 or not is_distributed:
            probe_params = sum(p.numel() for p in probe_heads.parameters())
            print(f"  Probe heads parameters: {probe_params:,}")
    else:
        if rank == 0 or not is_distributed:
            print(f"\n  Module probe heads disabled (use --enable_module_probes to enable, or remove --disable_module_probes)")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        if rank == 0 or not is_distributed:
            print(f"\nResuming from checkpoint: {args.resume}")
        
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Load model state
        if 'model_state_dict' in resume_checkpoint:
            model_state = resume_checkpoint['model_state_dict']
            # Handle DDP model state (remove 'module.' prefix if present)
            if is_distributed and not list(model_state.keys())[0].startswith('module.'):
                model_state = {f'module.{k}': v for k, v in model_state.items()}
            elif not is_distributed and list(model_state.keys())[0].startswith('module.'):
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            model.load_state_dict(model_state, strict=False)
        
        # Load optimizer state
        if 'optimizer_state_dict' in resume_checkpoint:
            try:
                optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
                if rank == 0 or not is_distributed:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Loaded optimizer state (current LR: {current_lr:.6f})")
            except Exception as e:
                if rank == 0 or not is_distributed:
                    print(f"  ⚠️  Could not load optimizer state: {e}")
                    print(f"     Will continue with fresh optimizer state")
        
        # Get start epoch
        if 'epoch' in resume_checkpoint:
            start_epoch = resume_checkpoint['epoch']
            if rank == 0 or not is_distributed:
                print(f"  Resuming from epoch {start_epoch}")
        
        if rank == 0 or not is_distributed:
            print(f"  ✓ Checkpoint loaded successfully")
    
    # Create scheduler (after optimizer is loaded)
    if getattr(args, 'use_plateau_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=getattr(args, 'plateau_factor', 0.8),
            patience=getattr(args, 'plateau_patience', 15),
            min_lr=1e-5,
        )
        
        # Load scheduler state if available (for ReduceLROnPlateau, this is just the last best value)
        if args.resume and Path(args.resume).exists() and 'scheduler_state_dict' in resume_checkpoint:
            try:
                scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
                if rank == 0 or not is_distributed:
                    print(f"  ✓ Loaded scheduler state")
            except Exception as e:
                if rank == 0 or not is_distributed:
                    print(f"  ⚠️  Could not load scheduler state: {e}")
                    print(f"     Scheduler will start fresh (may reduce LR too quickly)")
        
        # Option to reset learning rate when resuming (useful if model is stuck)
        if getattr(args, 'reset_lr_on_resume', False):
            if rank == 0 or not is_distributed:
                print(f"  🔄 Resetting learning rate to initial value: {args.lr:.6f}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    else:
        scheduler = None
    
    # Loss weights (base weights, KL will be annealed)
    base_loss_weights = {
        'recon': args.recon_weight,
        'forecast': args.forecast_weight,
        'cognitive': args.cognitive_weight,
        'consistency': args.consistency_weight,
        'spectral': args.spectral_weight,
        'rollout': args.rollout_weight,
    }
    
    # Curriculum scheduler
    curriculum = CurriculumScheduler(
        start_horizon=1,
        max_horizon=args.max_rollout_horizon,
        warmup_epochs=args.curriculum_warmup,
        increase_every=args.curriculum_increase_every
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon': [],
        'val_recon': [],
        'rollout_mae': [],
        'recon_r2': [],
        'calibration_ece': [],
        'spectral_radius': [],
        'rollout_horizon': [],
        'lr': [],
        'downstream_sex_accuracy': [],
        'downstream_sex_f1': [],
        'downstream_disease_metrics': {}  # {dataset_name: {best_balanced_accuracy, ...}}
    }
    
    # Load demographics for downstream evaluation
    # NOTE: Age evaluation is DISABLED - only sex evaluation is supported
    demographics = None
    if args.evaluate_downstream and evaluate_age_sex_during_training is not None:
        # Filter out age from downstream_tasks
        if hasattr(args, 'downstream_tasks') and args.downstream_tasks:
            tasks_list = args.downstream_tasks.lower().split(',')
            if 'age' in tasks_list:
                tasks_list.remove('age')
                args.downstream_tasks = ','.join(tasks_list)
                if rank == 0 or not is_distributed:
                    print(f"  ⚠️  Age evaluation is disabled, removed from downstream_tasks")
            if not tasks_list or (len(tasks_list) == 1 and tasks_list[0] == ''):
                args.evaluate_downstream = False
                if rank == 0 or not is_distributed:
                    print(f"  ⚠️  No valid downstream tasks (age disabled), disabling downstream evaluation")
        if args.demographics_csv:
            # Resolve demographics CSV path (handle relative paths)
            demo_path = Path(args.demographics_csv)
            if not demo_path.is_absolute():
                project_root = Path(__file__).parent.parent
                possible_paths = [
                    demo_path,
                    project_root / args.demographics_csv,
                    Path.cwd() / args.demographics_csv,
                    project_root.parent / args.demographics_csv,  # For ../../data/ paths
                ]
                
                for path in possible_paths:
                    if path.exists():
                        demo_path = path
                        break
                else:
                    # Try resolving relative to current working directory
                    demo_path = Path.cwd() / args.demographics_csv
            
            if demo_path.exists():
                try:
                    if rank == 0 or not is_distributed:
                        print(f"Loading demographics from: {demo_path}")
                    df = pd.read_csv(demo_path)
                    
                    if dataset_name == 'HCPA':
                        subject_col = 'Subject' if 'Subject' in df.columns else 'subjectkey'
                        if subject_col not in df.columns:
                            raise ValueError(f"Subject column not found. Available columns: {df.columns.tolist()}")
                        demo_df = df[[subject_col, 'Age', 'Sex']].copy()
                        demo_df.columns = ['subject_id', 'age', 'sex']
                        demo_df['subject_id'] = demo_df['subject_id'].astype(str).str.replace('sub-', '', regex=False)
                        if demo_df['sex'].dtype == 'object':
                            demo_df['sex'] = demo_df['sex'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
                    else:  # UKB
                        # UKB format can have different column names
                        # Check for 'eid' (subject ID)
                        if 'eid' not in df.columns:
                            raise ValueError(f"UKB data must have 'eid' column. Available columns: {df.columns.tolist()}")
                        subject_col = 'eid'
                        
                        # Check if age needs to be computed or is already present
                        if 'age' in df.columns or 'Age' in df.columns:
                            # Age already computed
                            age_col = 'age' if 'age' in df.columns else 'Age'
                            if 'sex' in df.columns or 'Sex' in df.columns:
                                sex_col = 'sex' if 'sex' in df.columns else 'Sex'
                                demo_df = df[[subject_col, age_col, sex_col]].copy()
                            elif 'p31' in df.columns:
                                # UKB uses p31 for sex (0=Female, 1=Male)
                                demo_df = df[[subject_col, age_col, 'p31']].copy()
                            else:
                                raise ValueError(f"UKB data must have 'sex', 'Sex', or 'p31' column. Available columns: {df.columns.tolist()}")
                            demo_df.columns = ['subject_id', 'age', 'sex']
                            if rank == 0 or not is_distributed:
                                print(f"  Using pre-computed age column: {age_col}")
                        elif 'p34' in df.columns and 'p53_i2' in df.columns:
                            # Need to compute age from birth year (p34) and scan date (p53_i2)
                            if rank == 0 or not is_distributed:
                                print(f"  Computing age from birth year (p34) and scan date (p53_i2)...")
                            
                            # Filter to subjects with both birth year and scan date
                            df_filtered = df[[subject_col, 'p34', 'p53_i2', 'p31']].copy()
                            n_total = len(df_filtered)
                            
                            # Drop rows with missing values
                            df_filtered = df_filtered.dropna(subset=['p34', 'p53_i2', 'p31'])
                            
                            # Extract year from p53_i2 (scan date)
                            import re
                            def extract_year(date_str):
                                if pd.isna(date_str):
                                    return np.nan
                                date_str = str(date_str)
                                # Try to extract first 4 digits (year)
                                match = re.search(r'(\d{4})', date_str)
                                if match:
                                    return int(match.group(1))
                                return np.nan
                            
                            df_filtered['scan_year'] = df_filtered['p53_i2'].apply(extract_year)
                            df_filtered['birth_year'] = pd.to_numeric(df_filtered['p34'], errors='coerce')
                            
                            # Drop any rows where year extraction failed
                            df_filtered = df_filtered.dropna(subset=['scan_year', 'birth_year'])
                            
                            # Compute age
                            df_filtered['age'] = df_filtered['scan_year'] - df_filtered['birth_year']
                            
                            # Filter reasonable ages (18-100 years)
                            df_filtered = df_filtered[(df_filtered['age'] >= 18) & (df_filtered['age'] <= 100)]
                            
                            demo_df = df_filtered[[subject_col, 'age', 'p31']].copy()
                            demo_df.columns = ['subject_id', 'age', 'sex']
                            
                            if rank == 0 or not is_distributed:
                                print(f"  ✓ Computed age for {len(demo_df)} subjects (dropped {n_total - len(demo_df)} incomplete/invalid entries)")
                        else:
                            raise ValueError(f"UKB data must have either 'age'/'Age' or ('p34' and 'p53_i2'). Available columns: {df.columns.tolist()}")
                        
                        demo_df['subject_id'] = demo_df['subject_id'].astype(str)
                        
                        # p31 is already 0/1 (0=Female, 1=Male) in UKB, but ensure numeric
                        demo_df['sex'] = pd.to_numeric(demo_df['sex'], errors='coerce')
                        demo_df = demo_df.dropna()  # Remove any rows with missing values
                    
                    # Convert age from months to years if needed
                    if demo_df['age'].max() > 120:
                        demo_df['age'] = demo_df['age'] / 12.0
                    
                    demographics = {
                        'subject_id': demo_df['subject_id'].values,
                        'age': demo_df['age'].values,
                        'sex': demo_df['sex'].values
                    }
                    if rank == 0 or not is_distributed:
                        print(f"✓ Loaded demographics for downstream evaluation: {len(demographics['subject_id'])} subjects")
                        print(f"  Sample subject IDs: {demographics['subject_id'][:5].tolist()}")
                except Exception as e:
                    if rank == 0 or not is_distributed:
                        print(f"⚠️  Warning: Failed to load demographics: {e}")
                        import traceback
                        traceback.print_exc()
                        print("   Downstream evaluation will be skipped")
                    args.evaluate_downstream = False
            else:
                if rank == 0 or not is_distributed:
                    print(f"⚠️  Warning: Demographics CSV not found: {demo_path}")
                    print(f"   Tried paths: {possible_paths if 'possible_paths' in locals() else [demo_path]}")
                    print("   Downstream evaluation will be skipped")
                args.evaluate_downstream = False
        else:
            if rank == 0 or not is_distributed:
                print(f"⚠️  Warning: Demographics CSV not provided")
                print("   Downstream evaluation will be skipped")
            args.evaluate_downstream = False
    
    # Training loop
    if rank == 0 or not is_distributed:
        print(f"Training for {args.epochs} epochs...")
        if not args.use_mae:
            print(f"Curriculum: horizon 1→{args.max_rollout_horizon}, warmup={args.curriculum_warmup}")
        if args.evaluate_downstream:
            print(f"Downstream evaluation: every {args.downstream_eval_freq} epochs")
        print("=" * 60)
    
    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = args.early_stopping_patience  # None = disabled
    best_downstream_score = float('inf')  # Lower is better (age_mse - sex_accuracy)
    best_downstream_epoch = 0
    
    # Module-level overfit tracking (Task 4)
    module_overfit_tracker = {}  # {module_name: {'best_val_loss': float, 'best_epoch': int, 'last_val_loss': float}}
    
    # Shape debug flag (only print once across all epochs)
    shape_debug_printed_global = False
    
    # Initialize CollapseGuard
    # NOTE: CollapseGuard patience=2 is very aggressive and may trigger too early
    # Consider increasing to 5-10 for more stable training, or disable if using full_train config
    collapse_patience = getattr(args, 'collapse_patience', 2)
    best_checkpoint_path = output_dir / 'best_model.pt' if (rank == 0 or not is_distributed) else None
    collapse_guard = CollapseGuard(
        best_checkpoint_path=best_checkpoint_path,
        patience=collapse_patience,
        logger=log_print if (rank == 0 or not is_distributed) else None
    ) if collapse_patience > 0 else None
    
    # Epoch snapshot storage
    epoch_snapshots = []
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Create epoch snapshot (for debugging epoch 8→9 transitions)
        if rank == 0 or not is_distributed:
            mask_ratio = getattr(args, 'mask_ratio', None)
            if args.fixed_mask_ratio is not None:
                mask_ratio = args.fixed_mask_ratio
            elif args.use_mae and hasattr(model, 'base_model') and hasattr(model.base_model, 'mae_wrapper'):
                # Get current mask ratio from MAE wrapper if available
                pass  # Will be logged separately
            
            snapshot = create_epoch_snapshot(
                epoch=epoch,
                optimizer=optimizer,
                model=model,
                args=args,
                loss_weights=base_loss_weights.copy(),
                mask_ratio=mask_ratio,
                horizon=horizon if 'horizon' in locals() else None
            )
            epoch_snapshots.append(snapshot)
            
            # Log snapshot to file and TensorBoard
            snapshot_file = output_dir / 'epoch_snapshots.json'
            with open(snapshot_file, 'w') as f:
                json.dump(epoch_snapshots, f, indent=2)
            
            if writer:
                for key, value in snapshot.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'epoch_config/{key}', value, epoch)
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float, bool)):
                                writer.add_scalar(f'epoch_config/{key}/{subkey}', subvalue, epoch)
            
            log_print(f"\n【Epoch {epoch} Config Snapshot】")
            log_print(f"  LR: {snapshot['lr']:.6f}")
            if 'mask_ratio' in snapshot:
                log_print(f"  Mask Ratio: {snapshot['mask_ratio']:.4f}")
            if 'horizon' in snapshot:
                log_print(f"  Horizon: {snapshot['horizon']}")
            log_print(f"  Loss Weights: {snapshot['loss_weights']}")
            if 'model_flags' in snapshot:
                log_print(f"  Model Flags: {snapshot['model_flags']}")
        
        # Update loss weights with optional schedule
        loss_weights = base_loss_weights.copy()
        
        # Apply ablation switches
        if args.disable_wavelet:
            loss_weights['wavelet'] = 0.0
        if args.disable_koopman:
            loss_weights['koopman'] = 0.0
        if args.koopman_weight is not None:
            loss_weights['koopman'] = args.koopman_weight
        
        # Loss schedule (phase-based) - optional
        if getattr(args, 'loss_schedule', 'none') != 'none':
            phaseA = getattr(args, 'phaseA_epochs', 0)
            phaseB = getattr(args, 'phaseB_epochs', 0)
            # Phase C is remaining epochs
            if args.loss_schedule == '2stage':
                if epoch <= phaseA:
                    # Phase A: recon-only (optionally KL)
                    loss_weights['forecast'] = 0.0
                    loss_weights['consistency'] = 0.0
                    loss_weights['spectral'] = 0.0
                    # 可选：先不约束动力学/对齐，集中压 recon_nll
                    loss_weights['align'] = 0.0
                    loss_weights['cycle'] = 0.0
                    loss_weights['smooth'] = 0.0
                else:
                    # Phase B: warm up forecast (and enable consistency/spectral)
                    warmup_str = getattr(args, 'forecast_warmup', '1.0')
                    factors = [float(x) for x in warmup_str.split(',') if x.strip()]
                    if len(factors) == 0:
                        factors = [1.0]
                    # Map epoch in PhaseB to index in factors
                    phase_epoch = max(1, epoch - phaseA)
                    idx = min(len(factors) - 1, (phase_epoch - 1) * len(factors) // max(1, phaseB))
                    loss_weights['forecast'] = base_loss_weights['forecast'] * factors[idx]
                    # Enable other regularizers at full weight
                    loss_weights['consistency'] = base_loss_weights['consistency']
                    loss_weights['spectral'] = base_loss_weights['spectral']
            elif args.loss_schedule == '3stage':
                phaseC_start = phaseA + phaseB + 1
                if epoch <= phaseA:
                    # Phase A: recon-only
                    loss_weights['forecast'] = 0.0
                    loss_weights['consistency'] = 0.0
                    loss_weights['spectral'] = 0.0
                    loss_weights['align'] = 0.0
                    loss_weights['cycle'] = 0.0
                    loss_weights['smooth'] = 0.0
                elif epoch <= phaseA + phaseB:
                    # Phase B: forecast warmup
                    warmup_str = getattr(args, 'forecast_warmup', '1.0')
                    factors = [float(x) for x in warmup_str.split(',') if x.strip()]
                    if len(factors) == 0:
                        factors = [1.0]
                    phase_epoch = max(1, epoch - phaseA)
                    idx = min(len(factors) - 1, (phase_epoch - 1) * len(factors) // max(1, phaseB))
                    loss_weights['forecast'] = base_loss_weights['forecast'] * factors[idx]
                    loss_weights['consistency'] = 0.0
                    loss_weights['spectral'] = 0.0
                else:
                    # Phase C: full joint training
                    loss_weights['forecast'] = base_loss_weights['forecast']
                    loss_weights['consistency'] = base_loss_weights['consistency']
                    loss_weights['spectral'] = base_loss_weights['spectral']
        # Set epoch for distributed sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Get curriculum horizon (only if not using MAE)
        horizon = curriculum.get_horizon(epoch) if not args.use_mae else 1
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, loss_weights,
            rollout_horizon=horizon,
            risk_aware=args.risk_aware,
            use_mae=args.use_mae,
            epoch=epoch,
            max_epochs=args.epochs,
            rank=rank,
            is_distributed=is_distributed,
            probe_heads=probe_heads,
            probe_optimizer=probe_optimizer,
            enable_module_probes=args.enable_module_probes,
            probe_weight=args.probe_weight,
            writer=writer if (rank == 0 or not is_distributed) else None,
            global_step=(epoch - 1) * len(train_loader),
            shape_debug_once=getattr(args, 'shape_debug_once', True),  # Pass shape debug flag
            shape_debug_printed_ref=[shape_debug_printed_global],  # Pass reference to global flag
            use_age_context=use_age_context,
            age_dict=age_dict,
            age_min=age_min,
            age_max=age_max,
            task_dim=args.task_dim
        )
        
        # Update global shape debug flag
        if train_metrics.get('shape_debug_printed', False):
            shape_debug_printed_global = True
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, device, loss_weights,
            rollout_horizon=args.max_rollout_horizon,
            use_mae=args.use_mae,
            epoch=epoch,
            max_epochs=args.epochs,
            rank=rank,
            is_distributed=is_distributed,
            probe_heads=probe_heads,
            enable_module_probes=args.enable_module_probes,
            use_age_context=use_age_context,
            age_dict=age_dict,
            age_min=age_min,
            age_max=age_max,
            task_dim=args.task_dim
        )
        
        # Get stats
        if is_distributed:
            spectral_radius = model.module.base_model.get_spectral_radius() if args.use_mae else model.module.get_spectral_radius()
        else:
            spectral_radius = model.base_model.get_spectral_radius() if args.use_mae else model.get_spectral_radius()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history (only on rank 0)
        if rank == 0 or not is_distributed:
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['train_recon'].append(train_metrics['recon_loss'])
            history['val_recon'].append(val_metrics['recon_loss'])
            history['rollout_mae'].append(val_metrics['rollout_mae'])
            history['recon_r2'].append(val_metrics.get('recon_r2', 0.0))
            history['calibration_ece'].append(val_metrics.get('ece', 0.0))
            history['spectral_radius'].append(spectral_radius)
            history['rollout_horizon'].append(horizon)
            history['lr'].append(current_lr)
        
        # Print summary (only on rank 0)
        if rank == 0 or not is_distributed:
            log_print(f"\nEpoch {epoch}/{args.epochs}")
            
            # Train metrics: show both weighted loss and raw metrics
            train_recon_loss = train_metrics['recon_loss']
            train_recon_mse = train_metrics.get('recon_mse_raw', 0.0)
            train_recon_mae = train_metrics.get('recon_mae_raw', 0.0)
            train_recon_nll = train_metrics.get('recon_nll_raw', 0.0)
            
            if args.use_mae:
                log_print(f"  Train: total={train_metrics['total_loss']:.4f}, "
                      f"recon_loss={train_recon_loss:.4f}, "
                      f"recon_mse={train_recon_mse:.4f}, "
                      f"recon_mae={train_recon_mae:.4f}, "
                      f"recon_nll={train_recon_nll:.4f}, "
                      f"mae_loss={train_metrics.get('mae_loss', 0.0):.4f}")
            else:
                log_print(f"  Train: total={train_metrics['total_loss']:.4f}, "
                      f"recon_loss={train_recon_loss:.4f}, "
                      f"recon_mse={train_recon_mse:.4f}, "
                      f"recon_mae={train_recon_mae:.4f}, "
                      f"recon_nll={train_recon_nll:.4f}, "
                      f"rollout={train_metrics['rollout_loss']:.4f}")
            
            # Val metrics: same format for consistency
            val_recon_loss = val_metrics['recon_loss']
            val_recon_mse = val_metrics.get('recon_mse_raw', 0.0)
            val_recon_mae = val_metrics.get('recon_mae_raw', 0.0)
            val_recon_nll = val_metrics.get('recon_nll_raw', 0.0)
            
            log_print(f"  Val:   total={val_metrics['total_loss']:.4f}, "
                  f"recon_loss={val_recon_loss:.4f}, "
                  f"recon_mse={val_recon_mse:.4f}, "
                  f"recon_mae={val_recon_mae:.4f}, "
                  f"recon_nll={val_recon_nll:.4f}, "
                  f"R²={val_metrics.get('recon_r2', 0.0):.4f}")
            
            if not args.use_mae:
                log_print(f"  Rollout MAE: {val_metrics['rollout_mae']:.4f}")
            if 'ece' in val_metrics:
                log_print(f"  Calibration: ECE={val_metrics['ece']:.4f}, NLL={val_metrics['nll']:.4f}")
            
            # Print R² debug info if available (helps diagnose R²=0 issues)
            # Note: debug['mse'] should match recon_mse_raw now
            if 'r2_debug' in val_metrics and epoch <= 3:  # Only print for first few epochs
                debug = val_metrics['r2_debug']
                log_print(f"  R² Debug: MSE={debug['mse']:.4f}, target_std={debug['target_std']:.4f}, "
                      f"pred_std={debug['pred_std']:.4f}, n_samples={debug['n_samples']}")
            
            log_print(f"  Spectral radius: {spectral_radius:.4f}, Horizon: {horizon}, LR: {current_lr:.6f}")
            
            # Check for collapse and recover if needed
            if collapse_guard is not None:
                pred_std = val_metrics.get('r2_debug', {}).get('pred_std', None)
                collapse_result = collapse_guard.check(
                    epoch=epoch,
                    val_loss=val_metrics['total_loss'],
                    val_r2=val_metrics.get('recon_r2', 0.0),
                    pred_std=pred_std
                )
            else:
                collapse_result = {'triggered': False, 'reason': '', 'actions': []}
            
            if collapse_result.get('triggered', False):
                log_print(f"\n⚠️  COLLAPSE DETECTED at epoch {epoch}")
                log_print(f"  Reason: {collapse_result['reason']}")
                log_print(f"  Actions: {collapse_result['actions']}")
                
                # Load best checkpoint
                if 'load_best' in collapse_result['actions'] and best_checkpoint_path and best_checkpoint_path.exists():
                    log_print(f"  → Loading best checkpoint from epoch {best_val_loss_epoch}")
                    checkpoint = torch.load(best_checkpoint_path, map_location=device)
                    
                    # Handle state_dict key mismatch
                    # Checkpoint was saved as model.base_model.state_dict() (no 'base_model.' prefix)
                    # But model.load_state_dict() expects 'base_model.xxx' prefix when use_mae=True
                    model_state = checkpoint['model_state_dict']
                    
                    # Check if we need to add 'base_model.' prefix
                    if args.use_mae and not any(k.startswith('base_model.') for k in model_state.keys()):
                        # Add 'base_model.' prefix to all checkpoint keys
                        model_state = {f'base_model.{k}': v for k, v in model_state.items()}
                        log_print(f"  → Added 'base_model.' prefix to checkpoint keys")
                    
                    # Try loading with strict=False to handle any remaining mismatches
                    try:
                        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                        if missing_keys:
                            log_print(f"  ⚠️  Missing keys (ignored): {len(missing_keys)} keys")
                            if len(missing_keys) <= 5:
                                log_print(f"     Missing: {missing_keys}")
                        if unexpected_keys:
                            log_print(f"  ⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
                            if len(unexpected_keys) <= 5:
                                log_print(f"     Unexpected: {unexpected_keys}")
                        log_print(f"  ✓ Checkpoint loaded")
                    except Exception as e:
                        log_print(f"  ⚠️  Failed to load checkpoint: {e}")
                        log_print(f"     Skipping checkpoint load, continuing with current model state")
                    
                    # Load optimizer state
                    if 'optimizer_state_dict' in checkpoint:
                        try:
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            log_print(f"  ✓ Optimizer state loaded")
                        except Exception as e:
                            log_print(f"  ⚠️  Failed to load optimizer state: {e}")
                            log_print(f"     Continuing with current optimizer state")
                
                # Reduce LR
                if 'reduce_lr' in collapse_result['actions']:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        param_group['lr'] = old_lr * 0.1
                        log_print(f"  → Reduced LR: {old_lr:.6f} → {param_group['lr']:.6f}")
                
                # Set grad clip (will be applied in next backward pass)
                if 'clip_grad' in collapse_result['actions']:
                    log_print(f"  → Gradient clipping set to 1.0")
            
            # Module probe loss table (if enabled and epoch matches frequency)
            if args.enable_module_probes and (epoch % args.module_probe_every == 0):
                train_probes = train_metrics.get('probe_losses', {})
                val_probes = val_metrics.get('probe_losses', {})
                
                if train_probes or val_probes:
                    log_print(f"\n  【模块Loss表格 (Module Loss Table)】")
                    log_print(f"  {'模块':<20} {'TrainLoss':<12} {'ValLoss':<12} {'Gap':<12} {'Gap%':<10}")
                    log_print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
                    
                    # Get all probe keys
                    all_probe_keys = set(train_probes.keys()) | set(val_probes.keys())
                    
                    # Track for Best Generalization
                    best_gen_module = None
                    best_gen_gap = float('inf')
                    
                    for key in sorted(all_probe_keys):
                        # Extract mean and count from train
                        train_info = train_probes.get(key, {})
                        train_mean = train_info.get('mean') if isinstance(train_info, dict) else (train_info if train_info is not None else None)
                        train_count = train_info.get('count', 0) if isinstance(train_info, dict) else 0
                        
                        # Extract mean and count from val
                        val_info = val_probes.get(key, {})
                        val_mean = val_info.get('mean') if isinstance(val_info, dict) else (val_info if val_info is not None else None)
                        val_count = val_info.get('count', 0) if isinstance(val_info, dict) else 0
                        
                        # Format values
                        train_str = f"{train_mean:.6f}" if train_mean is not None else "N/A"
                        val_str = f"{val_mean:.6f}" if val_mean is not None else "N/A"
                        
                        # Compute Gap and Gap%
                        # Gap = TrainLoss - ValLoss (positive gap means train > val, i.e., overfitting)
                        if train_mean is not None and val_mean is not None:
                            gap = train_mean - val_mean  # TrainLoss - ValLoss
                            eps = 1e-8
                            gap_pct = (gap / (max(abs(val_mean), eps))) * 100  # Gap% = Gap / max(|ValLoss|, eps) * 100
                            gap_str = f"{gap:.6f}"
                            gap_pct_str = f"{gap_pct:.2f}%"
                            
                            # Track Best Generalization (smallest |gap| with valid val, i.e., gap closest to 0)
                            if not (val_mean != val_mean):  # Check for NaN
                                if abs(gap) < abs(best_gen_gap):
                                    best_gen_gap = gap
                                    best_gen_module = key
                        else:
                            gap_str = "N/A"
                            gap_pct_str = "N/A"
                        
                        log_print(f"  {key:<20} {train_str:<12} {val_str:<12} {gap_str:<12} {gap_pct_str:<10}")
                        
                        # Debug warning if val count is 0
                        if val_count == 0:
                            log_print(f"     ⚠️  WARNING: {key} has count=0 in validation. "
                                  f"Possible causes: forward flag not passed, probe skipped in eval mode, or intermediates missing.")
                    
                    # Best Generalization
                    if best_gen_module is not None:
                        log_print(f"\n  ✓ 最佳泛化模块 (Best Generalization): {best_gen_module} (gap={best_gen_gap:.6f})")
                    else:
                        log_print(f"\n  ⚠️  Best Generalization: N/A (no valid ValLoss found)")
                    
                    # Also print decoder losses for comparison
                    log_print(f"\n  {'主Decoder':<20} {'TrainLoss':<12} {'ValLoss':<12} {'Gap':<12} {'Gap%':<10}")
                    log_print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
                    decoder_train = train_metrics.get('recon_loss', 0.0)
                    decoder_val = val_metrics.get('recon_loss', 0.0)
                    if decoder_train is not None and decoder_val is not None:
                        decoder_gap = decoder_train - decoder_val  # TrainLoss - ValLoss
                        eps = 1e-8
                        decoder_gap_pct = (decoder_gap / (max(abs(decoder_val), eps))) * 100
                        log_print(f"  {'decoder_recon':<20} {decoder_train:<12.6f} {decoder_val:<12.6f} {decoder_gap:<12.6f} {decoder_gap_pct:<10.2f}%")
                    
                    if 'forecast_loss' in train_metrics and 'forecast_loss' in val_metrics:
                        forecast_train = train_metrics.get('forecast_loss', 0.0)
                        forecast_val = val_metrics.get('forecast_loss', 0.0)
                        if forecast_train is not None and forecast_val is not None:
                            forecast_gap = forecast_train - forecast_val  # TrainLoss - ValLoss
                            eps = 1e-8
                            forecast_gap_pct = (forecast_gap / (max(abs(forecast_val), eps))) * 100
                            log_print(f"  {'decoder_forecast':<20} {forecast_train:<12.6f} {forecast_val:<12.6f} {forecast_gap:<12.6f} {forecast_gap_pct:<10.2f}%")
                else:
                    # Debug: show why probe losses are empty
                    log_print(f"\n  ⚠️  Module probe losses are empty")
                    log_print(f"     Train probe_losses keys: {list(train_probes.keys()) if train_probes else 'None'}")
                    log_print(f"     Val probe_losses keys: {list(val_probes.keys()) if val_probes else 'None'}")
                    log_print(f"     This may indicate that intermediates were not returned or probe heads failed")
        
        # Check if learning is happening (diagnostic for stuck training)
        if epoch > start_epoch + 1 and (rank == 0 or not is_distributed):
            if len(history['val_loss']) >= 2:
                recent_val_losses = history['val_loss'][-5:]  # Last 5 epochs
                if len(recent_val_losses) >= 3:
                    val_loss_std = np.std(recent_val_losses)
                    val_loss_change = recent_val_losses[-1] - recent_val_losses[0]
                    if val_loss_std < 0.01 and abs(val_loss_change) < 0.01:
                        log_print(f"  ⚠️  WARNING: Validation loss is flat (std={val_loss_std:.6f}, change={val_loss_change:.6f})")
                        log_print(f"     Model may be stuck. Consider:")
                        log_print(f"     1. Using --reset_lr_on_resume to restart learning rate")
                        log_print(f"     2. Checking if learning rate is too low (current: {current_lr:.6f})")
                        log_print(f"     3. Reducing plateau_patience or disabling scheduler")
        
        # Learning rate scheduling (optional)
        if scheduler is not None:
            if getattr(args, 'early_stop_metric', 'total') == 'recon_nll':
                scheduler.step(val_metrics.get('recon_nll_raw', val_metrics['total_loss']))
            else:
                scheduler.step(val_metrics['total_loss'])
        
        # Downstream evaluation (only on rank 0, periodically) - SEX + DISEASE (AGE DISABLED)
        downstream_metrics = None
        downstream_disease_metrics = {}
        downstream_score = None
        
        if args.evaluate_downstream and (epoch % args.downstream_eval_freq == 0 or epoch == 1) and (rank == 0 or not is_distributed):
            try:
                # Check downstream_tasks - evaluate sex and disease (age is disabled)
                allowed_tasks = args.downstream_tasks.lower().split(',') if hasattr(args, 'downstream_tasks') and args.downstream_tasks else ['sex', 'disease']
                if 'all' in allowed_tasks:
                    allowed_tasks = ['sex', 'disease']  # Sex + disease, no age
                if 'age' in allowed_tasks:
                    allowed_tasks.remove('age')  # Remove age if present
                    if rank == 0 or not is_distributed:
                        print(f"  ⚠️  Age evaluation is disabled, skipping age")
                
                # Get model for evaluation (unwrap DDP if needed)
                eval_model = model.module if is_distributed else model
                eval_model = eval_model.base_model if args.use_mae else eval_model
                
                # Save checkpoint temporarily for disease evaluation
                temp_checkpoint = output_dir / f'temp_checkpoint_epoch_{epoch}.pt'
                if rank == 0 or not is_distributed:
                    torch.save({
                        'model_state_dict': eval_model.state_dict(),
                        'config': {
                            'n_roi': args.n_roi,
                            'latent_dim': args.latent_dim,
                            'control_dim': args.control_dim,
                            'episodic_dim': args.episodic_dim,
                            'context_dim': args.context_dim,
                            'task_dim': args.task_dim,
                            'n_classes': args.n_classes,
                            'graph_type': args.graph_type,
                            'n_graph_layers': args.n_graph_layers,
                            'cheb_k': args.cheb_k,
                            'drop_edge_p': args.drop_edge_p,
                            'n_temporal_layers': args.n_temporal_layers,
                            'n_heads': args.n_heads,
                            'koopman_rank': args.koopman_rank,
                            'target_spectral_radius': args.target_spectral_radius,
                            'use_sde': args.use_sde,
                            'diffusion_mode': args.diffusion_mode,
                            'uncertainty_mode': args.uncertainty_mode,
                        }
                    }, temp_checkpoint)
                
                # 1. Evaluate SEX (if in allowed tasks)
                if 'sex' in allowed_tasks and evaluate_age_sex_during_training is not None and demographics is not None:
                    try:
                        # Try to call with skip_age parameter if supported
                        try:
                            downstream_metrics = evaluate_age_sex_during_training(
                                eval_model, val_loader, device, demographics,
                                feature_type='z_all', max_samples=500,
                                skip_age=True  # Explicitly skip age
                            )
                        except TypeError:
                            # Function doesn't support skip_age, call without it and filter results
                            downstream_metrics = evaluate_age_sex_during_training(
                                eval_model, val_loader, device, demographics,
                                feature_type='z_all', max_samples=500
                            )
                            # Remove age metrics if present (age is disabled)
                            if downstream_metrics:
                                downstream_metrics.pop('age_mse', None)
                                downstream_metrics.pop('age_pearson', None)
                        
                        if downstream_metrics and downstream_metrics.get('n_samples', 0) > 10:
                            # Track sex metrics
                            if 'sex_accuracy' in downstream_metrics:
                                history['downstream_sex_accuracy'].append(downstream_metrics['sex_accuracy'])
                            if 'sex_f1' in downstream_metrics:
                                history['downstream_sex_f1'].append(downstream_metrics['sex_f1'])
                            
                            print(f"  Downstream (Sex): "
                                  f"Acc={downstream_metrics.get('sex_accuracy', 0.0):.3f}, "
                                  f"F1={downstream_metrics.get('sex_f1', 0.0):.3f} "
                                  f"(n={downstream_metrics.get('n_samples', 0)})")
                    except Exception as eval_error:
                        if rank == 0 or not is_distributed:
                            print(f"  ⚠️  Sex evaluation error: {eval_error}, skipping")
                        downstream_metrics = None
                
                # 2. Evaluate DISEASE on all datasets (if in allowed tasks)
                if 'disease' in allowed_tasks and run_disease_evaluation is not None:
                    disease_datasets = getattr(args, 'downstream_datasets', DISEASE_DATASETS)
                    if isinstance(disease_datasets, str):
                        if disease_datasets.lower() == 'all':
                            disease_datasets = DISEASE_DATASETS
                        else:
                            disease_datasets = [d.strip() for d in disease_datasets.split(',')]
                    
                    print(f"  Evaluating disease on {len(disease_datasets)} datasets...")
                    for dataset_name in disease_datasets:
                        try:
                            disease_results = run_disease_evaluation(
                                checkpoint=str(temp_checkpoint),
                                dataset_name=dataset_name,
                                output_dir=str(output_dir / 'disease_eval' / dataset_name),
                                feature_type='z_concat',
                                classifiers=['LogisticRegression']  # Use fastest classifier for training-time eval
                            )
                            
                            if disease_results and 'error' not in disease_results:
                                downstream_disease_metrics[dataset_name] = disease_results
                                best_acc = disease_results.get('best_balanced_accuracy', 0.0)
                                print(f"    {dataset_name}: Balanced Acc={best_acc:.3f}")
                        except Exception as e:
                            if rank == 0 or not is_distributed:
                                print(f"    ⚠️  {dataset_name} evaluation failed: {e}")
                
                # Compute combined score: sex accuracy + average disease balanced accuracy
                if downstream_metrics and downstream_disease_metrics:
                    sex_acc = downstream_metrics.get('sex_accuracy', 0.0)
                    disease_accs = [v.get('best_balanced_accuracy', 0.0) for v in downstream_disease_metrics.values() if 'best_balanced_accuracy' in v]
                    avg_disease_acc = np.mean(disease_accs) if disease_accs else 0.0
                    # Score: higher is better, so use negative for consistency with loss-based early stopping
                    downstream_score = -(0.3 * sex_acc + 0.7 * avg_disease_acc)
                    print(f"  Combined Score: {abs(downstream_score):.3f} (Sex: {sex_acc:.3f}, Disease Avg: {avg_disease_acc:.3f})")
                elif downstream_metrics:
                    # Only sex available
                    sex_acc = downstream_metrics.get('sex_accuracy', 0.0)
                    downstream_score = -sex_acc
                elif downstream_disease_metrics:
                    # Only disease available
                    disease_accs = [v.get('best_balanced_accuracy', 0.0) for v in downstream_disease_metrics.values() if 'best_balanced_accuracy' in v]
                    avg_disease_acc = np.mean(disease_accs) if disease_accs else 0.0
                    downstream_score = -avg_disease_acc
                
                # Clean up temp checkpoint
                if temp_checkpoint.exists():
                    temp_checkpoint.unlink()
                    
            except Exception as e:
                if rank == 0 or not is_distributed:
                    print(f"  ⚠️  Downstream evaluation failed: {e}")
                    import traceback
                    log_print(f"  Downstream evaluation error: {traceback.format_exc()}")
                # Clean up temp checkpoint on error
                if 'temp_checkpoint' in locals() and temp_checkpoint.exists():
                    temp_checkpoint.unlink()
        
        # ---- Save best model and check early stopping (DDP-safe) ----
        should_stop = False  # decided on rank 0, then broadcast

        if rank == 0 or not is_distributed:
            # Determine monitored metric for early stopping / best model
            if getattr(args, 'early_stop_metric', 'total') == 'recon_nll':
                recon_nll_raw = val_metrics.get('recon_nll_raw', 0.0)
                # Fallback to total_loss if recon_nll_raw is 0.0 or not available
                if recon_nll_raw > 0.0:
                    monitor_value = recon_nll_raw
                    monitor_name = "recon_nll_raw"
                else:
                    monitor_value = val_metrics['total_loss']
                    monitor_name = "total_loss"
                    if epoch == 1:
                        print(f"  ⚠️  Warning: recon_nll_raw is 0.0 or not available, using total_loss instead")
            else:
                monitor_value = val_metrics['total_loss']
                monitor_name = "total_loss"
            
            # Determine if this is the best model
            # Handle case where best_val_loss might be 0.0 (shouldn't happen, but be safe)
            if best_val_loss == float('inf'):
                is_best_pretraining = True  # First epoch is always best
            else:
                is_best_pretraining = monitor_value < best_val_loss and best_val_loss > 0
            is_best_downstream = (
                args.save_best_downstream and downstream_metrics and
                downstream_metrics.get('n_samples', 0) > 10 and
                downstream_score is not None and
                downstream_score < best_downstream_score
            )
            
            # Update best scores
            if is_best_pretraining:
                best_val_loss = monitor_value
                best_val_loss_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if is_best_downstream:
                best_downstream_score = downstream_score
                best_downstream_epoch = epoch
            
            # Decide whether to save best model
            save_best_model = False
            save_reason = None
            if args.save_best_downstream and is_best_downstream:
                save_best_model = True
                save_reason = "downstream performance"
            elif is_best_pretraining:
                save_best_model = True
                save_reason = f"pretraining ({monitor_name})"
            
            # Save best model checkpoint
            if save_best_model:
                if is_distributed:
                    model_state = model.module.base_model.state_dict() if args.use_mae else model.module.state_dict()
                else:
                    model_state = model.base_model.state_dict() if args.use_mae else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': float(monitor_value),  # Standard key for manage_checkpoints.py
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'val_total_loss': float(val_metrics['total_loss']),
                    'val_monitor_name': monitor_name,
                    'val_monitor_value': float(monitor_value),
                    'config': vars(args),
                    'param_stats': stats
                }
                
                if downstream_metrics:
                    checkpoint['downstream_metrics'] = downstream_metrics
                
                torch.save(checkpoint, output_dir / 'best_model.pt')
                print(f"  ✓ Saved best model (epoch {epoch}, {save_reason})")
            
            # Early stopping decision
            if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered!")
                print(f"   Best {monitor_name}: {best_val_loss:.6f} at epoch {best_val_loss_epoch}")
                print(f"   Current {monitor_name}: {monitor_value:.6f}")
                print(f"   Current total_loss: {val_metrics['total_loss']:.6f}")
                print(f"   No improvement for {early_stopping_patience} epochs")
                should_stop = True
        
        # Broadcast stop flag to all ranks (prevents DDP hang)
        if is_distributed:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        
        if should_stop:
            if is_distributed:
                safe_barrier()
            break
        
        # Periodic checkpoints
        if rank == 0 or not is_distributed:
            if epoch % 10 == 0:
                if is_distributed:
                    model_state = model.module.base_model.state_dict() if args.use_mae else model.module.state_dict()
                else:
                    model_state = model.base_model.state_dict() if args.use_mae else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['total_loss'],
                    'config': vars(args)
                }
                torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        if rank == 0 or not is_distributed:
            print("-" * 60)
    
    # Save history and summary (only on rank 0)
    if rank == 0 or not is_distributed:
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save summary for easy comparison
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else best_val_loss
        # Avoid division by zero - use absolute threshold if best_val_loss is 0 or very small
        if best_val_loss > 0.01:
            overfitting_detected = final_val_loss > best_val_loss * 1.01  # 1% threshold
        else:
            overfitting_detected = final_val_loss > best_val_loss + 0.01  # Absolute threshold
        
        summary = {
            'best_val_loss': float(best_val_loss),
            'final_val_loss': float(final_val_loss),
            'best_recon_r2': float(max(history['recon_r2'])) if history['recon_r2'] else 0.0,
            'final_recon_r2': float(history['recon_r2'][-1]) if history['recon_r2'] else 0.0,
            'best_epoch': best_val_loss_epoch,
            'total_epochs': len(history['val_loss']),
            'overfitting_detected': overfitting_detected,
            'val_loss_increase': float(final_val_loss - best_val_loss),
            'val_loss_increase_pct': float((final_val_loss - best_val_loss) / best_val_loss * 100) if best_val_loss > 0 else 0.0,
            'model_name': str(output_dir.name),
            'ablation_info': {
                'use_episodic_feedback': args.use_episodic_feedback,
                'use_task_context': args.use_task_context,
                'use_control_fusion': args.use_control_fusion,
                'use_temporal_transformer': args.use_temporal_transformer,
                'use_cognitive_readout': args.use_cognitive_readout,
                'use_spatial_encoder': args.use_spatial_encoder,
                'diffusion_mode': args.diffusion_mode,
                'use_sde': args.use_sde
            }
        }
        try:
            with open(output_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save summary.json: {e}")
            # Try to save a minimal summary
            try:
                minimal_summary = {
                    'best_val_loss': float(best_val_loss),
                    'final_val_loss': float(final_val_loss),
                    'best_epoch': best_val_loss_epoch,
                    'total_epochs': len(history['val_loss']),
                    'error': str(e)
                }
                with open(output_dir / 'summary_minimal.json', 'w') as f:
                    json.dump(minimal_summary, f, indent=2)
            except:
                pass
        
        log_print("\n" + "=" * 60)
        log_print("Training complete!")
        log_print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_val_loss_epoch})")
        log_print(f"Final validation loss: {final_val_loss:.6f}")
        
        if overfitting_detected:
            # Avoid division by zero
            if best_val_loss > 0:
                increase_pct = (final_val_loss - best_val_loss) / best_val_loss * 100
                log_print(f"⚠️  Overfitting detected: Validation loss increased by {increase_pct:.2f}% after epoch {best_val_loss_epoch}")
            else:
                increase_abs = final_val_loss - best_val_loss
                log_print(f"⚠️  Overfitting detected: Validation loss increased by {increase_abs:.6f} (absolute) after epoch {best_val_loss_epoch}")
            log_print(f"   Consider using early stopping (--early_stopping_patience) or reducing learning rate")
        
        if history['recon_r2']:
            log_print(f"Best validation R²: {max(history['recon_r2']):.4f}")
        log_print(f"Final parameter reduction: {stats['total_dense'] / stats['total_structured']:.2f}x")
        log_print(f"Results saved to: {output_dir}")
        log_print("=" * 60)
        
        # Module-level overfit diagnosis summary (Task 4)
        if args.enable_module_probes and module_overfit_tracker:
            log_print("\n" + "=" * 70)
            log_print("【模块级Overfit诊断汇总表 (Module Overfit Diagnosis Summary)】")
            log_print("=" * 70)
            log_print(f"{'模块':<25} {'BestEpoch':<12} {'BestValLoss':<15} {'LastValLoss':<15} {'RelativeIncrease':<18}")
            log_print("-" * 70)
            
            # Sort by best_epoch (ascending)
            sorted_modules = sorted(module_overfit_tracker.items(), key=lambda x: x[1]['best_epoch'])
            
            for module_name, tracker in sorted_modules:
                best_epoch = tracker['best_epoch']
                best_val_loss = tracker['best_val_loss']
                last_val_loss = tracker['last_val_loss']
                
                # Calculate relative increase
                if best_val_loss > 0:
                    relative_increase = (last_val_loss - best_val_loss) / best_val_loss * 100
                else:
                    relative_increase = float('inf') if last_val_loss > best_val_loss else 0.0
                
                log_print(f"{module_name:<25} {best_epoch:<12} {best_val_loss:<15.6f} {last_val_loss:<15.6f} {relative_increase:<18.2f}%")
            
            log_print("-" * 70)
            
            # Top-5 earliest degradation (smallest best_epoch)
            earliest_degradation = sorted(module_overfit_tracker.items(), key=lambda x: x[1]['best_epoch'])[:5]
            log_print("\n【最早开始退化 Top-5 模块 (Earliest Degradation Top-5)】")
            for i, (module_name, tracker) in enumerate(earliest_degradation, 1):
                log_print(f"  {i}. {module_name}: best_epoch={tracker['best_epoch']}, "
                         f"best_val_loss={tracker['best_val_loss']:.6f}, "
                         f"last_val_loss={tracker['last_val_loss']:.6f}")
            
            # Top-5 largest relative increase
            largest_increase = []
            for module_name, tracker in module_overfit_tracker.items():
                best_val_loss = tracker['best_val_loss']
                last_val_loss = tracker['last_val_loss']
                if best_val_loss > 0:
                    relative_increase = (last_val_loss - best_val_loss) / best_val_loss * 100
                    largest_increase.append((module_name, tracker, relative_increase))
            
            largest_increase.sort(key=lambda x: x[2], reverse=True)
            log_print("\n【退化幅度最大 Top-5 模块 (Largest Degradation Top-5)】")
            for i, (module_name, tracker, rel_inc) in enumerate(largest_increase[:5], 1):
                log_print(f"  {i}. {module_name}: relative_increase={rel_inc:.2f}%, "
                         f"best_epoch={tracker['best_epoch']}, "
                         f"best_val_loss={tracker['best_val_loss']:.6f}, "
                         f"last_val_loss={tracker['last_val_loss']:.6f}")
            
            log_print("=" * 70)
        
        # Close log file
        if 'log_file' in locals() and log_file is not None:
            log_file.close()
            print(f"✓ Training log saved to: {log_file_path}")
    
    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Train BrainMech foundation model')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Configuration name from configs.pretrain_configs (e.g., sota_latent256, small_model). '
                             'If provided, will load all parameters from config and override command-line args.')
    
    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to BOLD directory with CSV files (alternative to --npz_path)')
    parser.add_argument('--npz_path', type=str, default=None,
                        help='Path to .npz file containing BOLD and FC data (recommended)')
    parser.add_argument('--dataset_type', type=str, default='',
                        choices=['', 'HCPA', 'UKB', 'COMBINED', 'hcpa', 'ukb', 'combined'],
                        help='Dataset type (HCPA, UKB, or COMBINED). If empty, will auto-detect from filename.')
    parser.add_argument('--n_roi', type=int, default=116)
    parser.add_argument('--sequence_length', type=int, default=100)
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--task_filter', type=str, nargs='+', default=None,
                        help='Task names to filter (e.g., REST, CARIT). Default: None (load all). For npz files, task filtering may not be needed if data is pre-filtered.')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to load (applies to all datasets)')
    parser.add_argument('--max_files_third_dataset', type=int, default=None,
                        help='Maximum number of files for third dataset (ABCD) - for dataset balancing')
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for subject-wise train/val split')
    parser.add_argument('--fc_matrix_path', type=str, default=None)
    parser.add_argument('--fc_mode', type=str, default='dynamic', choices=['static', 'dynamic', 'hybrid'],
                        help='FC computation mode: static (one FC per subject), dynamic (FC per window), hybrid (weighted combination)')
    parser.add_argument('--hybrid_weight', type=float, default=0.3,
                        help='Weight for dynamic FC in hybrid mode (0.0 = fully static, 1.0 = fully dynamic)')
    parser.add_argument('--additional_data_dir', type=str, default=None,
                        help='Path to additional dataset directory (for COMBINED dataset type)')
    parser.add_argument('--additional_fc_matrix_path', type=str, default=None,
                        help='Path to additional FC directory (for COMBINED dataset type)')
    parser.add_argument('--third_data_dir', type=str, default=None,
                        help='Path to third dataset directory (for COMBINED dataset type with 3 datasets)')
    parser.add_argument('--third_fc_matrix_path', type=str, default=None,
                        help='Path to third FC directory (for COMBINED dataset type with 3 datasets)')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--control_dim', type=int, default=64)
    parser.add_argument('--episodic_dim', type=int, default=64)
    parser.add_argument('--context_dim', type=int, default=64)
    parser.add_argument('--task_dim', type=int, default=32)
    parser.add_argument('--n_classes', type=int, default=4)
    
    # Graph encoder
    parser.add_argument('--graph_type', type=str, default='cheb', choices=['cheb', 'gat'])
    parser.add_argument('--n_graph_layers', type=int, default=3)
    parser.add_argument('--cheb_k', type=int, default=3)
    parser.add_argument('--drop_edge_p', type=float, default=0.1)
    
    # Temporal transformer
    parser.add_argument('--n_temporal_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    
    # Koopman core
    parser.add_argument('--koopman_rank', type=int, default=16)
    parser.add_argument('--target_spectral_radius', type=float, default=0.95)
    parser.add_argument('--use_sde', action='store_true', default=True)
    parser.add_argument('--diffusion_mode', type=str, default='learned', choices=['learned', 'fixed', 'none'])
    parser.add_argument('--init_noise_std', type=float, default=0.01,
                        help='Initial noise standard deviation for fixed diffusion mode')
    parser.add_argument('--uncertainty_mode', type=str, default='heteroscedastic', 
                        choices=['heteroscedastic', 'homoscedastic', 'none'])
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Number of epochs with no improvement before early stopping (None = disabled)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Subdirectory name in results/ folder (empty = root of results/)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # MAE pre-training
    parser.add_argument('--use_mae', action='store_true', default=True,
                        help='Use MAE (Masked Autoencoder) pre-training')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='Fixed ratio of input to mask (0.0-1.0)')
    parser.add_argument('--mask_strategy', type=str, default='random',
                        choices=['random', 'block', 'structured'])
    parser.add_argument('--mask_type', type=str, default='both',
                        choices=['spatial', 'temporal', 'both'],
                        help='Masking type: spatial (mask ROIs), temporal (mask timepoints), or both')
    parser.add_argument('--spatial_mask_ratio', type=float, default=None,
                        help='Masking ratio for spatial dimension (only used when mask_type=both). If None, uses mask_ratio')
    parser.add_argument('--temporal_mask_ratio', type=float, default=None,
                        help='Masking ratio for temporal dimension (only used when mask_type=both). If None, uses mask_ratio')
    parser.add_argument('--use_random_masking', action='store_true', default=True,
                        help='Use random masking ratio instead of fixed')
    parser.add_argument('--min_mask_ratio', type=float, default=0.1,
                        help='Minimum masking ratio for random masking')
    parser.add_argument('--max_mask_ratio', type=float, default=0.8,
                        help='Maximum masking ratio for random masking')
    parser.add_argument('--use_curriculum', action='store_true', default=True,
                        help='Use curriculum learning (start easy, end hard)')
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--forecast_weight', type=float, default=1.0)
    parser.add_argument('--cognitive_weight', type=float, default=0.5)
    parser.add_argument('--consistency_weight', type=float, default=0.5)
    parser.add_argument('--spectral_weight', type=float, default=0.1)
    parser.add_argument('--rollout_weight', type=float, default=0.5)
    
    # Curriculum learning
    parser.add_argument('--max_rollout_horizon', type=int, default=8)
    parser.add_argument('--curriculum_warmup', type=int, default=10)
    parser.add_argument('--curriculum_increase_every', type=int, default=5)
    parser.add_argument('--risk_aware', action='store_true', default=True)
    
    # Shape debugging
    parser.add_argument('--shape_debug_once', action='store_true', default=True,
                        help='Print key tensor shapes on first forward pass (default: True)')
    
    # Ablation switches
    parser.add_argument('--disable_wavelet', action='store_true', default=False,
                        help='Disable wavelet multiscale encoder')
    parser.add_argument('--disable_koopman', action='store_true', default=False,
                        help='Disable Koopman core (use identity dynamics)')
    parser.add_argument('--koopman_weight', type=float, default=None,
                        help='Override koopman loss weight (0.0 = disable koopman loss)')
    parser.add_argument('--fixed_mask_ratio', type=float, default=None,
                        help='Use fixed mask ratio (disable curriculum/schedule). Overrides mask_ratio schedule.')
    
    # Multi-dataset support
    parser.add_argument('--datasets', type=str, default='hcpa',
                        help='Comma-separated list of datasets: ukb,hcpa (default: hcpa)')
    parser.add_argument('--data_dir_ukb', type=str, default=None,
                        help='Path to UKB BOLD data directory')
    parser.add_argument('--data_dir_hcpa', type=str, default=None,
                        help='Path to HCPA BOLD data directory')
    parser.add_argument('--dataset_weights', type=str, default=None,
                        help='Comma-separated dataset weights (e.g., "0.8,0.2" for 80% UKB, 20% HCPA)')
    
    # Temporal FC mode (ablation)
    parser.add_argument('--temporal_fc_mode', action='store_true', default=False,
                        help='Replace temporal transformer with simple FC layers (ablation)')
    
    # Evaluation mode
    parser.add_argument('--eval_mode', type=str, default='linear_probe',
                        choices=['linear_probe', 'mlp_readout', 'finetune_decoder'],
                        help='Evaluation mode: linear_probe (frozen backbone), mlp_readout (frozen + MLP), finetune_decoder (train decoder)')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                        help='Hidden dimension for MLP readout head')
    parser.add_argument('--eval_epochs', type=int, default=50,
                        help='Epochs for finetune_decoder mode')
    parser.add_argument('--eval_lr', type=float, default=1e-4,
                        help='Learning rate for finetune_decoder mode')
    parser.add_argument('--finetune_modules', type=str, default='decoder,readout',
                        help='Comma-separated modules to unfreeze in finetune_decoder mode')
    
    # Mamba architecture flags
    parser.add_argument('--temporal_model', type=str, default='transformer',
                        choices=['transformer'],
                        help='Temporal model: transformer (mamba deprecated)')
    parser.add_argument('--use_dual_path_mamba', action='store_true', default=False,
                        help='Deprecated - not used')
    parser.add_argument('--allow_mamba_fallback', action='store_true', default=False,
                        help='Deprecated - not used')
    
    # Loss scheduling / training phases
    parser.add_argument('--loss_schedule', type=str, default='none',
                        choices=['none', '2stage', '3stage'],
                        help='Loss schedule: none, 2stage (recon→recon+forecast), or 3stage')
    parser.add_argument('--phaseA_epochs', type=int, default=20,
                        help='Number of epochs for Phase A (recon-only)')
    parser.add_argument('--phaseB_epochs', type=int, default=20,
                        help='Number of epochs for Phase B (forecast warmup)')
    parser.add_argument('--phaseC_epochs', type=int, default=10,
                        help='Number of epochs for Phase C (joint training, only used for 3stage)')
    parser.add_argument('--forecast_warmup', type=str, default='0.2,0.5,1.0',
                        help='Comma-separated factors for forecast weight warmup during Phase B')
    parser.add_argument('--early_stop_metric', type=str, default='total',
                        choices=['total', 'recon_nll'],
                        help='Metric for early stopping: total (total_loss) or recon_nll (validation recon NLL)')
    
    # LR scheduler (plateau)
    parser.add_argument('--use_plateau_scheduler', action='store_true', default=False,
                        help='Use ReduceLROnPlateau scheduler on validation loss')
    parser.add_argument('--plateau_factor', type=float, default=0.8,
                        help='Factor by which the learning rate will be reduced')
    parser.add_argument('--plateau_patience', type=int, default=15,
                        help='Number of epochs with no improvement before reducing LR')
    parser.add_argument('--reset_lr_on_resume', action='store_true', default=False,
                        help='Reset learning rate to initial value when resuming from checkpoint (useful if training is stuck)')
    parser.add_argument('--collapse_patience', type=int, default=2,
                        help='CollapseGuard patience (number of epochs with no improvement before triggering collapse recovery). Set to 0 to disable. Default: 2 (very aggressive, consider 5-10 for stable training)')
    
    # Sigma / uncertainty calibration
    parser.add_argument('--detach_sigma_head', action='store_true', default=False,
                        help='Detach features before sigma head to decouple from mean path')
    parser.add_argument('--sigma_calibration', type=str, default='none',
                        choices=['none', 'roi', 'roi+time'],
                        help='Sigma calibration strategy: none, roi bias, or roi+time bias')
    parser.add_argument('--sigma_max_time_steps', type=int, default=512,
                        help='Maximum time steps for sigma time calibration (for roi+time)')
    
    # Ablation study flags (all True by default for full model)
    # Use store_false so that --no_episodic_feedback disables it
    parser.add_argument('--no_episodic_feedback', action='store_false', dest='use_episodic_feedback', default=True,
                        help='Disable episodic feedback module (for ablation studies)')
    parser.add_argument('--no_task_context', action='store_false', dest='use_task_context', default=True,
                        help='Disable task context encoder (for ablation studies)')
    parser.add_argument('--no_control_fusion', action='store_false', dest='use_control_fusion', default=True,
                        help='Disable control fusion module (for ablation studies)')
    parser.add_argument('--no_temporal_transformer', action='store_false', dest='use_temporal_transformer', default=True,
                        help='Disable temporal transformer (for ablation studies)')
    parser.add_argument('--no_cognitive_readout', action='store_false', dest='use_cognitive_readout', default=True,
                        help='Disable cognitive readout head (for ablation studies)')
    parser.add_argument('--no_spatial_encoder', action='store_false', dest='use_spatial_encoder', default=True,
                        help='Disable graph spatial encoder, use simple linear projection (for ablation studies)')
    
    # Downstream evaluation during training
    parser.add_argument('--evaluate_downstream', action='store_true', default=False,
                        help='Evaluate sex + disease prediction during training (age is DISABLED)')
    parser.add_argument('--downstream_tasks', type=str, default='sex,disease',
                        help='Comma-separated tasks: sex,disease (age is DISABLED, will be ignored if included)')
    parser.add_argument('--downstream_datasets', type=str, default='all',
                        help='Comma-separated disease datasets: ABIDE,ADNI,PPMI,NEUROCON,TAOWU or "all" (default: all)')
    parser.add_argument('--downstream_eval_freq', type=int, default=10,
                        help='Evaluate downstream tasks every N epochs')
    parser.add_argument('--demographics_csv', type=str, default=None,
                        help='Path to demographics CSV for downstream evaluation')
    parser.add_argument('--save_best_downstream', action='store_true', default=False,
                        help='Save best model based on downstream task performance (age MSE + sex accuracy)')
    
    # Module probe heads for monitoring module-level losses
    parser.add_argument('--enable_module_probes', action='store_true', default=True,
                        help='Enable module-level probe heads for monitoring intermediate representations (default: True)')
    parser.add_argument('--disable_module_probes', dest='enable_module_probes', action='store_false',
                        help='Disable module-level probe heads (use this flag to disable)')
    parser.add_argument('--module_probe_every', type=int, default=1,
                        help='Print module probe loss table every N epochs (default: 1)')
    parser.add_argument('--probe_lr', type=float, default=1e-3,
                        help='Learning rate for probe heads (separate optimizer)')
    parser.add_argument('--probe_weight', type=float, default=1.0,
                        help='Weight for probe losses (if added to total loss)')
    
    # Parse args first to get --config if provided
    args, remaining = parser.parse_known_args()
    
    # Load configuration if specified
    if args.config:
        try:
            from configs.pretrain_configs import get_config
            config = get_config(args.config)
            
            # Convert config to args list
            config_args = config.to_args()
            
            # Create a mapping of argument names to their actions for type conversion
            arg_actions = {action.dest: action for action in parser._actions if hasattr(action, 'dest')}
            
            # Parse config args and update args namespace
            # Only update if CLI arg was not explicitly provided (i.e., is default/None)
            i = 0
            while i < len(config_args):
                if config_args[i].startswith('--'):
                    key = config_args[i].lstrip('--')
                    if i + 1 < len(config_args) and not config_args[i + 1].startswith('--'):
                        value = config_args[i + 1]
                        i += 2
                    else:
                        # Boolean flag (no value)
                        value = True
                        i += 1
                    
                    # Only override if current value is None or default
                    if hasattr(args, key):
                        current_value = getattr(args, key)
                        action = arg_actions.get(key)
                        
                        if action:
                            default_value = action.default
                            # Check if it's a default value
                            if current_value is None or current_value == default_value:
                                # Convert value to appropriate type based on action
                                try:
                                    if action.type:
                                        value = action.type(value)
                                    elif isinstance(default_value, bool):
                                        value = str(value).lower() in ('true', '1', 'yes', 'on')
                                    elif isinstance(default_value, int):
                                        value = int(value)
                                    elif isinstance(default_value, float):
                                        value = float(value)
                                    elif isinstance(default_value, list):
                                        # Handle list types (e.g., task_filter)
                                        value = value.split(',') if ',' in str(value) else [value]
                                except (ValueError, AttributeError, TypeError):
                                    # If conversion fails, keep as string
                                    pass
                                setattr(args, key, value)
                        else:
                            # Action not found, try direct assignment
                            if current_value is None or current_value == parser.get_default(key):
                                setattr(args, key, value)
            
            print(f"✓ Loaded configuration: {args.config}")
            if args.data_dir:
                print(f"  Data dir: {args.data_dir}")
            if hasattr(args, 'latent_dim'):
                print(f"  Latent dim: {args.latent_dim}")
        except Exception as e:
            print(f"❌ Error loading configuration '{args.config}': {e}")
            import traceback
            traceback.print_exc()
            return
    
    train(args)


if __name__ == '__main__':
    main()


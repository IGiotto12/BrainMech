"""
Training utilities for KoopmanFM
- Dataset kwargs filtering
- Epoch snapshot logging
- Tensor monitoring
- Collapse detection and recovery
"""

import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import torch
import numpy as np
import json
from datetime import datetime
import re


def filter_kwargs_by_signature(callable_obj: Callable, kwargs: Dict[str, Any], 
                                log_dropped: bool = True, logger=None) -> Dict[str, Any]:
    """
    Filter kwargs to only include parameters that the callable accepts.
    
    Args:
        callable_obj: Function or class constructor to check
        kwargs: Dictionary of keyword arguments
        log_dropped: Whether to log dropped parameters (once per callable)
        logger: Optional logger function (default: print)
    
    Returns:
        Filtered kwargs dictionary
    """
    if logger is None:
        logger = print
    
    try:
        sig = inspect.signature(callable_obj)
        param_names = set(sig.parameters.keys())
        
        # Filter kwargs
        filtered = {k: v for k, v in kwargs.items() if k in param_names}
        dropped = set(kwargs.keys()) - param_names
        
        # Log dropped parameters (only once per callable, or if log_dropped=True)
        if dropped and log_dropped:
            logger(f"  ℹ️  Filtered kwargs for {callable_obj.__name__}: "
                   f"dropped {dropped} (not in signature)")
        
        return filtered
    except Exception as e:
        # If signature inspection fails, return empty dict (safer than passing all)
        if log_dropped:
            logger(f"  ⚠️  Could not inspect signature for {callable_obj.__name__}: {e}")
        return {}


def parse_training_log(log_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Parse training.log to extract epoch-level metrics.
    
    Returns:
        Dictionary mapping epoch -> metrics dict
    """
    epochs = {}
    
    if not log_path.exists():
        return epochs
    
    current_epoch = None
    current_metrics = {}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Match epoch header: "Epoch 1/50"
            epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
            if epoch_match:
                if current_epoch is not None:
                    epochs[current_epoch] = current_metrics.copy()
                current_epoch = int(epoch_match.group(1))
                current_metrics = {'epoch': current_epoch}
                continue
            
            # Match train metrics: "Train: total=4.5839, recon_loss=3.4678, ..."
            if 'Train:' in line and current_epoch is not None:
                train_match = re.search(r'total=([\d.]+)', line)
                if train_match:
                    current_metrics['train_total'] = float(train_match.group(1))
                
                recon_match = re.search(r'recon_mse=([\d.]+)', line)
                if recon_match:
                    current_metrics['train_recon_mse'] = float(recon_match.group(1))
            
            # Match val metrics: "Val: total=1.8753, recon_loss=0.9198, ..."
            if 'Val:' in line and current_epoch is not None:
                val_match = re.search(r'total=([\d.]+)', line)
                if val_match:
                    current_metrics['val_total'] = float(val_match.group(1))
                
                recon_match = re.search(r'recon_mse=([\d.]+)', line)
                if recon_match:
                    current_metrics['val_recon_mse'] = float(recon_match.group(1))
            
            # Match R²: "R²=0.5808"
            r2_match = re.search(r'R²=([\d.]+)', line)
            if r2_match and current_epoch is not None:
                current_metrics['val_r2'] = float(r2_match.group(1))
            
            # Match LR: "LR: 0.001000"
            lr_match = re.search(r'LR:\s+([\d.]+)', line)
            if lr_match and current_epoch is not None:
                current_metrics['lr'] = float(lr_match.group(1))
            
            # Match mask_ratio or horizon
            mask_match = re.search(r'mask_ratio[=:]\s*([\d.]+)', line, re.IGNORECASE)
            if mask_match and current_epoch is not None:
                current_metrics['mask_ratio'] = float(mask_match.group(1))
            
            horizon_match = re.search(r'horizon[=:]\s*(\d+)', line, re.IGNORECASE)
            if horizon_match and current_epoch is not None:
                current_metrics['horizon'] = int(horizon_match.group(1))
    
    # Save last epoch
    if current_epoch is not None:
        epochs[current_epoch] = current_metrics
    
    return epochs


def create_epoch_snapshot(epoch: int, optimizer, model, args, loss_weights: Dict,
                          mask_ratio: Optional[float] = None,
                          horizon: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a snapshot of epoch configuration for debugging.
    
    Returns:
        Dictionary with epoch configuration
    """
    snapshot = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'lr': optimizer.param_groups[0]['lr'] if optimizer.param_groups else None,
        'loss_weights': loss_weights.copy(),
    }
    
    # Mask ratio / horizon
    if mask_ratio is not None:
        snapshot['mask_ratio'] = mask_ratio
    if horizon is not None:
        snapshot['horizon'] = horizon
    
    # Model flags
    if hasattr(model, 'base_model'):
        base = model.base_model
        snapshot['model_flags'] = {
            'use_wavelet': hasattr(base, 'wavelet_encoder') and base.wavelet_encoder is not None,
            'use_koopman': hasattr(base, 'koopman_core') and base.koopman_core is not None,
            'use_forecast': hasattr(base, 'decoder') and hasattr(base.decoder, 'forecast_decoder'),
        }
    
    # Args flags
    snapshot['args_flags'] = {
        'enable_module_probes': getattr(args, 'enable_module_probes', False),
        'use_mae': getattr(args, 'use_mae', True),
        'use_sde': getattr(args, 'use_sde', True),
    }
    
    return snapshot


def log_tensor_stats(tensor: torch.Tensor, name: str, logger=None) -> Dict[str, float]:
    """
    Compute and log statistics for a tensor.
    
    Returns:
        Dictionary with statistics
    """
    if logger is None:
        logger = print
    
    with torch.no_grad():
        stats = {
            'shape': list(tensor.shape),
            'mean': float(tensor.mean().item()),
            'std': float(tensor.std().item()),
            'min': float(tensor.min().item()),
            'max': float(tensor.max().item()),
            'has_nan': bool(torch.isnan(tensor).any().item()),
            'has_inf': bool(torch.isinf(tensor).any().item()),
        }
        
        # Fraction of near-zero values
        frac_zero = (tensor.abs() < 1e-8).float().mean().item()
        stats['frac_zero'] = frac_zero
        
        # Log
        logger(f"  {name}: shape={stats['shape']}, "
               f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
               f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
               f"frac_zero={frac_zero:.4f}, "
               f"has_nan={stats['has_nan']}, has_inf={stats['has_inf']}")
    
    return stats


class CollapseGuard:
    """
    Detects training collapse and automatically recovers.
    """
    
    def __init__(self, best_checkpoint_path: Optional[Path] = None,
                 patience: int = 2, logger=None):
        self.best_checkpoint_path = best_checkpoint_path
        self.patience = patience
        self.logger = logger if logger else print
        
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.no_improve_count = 0
        self.recovery_count = 0
        self.max_recoveries = 3
        
    def check(self, epoch: int, val_loss: float, val_r2: float, 
              pred_std: Optional[float] = None) -> Dict[str, Any]:
        """
        Check for collapse conditions.
        
        Returns:
            Dictionary with 'triggered', 'reason', 'actions'
        """
        result = {
            'triggered': False,
            'reason': None,
            'actions': []
        }
        
        # Update best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        
        # Check collapse conditions
        collapse_conditions = []
        
        # Skip collapse detection in early epochs (first 5 epochs are warmup)
        # Training loss is typically high in early epochs, which is normal
        if epoch <= 5:
            return result
        
        # Condition 1: Very high recon MSE (near 1.0 for normalized data)
        # Only trigger if loss is extremely high AND we've seen improvement before
        if val_loss > 0.95 and self.best_epoch > 0 and epoch > self.best_epoch + 2:
            # Only trigger if loss increased significantly from best
            if val_loss > self.best_val_loss * 1.5:  # 50% worse than best
                collapse_conditions.append(f"val_loss={val_loss:.4f} > 0.95 and {val_loss/self.best_val_loss:.2f}x worse than best")
        
        # Condition 2: Very low R² (only if we had good R² before)
        if val_r2 < 0.05 and self.best_epoch > 0:
            # Only trigger if R² dropped significantly from a good value
            # (We don't track best R², so use epoch > 5 as proxy for "should have learned something")
            if epoch > 10:
                collapse_conditions.append(f"val_r2={val_r2:.4f} < 0.05 after epoch 10")
        
        # Condition 3: Very low prediction std (collapse to constant)
        if pred_std is not None and pred_std < 0.05 and epoch > 5:
            collapse_conditions.append(f"pred_std={pred_std:.4f} < 0.05")
        
        # Condition 4: No improvement for patience epochs (most reliable indicator)
        if self.no_improve_count >= self.patience and epoch > 5:
            collapse_conditions.append(f"no_improve_count={self.no_improve_count} >= {self.patience}")
        
        if collapse_conditions and self.recovery_count < self.max_recoveries:
            result['triggered'] = True
            result['reason'] = " | ".join(collapse_conditions)
            result['actions'] = ['load_best', 'reduce_lr', 'clip_grad']
            self.recovery_count += 1
        
        return result
    
    def should_recover(self) -> bool:
        """Check if recovery should be attempted."""
        return self.recovery_count < self.max_recoveries


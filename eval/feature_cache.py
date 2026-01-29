"""
Feature Caching Utility for Evaluation

Caches extracted features to speed up repeated evaluations.
Cache key is based on:
- Checkpoint path (or hash)
- Dataset path/identifier
- Feature type
- Sequence length, stride
- Model architecture parameters
"""

import hashlib
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch


def get_cache_dir() -> Path:
    """Get cache directory for features"""
    cache_dir = Path(".cache") / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(
    checkpoint_path: str,
    dataset_path: str,
    dataset_type: str,
    feature_type: str,
    sequence_length: int = 100,
    stride: int = 100,
    n_roi: Optional[int] = None,
    latent_dim: Optional[int] = None,
    **kwargs
) -> str:
    """
    Generate a cache key based on evaluation parameters
    
    Args:
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to dataset
        dataset_type: Type of dataset (HCPA, UKB, etc.)
        feature_type: Type of features (z_concat, z_mean, etc.)
        sequence_length: Sequence length used
        stride: Stride used
        n_roi: Number of ROIs (optional, for model architecture)
        latent_dim: Latent dimension (optional, for model architecture)
        **kwargs: Additional parameters to include in cache key
    
    Returns:
        MD5 hash string for cache key
    """
    # Get checkpoint modification time or hash for cache invalidation
    checkpoint_path_obj = Path(checkpoint_path)
    if checkpoint_path_obj.exists():
        checkpoint_mtime = checkpoint_path_obj.stat().st_mtime
        checkpoint_size = checkpoint_path_obj.stat().st_size
    else:
        checkpoint_mtime = 0
        checkpoint_size = 0
    
    # Create key string from all parameters
    key_parts = [
        str(checkpoint_path),
        str(checkpoint_mtime),
        str(checkpoint_size),
        str(dataset_path),
        str(dataset_type),
        str(feature_type),
        str(sequence_length),
        str(stride),
    ]
    
    if n_roi is not None:
        key_parts.append(f"n_roi_{n_roi}")
    if latent_dim is not None:
        key_parts.append(f"latent_dim_{latent_dim}")
    
    # Add any additional kwargs
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"{key}_{value}")
    
    key_str = "_".join(key_parts)
    
    # Generate MD5 hash
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cached_features(cache_key: str) -> Optional[Tuple[np.ndarray, list, Optional[np.ndarray], Optional[list]]]:
    """
    Load cached features if available
    
    Args:
        cache_key: Cache key string
    
    Returns:
        Tuple of (features, subjects, labels, eventnames) if found, None otherwise
        For backward compatibility, eventnames may be None
    """
    cache_dir = get_cache_dir()
    cache_path = cache_dir / f"features_{cache_key}.pkl"
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Validate cache structure
        if isinstance(data, dict) and 'features' in data and 'subjects' in data:
            features = data['features']
            subjects = data['subjects']
            labels = data.get('labels', None)
            eventnames = data.get('eventnames', None)  # For ABCD
            return features, subjects, labels, eventnames
        else:
            # Old format or corrupted
            return None
    except Exception as e:
        print(f"⚠️  Warning: Failed to load cache: {e}")
        return None


def save_cached_features(
    cache_key: str,
    features: np.ndarray,
    subjects: list,
    labels: Optional[np.ndarray] = None,
    eventnames: Optional[list] = None
):
    """
    Save features to cache
    
    Args:
        cache_key: Cache key string
        features: Feature array [N, D]
        subjects: List of subject IDs
        labels: Optional labels array [N]
        eventnames: Optional list of eventnames for ABCD matching [N]
    """
    cache_dir = get_cache_dir()
    cache_path = cache_dir / f"features_{cache_key}.pkl"
    
    try:
        data = {
            'features': features,
            'subjects': subjects,
            'labels': labels,
            'eventnames': eventnames,  # For ABCD
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Cached features to: {cache_path}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save cache: {e}")


def get_model_params(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract model parameters for cache key"""
    params = {}
    
    # Try to get n_roi
    if hasattr(model, 'n_roi'):
        params['n_roi'] = model.n_roi
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'n_roi'):
        params['n_roi'] = model.base_model.n_roi
    
    # Try to get latent_dim
    if hasattr(model, 'd_latent'):
        params['latent_dim'] = model.d_latent
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'd_latent'):
        params['latent_dim'] = model.base_model.d_latent
    
    return params


"""
Data loader for fMRI BOLD time series data
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import List, Tuple, Optional
import glob
import hashlib
import pickle
import json


def _get_cache_key(data_dir: Path, sequence_length: int, stride: int, 
                   task_filter: Optional[List[str]], max_files: Optional[int],
                   dataset_type: str = "HCPA") -> str:
    """Generate a cache key based on dataset parameters"""
    # Create a hash from the parameters
    key_str = f"{str(data_dir)}_{sequence_length}_{stride}_{task_filter}_{max_files}_{dataset_type}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_dir(data_dir: Path) -> Path:
    """Get cache directory path (in project root or data directory parent)"""
    # Try to find project root (where .git or setup.py might be)
    current = Path(data_dir).resolve()
    for parent in [current, current.parent, current.parent.parent]:
        if (parent / '.git').exists() or (parent / 'setup.py').exists():
            cache_dir = parent / '.cache' / 'datasets'
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
    
    # Fallback: use data directory parent
    cache_dir = current.parent / '.cache' / 'datasets'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_from_cache(cache_path: Path) -> Optional[dict]:
    """Load processed data from cache"""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"⚠️  Warning: Failed to load cache: {e}")
        return None


def _cache_only_enabled() -> bool:
    return os.environ.get("CACHE_ONLY", "").strip().lower() in {"1", "true", "yes", "y"}


def _save_to_cache(cache_path: Path, sequences: List, fc_matrices: Optional[List],
                   metadata: List, mean: Optional[np.ndarray], std: Optional[np.ndarray]):
    """Save processed data to cache"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'sequences': sequences,
                'fc_matrices': fc_matrices,
                'metadata': metadata,
                'mean': mean,
                'std': std
            }, f)
    except Exception as e:
        print(f"⚠️  Warning: Failed to save cache: {e}")


def _compute_stats_from_seqs(sequences: List[np.ndarray],
                             indices: Optional[List[int]] = None
                             ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute (mean, std) over a list of [T, ROI] segments.

    If `indices` is provided, only those segments contribute. Returns
    (None, None) when there is nothing to aggregate.
    """
    seqs = sequences if indices is None else [sequences[i] for i in indices]
    if not seqs:
        return None, None
    all_data = np.concatenate(seqs, axis=0)
    mean = np.mean(all_data, axis=0, keepdims=True)
    std = np.std(all_data, axis=0, keepdims=True) + 1e-8
    return mean, std


class HCPABOLDWithFCDataset(Dataset):
    """
    Dataset for loading HCPA BOLD time series data with paired FC matrices from .npz file
    """
    def __init__(
        self, 
        npz_path: str,
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        task_filter: Optional[List[str]] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            npz_path: Path to .npz file containing 'bold', 'fc', 'sub_id', 'task' arrays
            sequence_length: Length of each sequence segment
            stride: Stride for sliding window
            normalize: Whether to z-normalize the data
            task_filter: List of task names to filter (e.g., ['REST', 'CARIT'])
            max_samples: Maximum number of samples to load (for testing)
        """
        self.npz_path = Path(npz_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        # Load data from npz file
        print(f"Loading data from {self.npz_path}...")
        data = np.load(str(self.npz_path))
        
        bold_data = data['bold'].astype(np.float32)  # [N, T, ROI] - ensure float32
        fc_data = data['fc'].astype(np.float32)  # [N, ROI, ROI] - ensure float32
        sub_ids = data['sub_id']  # [N]
        tasks = data['task']  # [N]
        
        # Show available tasks
        unique_tasks = np.unique([str(t).strip() for t in tasks])
        print(f"Available tasks in dataset: {unique_tasks}")
        
        # Filter by task if specified
        # Note: If task_filter doesn't match, we'll load all samples (npz files often contain pre-filtered data)
        if task_filter is not None:
            # Convert tasks to strings for comparison (handle bytes, numpy strings, etc.)
            tasks_str = np.array([str(t).strip() for t in tasks])
            task_filter_upper = [t.upper().strip() for t in task_filter]
            
            # More flexible matching: check if task string contains any of the filter terms
            # Handles cases like 'REST', 'rest', 'task-REST', 'REST1', etc.
            mask = np.array([any(
                task.upper() in t_str.upper() or 
                t_str.upper() in task.upper() or
                task.upper().replace('-', '').replace('_', '').replace(' ', '') in 
                t_str.upper().replace('-', '').replace('_', '').replace(' ', '')
                for task in task_filter_upper
            ) for t_str in tasks_str])
            
            if mask.sum() == 0:
                # If no matches, check if all tasks are already the desired type
                unique_tasks = np.unique(tasks_str)
                print(f"⚠️  Warning: No samples match task filter {task_filter}")
                print(f"   Available tasks in dataset: {unique_tasks}")
                print(f"   Note: NPZ file may contain pre-filtered data (e.g., all REST)")
                print(f"   Loading all samples without task filtering...")
                # Don't filter - load all samples
            else:
                bold_data = bold_data[mask]
                fc_data = fc_data[mask]
                sub_ids = sub_ids[mask]
                tasks = tasks[mask]
                print(f"Filtered to {len(bold_data)} samples matching tasks: {task_filter}")
                unique_filtered = np.unique([str(t).strip() for t in tasks])
                print(f"   Task distribution: {dict(zip(*np.unique(tasks, return_counts=True)))}")
        
        # Limit number of samples for testing
        if max_samples is not None:
            bold_data = bold_data[:max_samples]
            fc_data = fc_data[:max_samples]
            sub_ids = sub_ids[:max_samples]
            tasks = tasks[:max_samples]
        
        # Create segments from each BOLD sequence
        self.sequences = []
        self.fc_matrices = []
        self.metadata = []
        
        for i in range(len(bold_data)):
            bold_seq = bold_data[i]  # [T, ROI]
            fc_matrix = fc_data[i]  # [ROI, ROI]
            sub_id = sub_ids[i]
            task = tasks[i]
            
            # Create sliding window segments
            segments = self._create_segments(bold_seq)
            
            for segment in segments:
                self.sequences.append(segment)
                self.fc_matrices.append(fc_matrix.copy())  # Same FC for all segments from same sample
                self.metadata.append({
                    'subject': str(sub_id),
                    'task': str(task),
                    'sample_idx': i
                })
        
        print(f"Created {len(self.sequences)} sequence segments from {len(bold_data)} samples")
        
        # Check if we have any sequences
        if len(self.sequences) == 0:
            raise ValueError(
                f"No sequence segments created! "
                f"Check that task_filter matches tasks in the dataset. "
                f"Available tasks: {np.unique(data['task']) if 'task' in data else 'unknown'}"
            )

        self.mean = None
        self.std = None
    
    def _create_segments(self, data: np.ndarray) -> List[np.ndarray]:
        """Create sliding window segments from time series"""
        segments = []
        num_timepoints = data.shape[0]
        
        for start in range(0, num_timepoints - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            segments.append(data[start:end])
        
        return segments
    
    def compute_normalization_stats(self, indices: Optional[List[int]] = None) -> None:
        """Compute and store mean/std over `self.sequences`.

        With no indices, all segments are used.
        """
        if not self.normalize:
            return
        self.mean, self.std = _compute_stats_from_seqs(self.sequences, indices)

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            sequence: [T, N_ROI] tensor
            metadata: dict with file info and FC matrix
        """
        sequence = self.sequences[idx].copy()
        
        # Normalize
        if self.normalize and self.mean is not None and self.std is not None:
            sequence = (sequence - self.mean) / self.std
        
        # Convert to tensor [T, N_ROI] - ensure float32 (not float64)
        sequence = torch.from_numpy(sequence.astype(np.float32))
        
        # Include FC matrix in metadata - ensure float32 (not float64)
        metadata = self.metadata[idx].copy()
        fc_matrix = torch.from_numpy(self.fc_matrices[idx].astype(np.float32))
        metadata['fc_matrix'] = fc_matrix
        
        return sequence, metadata


class HCPABOLDDataset(Dataset):
    """
    Dataset for loading HCPA BOLD time series data
    """
    def __init__(
        self, 
        data_dir: str,
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        task_filter: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ):
        """
        Args:
            data_dir: Path to BOLD directory with CSV files
            sequence_length: Length of each sequence segment
            stride: Stride for sliding window
            normalize: Whether to z-normalize the data
            task_filter: List of task names to filter (e.g., ['REST', 'CARIT'])
            max_files: Maximum number of files to load (for testing)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        # Check cache first
        cache_dir = _get_cache_dir(self.data_dir)
        cache_key = _get_cache_key(self.data_dir, sequence_length, stride, task_filter, max_files, "HCPA")
        cache_path = cache_dir / f"hcpa_{cache_key}.pkl"
        
        cached_data = _load_from_cache(cache_path)
        if cached_data is not None:
            print(f"✓ Loading from cache: {cache_path}")
            self.sequences = cached_data['sequences']
            self.metadata = cached_data['metadata']
            self.mean = None
            self.std = None
            # Ensure metadata has task field (older caches may not)
            for meta in self.metadata:
                if 'task' not in meta:
                    filename = meta.get('file', '')
                    meta['task'] = self._extract_task(filename) if filename else 'unknown'
            print(f"✓ Loaded {len(self.sequences)} cached sequence segments")
            return
        if _cache_only_enabled():
            raise RuntimeError(f"CACHE_ONLY is set but cache missing: {cache_path}")
        
        # Find all files (CSV and TSV)
        csv_files = sorted(glob.glob(str(self.data_dir / "*.csv")) + glob.glob(str(self.data_dir / "*.tsv")))
        
        # Filter by task if specified (case-insensitive)
        # Matches patterns like:
        # - sub-HCA6002236_task-REST_dir-AP_run-2_...
        # - sub-1000195_ses-3_task-rest_...
        # - sub-NDARINV0A4P0LWM_..._task-rest_... (ABCD format)
        if task_filter is not None:
            task_filter_lower = [t.lower() for t in task_filter]
            
            def matches_task(filename_lower: str) -> bool:
                """Check if filename matches any task in filter"""
                for task in task_filter_lower:
                    # Match direct task name (e.g., 'rest')
                    if task in filename_lower:
                        return True
                    # Match 'task-{taskname}' pattern (e.g., 'task-rest')
                    if f'task-{task}' in filename_lower:
                        return True
                    # Match '{taskname}' as standalone word (for flexibility)
                    if f'_{task}_' in filename_lower or filename_lower.endswith(f'_{task}'):
                        return True
                return False
            
            csv_files = [f for f in csv_files if matches_task(f.lower())]
        
        # Limit number of files for testing
        if max_files is not None:
            csv_files = csv_files[:max_files]
        
        print(f"Found {len(csv_files)} files")
        
        # Load all data and create segments
        self.sequences = []
        self.metadata = []
        
        for csv_file in csv_files:
            try:
                data = self._load_csv(csv_file)
                segments = self._create_segments(data)
                
                for segment in segments:
                    self.sequences.append(segment)
                    self.metadata.append({
                        'file': os.path.basename(csv_file),
                        'subject': self._extract_subject_id(csv_file),
                        'task': self._extract_task(csv_file)
                    })
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        print(f"Created {len(self.sequences)} sequence segments")
        
        # Check if we have any data
        if len(self.sequences) == 0:
            raise ValueError(
                f"No sequence segments were created from {len(csv_files)} files in {self.data_dir}. "
                f"Please check:\n"
                f"  1. File format: Expected CSV or TSV files\n"
                f"  2. File naming: Files should match the expected patterns\n"
                f"  3. Task filter: If using --task_filter, ensure files match the filter\n"
                f"  4. Directory path: Verify the data directory is correct"
            )
        
        self.mean = None
        self.std = None

        # Save to cache
        print(f"💾 Saving to cache: {cache_path}")
        _save_to_cache(cache_path, self.sequences, None, self.metadata, None, None)
    
    def _load_csv(self, csv_file: str) -> np.ndarray:
        """Load a single CSV or TSV file and return as numpy array"""
        # Detect delimiter based on file extension
        file_ext = Path(csv_file).suffix.lower()
        delimiter = '\t' if file_ext == '.tsv' else ','
        df = pd.read_csv(csv_file, index_col=0, sep=delimiter)
        return df.values.astype(np.float32)
    
    def _create_segments(self, data: np.ndarray) -> List[np.ndarray]:
        """Create sliding window segments from time series"""
        segments = []
        num_timepoints = data.shape[0]
        
        for start in range(0, num_timepoints - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            segments.append(data[start:end])
        
        return segments
    
    def _extract_subject_id(self, filename: str) -> str:
        """Extract subject ID from filename"""
        basename = os.path.basename(filename)
        return basename.split('_')[0]
    
    def _extract_task(self, filename: str) -> str:
        """Extract task name from filename"""
        basename = os.path.basename(filename)
        parts = basename.split('_')
        for part in parts:
            if part.startswith('task-'):
                return part.replace('task-', '')
        return 'unknown'
    
    def compute_normalization_stats(self, indices: Optional[List[int]] = None) -> None:
        """Compute and store mean/std over `self.sequences`.

        With no indices, all segments are used.
        """
        if not self.normalize:
            return
        self.mean, self.std = _compute_stats_from_seqs(self.sequences, indices)

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            sequence: [T, N_ROI] tensor
            metadata: dict with file info
        """
        sequence = self.sequences[idx].copy()
        
        # Normalize
        if self.normalize and self.mean is not None and self.std is not None:
            sequence = (sequence - self.mean) / self.std
        
        # Convert to tensor [T, N_ROI] - ensure float32 (not float64)
        sequence = torch.from_numpy(sequence.astype(np.float32))
        
        metadata = self.metadata[idx]
        if 'task' not in metadata:
            filename = metadata.get('file', '')
            metadata['task'] = self._extract_task(filename) if filename else 'unknown'
        return sequence, metadata


class UKBiobankBOLDDataset(HCPABOLDDataset):
    """
    Dataset for loading UK Biobank BOLD time series data
    Inherits from HCPABOLDDataset since UKB has the same structure (CSV files)
    
    The only difference is the metadata tagging for output naming purposes.
    """
    def __init__(
        self, 
        data_dir: str,
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        task_filter: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ):
        """
        Args:
            data_dir: Path to BOLD directory with CSV files (e.g., ../data/ukb/AAL_116/BOLD)
            sequence_length: Length of each sequence segment
            stride: Stride for sliding window
            normalize: Whether to z-normalize the data
            task_filter: List of task names to filter (e.g., ['REST']). Case-insensitive.
            max_files: Maximum number of files to load (for testing)
        """
        print(f"Loading UKB data from {data_dir}...")
        
        # Use parent class initialization
        super().__init__(
            data_dir=data_dir,
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            task_filter=task_filter,
            max_files=max_files
        )
        
        # Update metadata to mark as UKB dataset
        for meta in self.metadata:
            meta['dataset'] = 'UKB'
        
        print(f"✓ Loaded UKB dataset with {len(self.sequences)} segments from {len(set([m['file'] for m in self.metadata]))} subjects")


class UKBiobankBOLDWithFCDataset(Dataset):
    """
    Dataset for loading BOLD time series with pre-computed FC matrices
    Loads BOLD from CSV files and FC from separate CSV files
    
    Works for both UKB and HCPA datasets (same directory structure)
    
    Directory structure:
        data/AAL_116/
        ├── BOLD/
        │   ├── subject_001.csv
        │   ├── subject_002.csv
        │   └── ...
        └── FC/
            ├── subject_001.csv
            ├── subject_002.csv
            └── ...
    """
    def __init__(
        self, 
        bold_dir: str,
        fc_dir: Optional[str] = None,
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        task_filter: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ):
        """
        Args:
            bold_dir: Path to BOLD directory with CSV files
            fc_dir: Path to FC directory with CSV files (if None, will look for ../FC relative to bold_dir)
            sequence_length: Length of each sequence segment
            stride: Stride for sliding window
            normalize: Whether to z-normalize the data
            task_filter: List of task names to filter (e.g., ['REST']). Case-insensitive.
            max_files: Maximum number of files to load (for testing)
        """
        self.bold_dir = Path(bold_dir)
        
        # Auto-detect FC directory if not provided
        if fc_dir is None:
            self.fc_dir = self.bold_dir.parent / 'FC'
        else:
            self.fc_dir = Path(fc_dir)
        
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        # Auto-detect dataset type from path (for correct cache naming)
        # Check if path contains 'hcpa' or 'abcd' (case-insensitive)
        path_str = str(self.bold_dir).lower()
        if 'hcpa' in path_str:
            self.dataset_type = "HCPA"
            cache_prefix = "hcpa"
        elif 'abcd' in path_str:
            self.dataset_type = "ABCD"
            cache_prefix = "abcd"
        else:
            self.dataset_type = "UKB"
            cache_prefix = "ukb"
        
        # Check cache first
        cache_dir = _get_cache_dir(self.bold_dir)
        cache_key = _get_cache_key(self.bold_dir, sequence_length, stride, task_filter, max_files, self.dataset_type)
        cache_path = cache_dir / f"{cache_prefix}_{cache_key}.pkl"
        
        cached_data = _load_from_cache(cache_path)
        if cached_data is not None:
            print(f"✓ Loading from cache: {cache_path}")
            self.sequences = cached_data['sequences']
            self.fc_matrices = cached_data.get('fc_matrices', [])
            self.metadata = cached_data['metadata']
            self.mean = None
            self.std = None
            # Ensure metadata has correct dataset type
            for meta in self.metadata:
                if 'dataset' not in meta:
                    meta['dataset'] = self.dataset_type
                if 'task' not in meta:
                    filename = meta.get('file', '')
                    meta['task'] = self._extract_task(filename) if filename else 'unknown'
            print(f"✓ Loaded {len(self.sequences)} cached sequence segments")
            return
        if _cache_only_enabled():
            raise RuntimeError(f"CACHE_ONLY is set but cache missing: {cache_path}")
        
        print(f"Loading data from {self.bold_dir}...")
        if self.fc_dir.exists():
            print(f"Loading FC matrices from {self.fc_dir}...")
        else:
            print(f"⚠️  Warning: FC directory not found at {self.fc_dir}")
            print(f"   Will compute FC matrices on-the-fly")
        
        # Find all BOLD files (CSV and TSV)
        bold_files = sorted(glob.glob(str(self.bold_dir / "*.csv")) + glob.glob(str(self.bold_dir / "*.tsv")))
        
        # Filter by task if specified (case-insensitive)
        # Handles patterns like: 'task-rest', 'task-REST', 'REST', 'rest', etc.
        if task_filter is not None:
            task_filter_lower = [t.lower() for t in task_filter]
            original_count = len(bold_files)
            
            def matches_task(filename_lower: str) -> bool:
                """Check if filename matches any task in filter"""
                for task in task_filter_lower:
                    # Match direct task name (e.g., 'rest')
                    if task in filename_lower:
                        return True
                    # Match 'task-{taskname}' pattern (e.g., 'task-rest')
                    if f'task-{task}' in filename_lower:
                        return True
                    # Match '{taskname}' as standalone word (for flexibility)
                    if f'_{task}_' in filename_lower or filename_lower.endswith(f'_{task}'):
                        return True
                return False
            
            bold_files = [f for f in bold_files if matches_task(f.lower())]
            print(f"  Task filter: {task_filter}")
            print(f"  Filtered {original_count} files to {len(bold_files)} matching files")
        
        # Limit number of files for testing
        if max_files is not None:
            bold_files = bold_files[:max_files]
        
        print(f"Found {len(bold_files)} BOLD files")
        
        # Load all data and create segments
        self.sequences = []
        self.fc_matrices = []
        self.metadata = []
        
        # Track data quality issues
        n_low_var = 0
        n_nan_inf = 0
        n_fc_computed = 0
        n_fc_loaded = 0
        
        print(f"Processing {len(bold_files)} subjects...")
        for i, bold_file in enumerate(bold_files):
            try:
                # Load BOLD
                bold_data = self._load_csv(bold_file)
                
                # Check for data quality issues
                if np.isnan(bold_data).any() or np.isinf(bold_data).any():
                    n_nan_inf += 1
                
                std_vals = bold_data.std(axis=0)
                if (std_vals < 1e-6).any():
                    n_low_var += 1
                
                # Load or compute FC
                subject_id = self._extract_subject_id(bold_file)
                bold_basename = os.path.basename(bold_file)
                bold_basename_no_ext = bold_basename.split('.')[0]
                
                # Try multiple FC file patterns to match different naming conventions
                # Support both CSV and TSV files
                file_ext = Path(bold_file).suffix.lower()  # Get extension from BOLD file
                fc_file = None
                fc_patterns = [
                    # Pattern 1: Simple subject_id with same extension as BOLD file
                    f"{subject_id}{file_ext}",
                    # Pattern 2: Match BOLD filename but replace _time_series with _connectivity_matrix
                    bold_basename_no_ext.replace('_time_series', '_connectivity_matrix') + file_ext,
                    bold_basename_no_ext.replace('_time_series', '_connectivity_matrix') + '.csv',  # Also try CSV
                    bold_basename_no_ext.replace('_time_series', '_connectivity_matrix') + '.tsv',  # Also try TSV
                    bold_basename_no_ext.replace('_AAL_time_series', '_AAL_connectivity_matrix') + file_ext,
                    bold_basename_no_ext.replace('_AAL_time_series', '_AAL_connectivity_matrix') + '.csv',
                    bold_basename_no_ext.replace('_AAL_time_series', '_AAL_connectivity_matrix') + '.tsv',
                    # Pattern 3: Match with ses- added (for files that have ses- in FC but not in BOLD)
                    bold_basename_no_ext.replace('_task-', '_ses-2_task-').replace('_time_series', '_connectivity_matrix') + file_ext,
                    bold_basename_no_ext.replace('_task-', '_ses-2_task-').replace('_time_series', '_connectivity_matrix') + '.csv',
                    bold_basename_no_ext.replace('_task-', '_ses-2_task-').replace('_time_series', '_connectivity_matrix') + '.tsv',
                    bold_basename_no_ext.replace('_task-', '_ses-2_task-').replace('_AAL_time_series', '_AAL_connectivity_matrix') + file_ext,
                    bold_basename_no_ext.replace('_task-', '_ses-2_task-').replace('_AAL_time_series', '_AAL_connectivity_matrix') + '.csv',
                    bold_basename_no_ext.replace('_task-', '_ses-2_task-').replace('_AAL_time_series', '_AAL_connectivity_matrix') + '.tsv',
                    # Pattern 4: Try with uppercase REST
                    bold_basename_no_ext.replace('_task-rest', '_task-REST').replace('_time_series', '_connectivity_matrix') + file_ext,
                    bold_basename_no_ext.replace('_task-rest', '_task-REST').replace('_time_series', '_connectivity_matrix') + '.csv',
                    bold_basename_no_ext.replace('_task-rest', '_task-REST').replace('_time_series', '_connectivity_matrix') + '.tsv',
                    bold_basename_no_ext.replace('_task-rest', '_task-REST').replace('_AAL_time_series', '_AAL_connectivity_matrix') + file_ext,
                    bold_basename_no_ext.replace('_task-rest', '_task-REST').replace('_AAL_time_series', '_AAL_connectivity_matrix') + '.csv',
                    bold_basename_no_ext.replace('_task-rest', '_task-REST').replace('_AAL_time_series', '_AAL_connectivity_matrix') + '.tsv',
                ]
                
                # Try each pattern
                for pattern in fc_patterns:
                    candidate = self.fc_dir / pattern
                    if candidate.exists():
                        fc_file = candidate
                        break
                
                # If still not found, try fuzzy matching by subject ID (search for any FC file starting with subject_id)
                if fc_file is None or not fc_file.exists():
                    # Extract just the numeric/alphanumeric part after 'sub-' for matching
                    if subject_id.startswith('sub-'):
                        subject_base = subject_id.replace('sub-', '')
                        # Search for any FC file that starts with subject_id and contains 'connectivity'
                        # This handles cases where FC files have different desc- or ses- values
                        # Try both CSV and TSV
                        fc_candidates = list(self.fc_dir.glob(f"{subject_id}*connectivity*.csv")) + \
                                       list(self.fc_dir.glob(f"{subject_id}*connectivity*.tsv"))
                        if not fc_candidates:
                            # Also try with just the base (without sub- prefix)
                            fc_candidates = list(self.fc_dir.glob(f"sub-{subject_base}*connectivity*.csv")) + \
                                           list(self.fc_dir.glob(f"sub-{subject_base}*connectivity*.tsv"))
                        if fc_candidates:
                            fc_file = fc_candidates[0]  # Use first match
                
                if fc_file and fc_file.exists():
                    fc_matrix = self._load_csv(fc_file)
                    n_fc_loaded += 1
                else:
                    # Compute FC if not found
                    fc_matrix = self._compute_fc_matrix(bold_data)
                    n_fc_computed += 1
                
                # Create sliding window segments
                segments = self._create_segments(bold_data)
                
                for segment in segments:
                    self.sequences.append(segment)
                    self.fc_matrices.append(fc_matrix.copy())
                    self.metadata.append({
                        'file': os.path.basename(bold_file),
                        'subject': subject_id,
                        'task': self._extract_task(bold_file),
                        'dataset': self.dataset_type
                    })
                
                # Progress update
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i+1}/{len(bold_files)} subjects...")
            
            except Exception as e:
                print(f"Error loading {bold_file}: {e}")
                continue
        
        print(f"\n✓ Created {len(self.sequences)} sequence segments from {len(bold_files)} subjects")
        
        # Check if we have any data
        if len(self.sequences) == 0:
            raise ValueError(
                f"No sequence segments were created from {len(bold_files)} BOLD files in {self.bold_dir}. "
                f"Please check:\n"
                f"  1. File format: Expected CSV or TSV files\n"
                f"  2. File naming: Files should match the expected patterns\n"
                f"  3. Task filter: If using --task_filter, ensure files match the filter\n"
                f"  4. Directory path: Verify the data directory is correct"
            )
        
        # Report data quality issues
        if n_low_var > 0:
            print(f"  ⚠️  {n_low_var} subjects had low-variance ROIs (handled automatically)")
        if n_nan_inf > 0:
            print(f"  ⚠️  {n_nan_inf} subjects had NaN/Inf values (cleaned automatically)")
        if n_fc_computed > 0:
            print(f"  ℹ️  Computed FC for {n_fc_computed} subjects (not found in FC directory)")
        if n_fc_loaded > 0:
            print(f"  ℹ️  Loaded pre-computed FC for {n_fc_loaded} subjects")
        
        self.mean = None
        self.std = None

        # Save to cache
        print(f"💾 Saving to cache: {cache_path}")
        _save_to_cache(cache_path, self.sequences, self.fc_matrices, self.metadata, None, None)
    
    def _load_csv(self, csv_file: str) -> np.ndarray:
        """Load a single CSV or TSV file and return as numpy array"""
        # Detect delimiter based on file extension
        file_ext = Path(csv_file).suffix.lower()
        delimiter = '\t' if file_ext == '.tsv' else ','
        df = pd.read_csv(csv_file, index_col=0, sep=delimiter)
        return df.values.astype(np.float32)
    
    def _compute_fc_matrix(self, bold_timeseries: np.ndarray) -> np.ndarray:
        """Compute FC matrix as Pearson correlation with robust handling of low-variance ROIs"""
        # Check for NaN/Inf values
        if np.isnan(bold_timeseries).any() or np.isinf(bold_timeseries).any():
            bold_timeseries = np.nan_to_num(bold_timeseries, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute std and identify low-variance ROIs
        std_vals = bold_timeseries.std(axis=0)
        low_var_mask = std_vals < 1e-6  # Threshold for near-zero variance
        
        # Replace low-variance ROIs with small noise to avoid division by zero
        if low_var_mask.any():
            bold_timeseries = bold_timeseries.copy()
            for roi_idx in np.where(low_var_mask)[0]:
                bold_timeseries[:, roi_idx] = np.random.randn(bold_timeseries.shape[0]) * 1e-6
        
        # Standardize with safe division
        mean_vals = bold_timeseries.mean(axis=0)
        std_vals = bold_timeseries.std(axis=0)
        std_vals = np.where(std_vals < 1e-8, 1e-8, std_vals)  # Avoid division by zero
        bold_norm = (bold_timeseries - mean_vals) / std_vals
        
        # Compute correlation with warnings suppressed
        with np.errstate(divide='ignore', invalid='ignore'):
            fc = np.corrcoef(bold_norm.T)
        
        # Handle NaN values (from constant signals)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure symmetric and positive
        fc = (fc + fc.T) / 2  # Symmetrize
        fc = np.abs(fc)  # Unsigned graph
        
        # Ensure diagonal is 1
        np.fill_diagonal(fc, 1.0)
        
        return fc.astype(np.float32)
    
    def _create_segments(self, data: np.ndarray) -> List[np.ndarray]:
        """Create sliding window segments from time series"""
        segments = []
        num_timepoints = data.shape[0]
        
        for start in range(0, num_timepoints - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            segments.append(data[start:end])
        
        return segments
    
    def _extract_subject_id(self, filename: str) -> str:
        """
        Extract subject ID from filename.
        For UKB format: sub-XXXXX_task-... -> extract sub-XXXXX
        For HCPA format: subject_001 -> extract subject_001
        """
        basename = os.path.basename(filename)
        # Remove extension
        basename_no_ext = basename.split('.')[0]
        
        # If it starts with 'sub-', extract just the subject part (sub-XXXXX)
        if basename_no_ext.startswith('sub-'):
            # Extract 'sub-XXXXX' part (before first underscore after sub-)
            parts = basename_no_ext.split('_')
            if len(parts) > 0:
                return parts[0]  # Returns 'sub-XXXXX'
        
        # Fallback: return first part before underscore (for HCPA format: subject_001)
        return basename_no_ext.split('_')[0]
    
    def compute_normalization_stats(self, indices: Optional[List[int]] = None) -> None:
        """Compute and store mean/std over `self.sequences`.

        With no indices, all segments are used.
        """
        if not self.normalize:
            return
        self.mean, self.std = _compute_stats_from_seqs(self.sequences, indices)

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            sequence: [T, N_ROI] tensor
            metadata: dict with file info and FC matrix
        """
        sequence = self.sequences[idx].copy()
        
        # Normalize
        if self.normalize and self.mean is not None and self.std is not None:
            sequence = (sequence - self.mean) / self.std
        
        # Convert to tensor [T, N_ROI] - ensure float32
        sequence = torch.from_numpy(sequence.astype(np.float32))
        
        # Include FC matrix in metadata - ensure float32
        metadata = self.metadata[idx].copy()
        if 'task' not in metadata:
            filename = metadata.get('file', '')
            metadata['task'] = self._extract_task(filename) if filename else 'unknown'
        fc_matrix = torch.from_numpy(self.fc_matrices[idx].astype(np.float32))
        metadata['fc_matrix'] = fc_matrix
        
        return sequence, metadata

    @staticmethod
    def _extract_task(filename: str) -> str:
        """Extract task name from filename (handles task-rest, task-REST, etc.)"""
        basename = os.path.basename(filename)
        parts = basename.split('_')
        for part in parts:
            if part.startswith('task-'):
                return part.replace('task-', '')
        return 'unknown'


class ABCDBOLDWithFCDataset(UKBiobankBOLDWithFCDataset):
    """
    Dataset for loading ABCD BOLD time series with optional FC matrices.

    Expected directory structure:
        data/abcd/
        ├── BOLD/
        │   ├── sub-NDARINV0A4P0LWM_ses-baselineYear1Arm1_task-rest_run-04_space-MNI152NLin2009cAsym_desc-interpolated_bold_AAL_time_series.csv
        │   ├── sub-..._task-rest_...csv
        │   └── ...
        └── FC/   (optional)
            ├── <matching CSVs or subject-level FC>
            └── ...

    Notes:
    - If FC files are present and share the same basename as BOLD files (minus extension),
      they will be loaded; otherwise FC is computed on-the-fly per subject from BOLD.
    - Task filtering: Use task_filter=['REST'] to filter for files containing 'task-rest' in filename.
      The filter matches both 'rest' and 'task-rest' patterns (case-insensitive).
    """

    def __init__(
        self,
        bold_dir: str,
        fc_dir: Optional[str] = None,
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        task_filter: Optional[List[str]] = None,
        max_files: Optional[int] = None,
    ):
        # Override cache key for ABCD
        self.bold_dir = Path(bold_dir)
        if fc_dir is None:
            self.fc_dir = self.bold_dir.parent / 'FC'
        else:
            self.fc_dir = Path(fc_dir)
        
        # Set attributes that are needed even when loading from cache
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        # Check cache first with ABCD key
        cache_dir = _get_cache_dir(self.bold_dir)
        cache_key = _get_cache_key(self.bold_dir, sequence_length, stride, task_filter, max_files, "ABCD")
        cache_path = cache_dir / f"abcd_{cache_key}.pkl"
        
        cached_data = _load_from_cache(cache_path)
        if cached_data is not None:
            print(f"✓ Loading from cache: {cache_path}")
            self.sequences = cached_data['sequences']
            self.fc_matrices = cached_data.get('fc_matrices', [])
            self.metadata = cached_data['metadata']
            self.mean = None
            self.std = None
            # Update metadata to mark as ABCD dataset
            for meta in self.metadata:
                meta['dataset'] = 'ABCD'
            print(f"✓ Loaded {len(self.sequences)} cached sequence segments")
            return
        
        super().__init__(
            bold_dir=bold_dir,
            fc_dir=fc_dir,
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            task_filter=task_filter,
            max_files=max_files,
        )

        # Update metadata to mark as ABCD dataset and extract session info
        for meta in self.metadata:
            meta['dataset'] = 'ABCD'
            # Extract session from filename if present
            filename = meta.get('file', '')
            session = self._extract_session_from_filename(filename)
            if session:
                meta['session'] = session
                # Also store normalized eventname format for matching
                meta['eventname'] = self._session_to_eventname(session)
        
        # Save to cache with ABCD key
        print(f"💾 Saving to cache: {cache_path}")
        _save_to_cache(cache_path, self.sequences, self.fc_matrices, self.metadata, self.mean, self.std)
    
    def _extract_subject_id(self, filename: str) -> str:
        """
        Extract subject ID from ABCD BOLD filename.
        Format: sub-NDARINV0A4P0LWM_ses-...
        Returns: NDARINV0A4P0LWM (without 'sub-' prefix)
        """
        basename = os.path.basename(filename)
        # Extract subject ID from 'sub-XXXXX' pattern
        if basename.startswith('sub-'):
            subject_part = basename.split('_')[0]  # 'sub-NDARINV0A4P0LWM'
            return subject_part.replace('sub-', '')
        # Fallback: use first part before underscore
        return basename.split('_')[0]
    
    def _extract_session_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract session from ABCD BOLD filename.
        Format: ..._ses-2YearFollowUpYArm1_...
        Returns: 2YearFollowUpYArm1 (without 'ses-' prefix) or None
        """
        basename = os.path.basename(filename)
        parts = basename.split('_')
        for part in parts:
            if part.startswith('ses-'):
                return part.replace('ses-', '')
        return None
    
    def _session_to_eventname(self, session: str) -> str:
        """
        Convert ABCD session format to eventname format for metadata matching.
        
        Examples:
            baselineYear1Arm1 -> baseline_year_1_arm_1
            2YearFollowUpYArm1 -> 2_year_follow_up_y_arm_1
            1YearFollowUpYArm1 -> 1_year_follow_up_y_arm_1
            3YearFollowUpYArm1 -> 3_year_follow_up_y_arm_1
        
        Strategy: Insert underscores before capital letters and numbers, then lowercase.
        """
        import re
        # Insert underscore before capital letters and numbers
        # e.g., "2YearFollowUpYArm1" -> "2_Year_Follow_Up_Y_Arm_1"
        normalized = re.sub(r'([a-z])([A-Z0-9])', r'\1_\2', session)
        normalized = re.sub(r'([A-Z0-9])([A-Z][a-z])', r'\1_\2', normalized)
        # Convert to lowercase
        return normalized.lower()


class ABCDBOLDDataset(HCPABOLDDataset):
    """
    Dataset for loading ABCD BOLD time series without explicit FC matrices.

    Uses the same CSV loading and task filtering logic as HCPABOLDDataset, but
    tags metadata with dataset='ABCD' for downstream analysis.
    
    Task filtering: Use task_filter=['REST'] to filter for files containing 'task-rest' in filename.
    The filter matches both 'rest' and 'task-rest' patterns (case-insensitive).
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        task_filter: Optional[List[str]] = None,
        max_files: Optional[int] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            task_filter=task_filter,
            max_files=max_files,
        )

        # Tag ABCD dataset in metadata
        for meta in self.metadata:
            meta['dataset'] = 'ABCD'

def get_dataloaders(
    data_dir: Optional[str] = None,
    npz_path: Optional[str] = None,
    batch_size: int = 32,
    sequence_length: int = 100,
    stride: int = 50,
    train_split: float = 0.8,
    num_workers: int = 0,
    task_filter: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    max_samples: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    dataset_type: str = 'HCPA'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Path to BOLD directory with CSV files (if using CSV files)
        npz_path: Path to .npz file with 'bold', 'fc', 'sub_id', 'task' (if using npz file)
        max_files: Max CSV files to load (for CSV mode)
        max_samples: Max samples to load (for npz mode)
        distributed: Whether to use DistributedSampler
        rank: Process rank for distributed training
        world_size: World size for distributed training
        dataset_type: 'HCPA' or 'UKB' - determines which dataset class to use
    """
    # Load full dataset
    if npz_path is not None:
        # Use npz file (with paired FC matrices) - only HCPA for now
        dataset = HCPABOLDWithFCDataset(
            npz_path=npz_path,
            sequence_length=sequence_length,
            stride=stride,
            normalize=True,
            task_filter=task_filter,
            max_samples=max_samples
        )
    elif data_dir is not None:
        # Use CSV files (both HCPA and UKB use this path)
        if dataset_type.upper() == 'UKB':
            # For UKB with separate BOLD/FC directories
            bold_dir = Path(data_dir)
            if (bold_dir.parent / 'FC').exists():
                # UKB with FC matrices
                dataset = UKBiobankBOLDWithFCDataset(
                    bold_dir=data_dir,
                    sequence_length=sequence_length,
                    stride=stride,
                    normalize=True,
                    max_files=max_files
                )
            else:
                # UKB without FC matrices
                dataset = UKBiobankBOLDDataset(
                    data_dir=data_dir,
                    sequence_length=sequence_length,
                    stride=stride,
                    normalize=True,
                    max_files=max_files
                )
        else:  # HCPA
            dataset = HCPABOLDDataset(
                data_dir=data_dir,
                sequence_length=sequence_length,
                stride=stride,
                normalize=True,
                task_filter=task_filter,
                max_files=max_files
            )
    else:
        raise ValueError("Either data_dir or npz_path must be provided")
    
    # Split by subject — every segment from a given subject stays in one fold.
    subjects = [m['subject'] for m in dataset.metadata]
    unique_subjects = sorted(set(subjects))
    rng = np.random.default_rng(42)
    shuffled_subjects = list(unique_subjects)
    rng.shuffle(shuffled_subjects)
    n_train_subj = int(len(shuffled_subjects) * train_split)
    train_subjects = set(shuffled_subjects[:n_train_subj])

    train_indices = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_indices = [i for i, s in enumerate(subjects) if s not in train_subjects]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Subject split: {len(train_subjects)} train / "
          f"{len(unique_subjects) - len(train_subjects)} val "
          f"({len(train_indices)} / {len(val_indices)} segments)")

    if hasattr(dataset, 'compute_normalization_stats'):
        dataset.compute_normalization_stats(train_indices)

    # Create distributed samplers if needed
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    data_dir = "datasets/hcpa/AAL_116/BOLD"
    
    # Test with small subset
    dataset = HCPABOLDDataset(
        data_dir=data_dir,
        sequence_length=100,
        stride=50,
        task_filter=['REST'],
        max_files=5
    )
    dataset.compute_normalization_stats()

    print(f"\nDataset size: {len(dataset)}")
    
    # Test a single sample
    sequence, metadata = dataset[0]
    print(f"Sequence shape: {sequence.shape}")
    print(f"Metadata: {metadata}")
    
    # Test dataloader
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        sequence_length=100,
        task_filter=['REST'],
        max_files=10
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch, metadata in train_loader:
        print(f"Batch shape: {batch.shape}")  # [B, T, N_ROI]
        break


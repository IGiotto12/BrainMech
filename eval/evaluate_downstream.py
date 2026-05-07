"""



Downstream Task Evaluation for Foundation Model







This script evaluates a pre-trained Koopman Foundation Model on downstream tasks



using out-of-domain datasets (ABIDE, PPMI, etc.).







The script:



1. Loads a pre-trained foundation model



2. Loads downstream dataset (different from training data)



3. Extracts features from the model



4. Trains simple classifiers/regressors on extracted features



5. Evaluates performance with proper subject-level splits







Usage:



    # Classification task (e.g., disease diagnosis)



    python scripts/evaluate_downstream.py \



        --checkpoint results/model.pt \



        --dataset ABIDE \



        --data_dir data/ABIDE/BOLD \



        --task_type classification \



        --labels_file data/ABIDE/labels.csv \



        --label_column diagnosis







    # Regression task (e.g., age prediction)



    python scripts/evaluate_downstream.py \



        --checkpoint results/model.pt \



        --dataset PPMI \



        --data_dir data/PPMI/BOLD \



        --task_type regression \



        --labels_file data/PPMI/labels.csv \



        --label_column age



"""







import sys



from pathlib import Path



sys.path.insert(0, str(Path(__file__).parent.parent))







import torch



import numpy as np



import pandas as pd



from sklearn.svm import SVC



from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor



from sklearn.linear_model import LogisticRegression, Ridge



from sklearn.neural_network import MLPRegressor



from sklearn.metrics import (



    accuracy_score, f1_score, classification_report, confusion_matrix,



    mean_absolute_error, mean_squared_error, r2_score



)



from sklearn.model_selection import train_test_split



import json



from tqdm import tqdm



import argparse



from typing import Dict, List, Tuple, Optional, Union, Any



from torch.utils.data import Dataset







from models.brainmech import BrainMech



# Try to import OLD model from parallel directory (../OLD/) or subdirectory (OLD/)



try:



    # First try parallel directory (../OLD/) - OLD is parallel to koopmanFM_new



    project_root = Path(__file__).parent.parent  # koopmanFM_new/



    old_path = project_root.parent / "OLD"  # ../OLD/



    if old_path.exists():



        # Add parent directory to path so we can import OLD



        if str(old_path.parent) not in sys.path:



            sys.path.insert(0, str(old_path.parent))



        from OLD.models import KoopmanBrainModelOld



    else:



        # Fallback to subdirectory (OLD/ inside project)



        from OLD.models import KoopmanBrainModelOld



except ImportError:



    KoopmanBrainModelOld = None











# ============================================================================



# Dataset Loaders (Implement for each dataset)



# ============================================================================







class BaseDownstreamDataset(Dataset):



    """



    Base class for downstream evaluation datasets.



    Implement dataset-specific loaders by inheriting from this class.



    """



    



    def __init__(



        self,



        data_dir: str,



        labels_file: Optional[str] = None,



        label_column: Optional[str] = None,



        sequence_length: int = 100,



        stride: int = 50,



        normalize: bool = True,



        max_files: Optional[int] = None



    ):



        """



        Args:



            data_dir: Path to BOLD time series data directory



            labels_file: Path to CSV file with labels (subject_id, label_column, ...)



            label_column: Column name in labels_file containing the target labels



            sequence_length: Length of time series segments



            stride: Stride for sliding window



            normalize: Whether to z-normalize data



            max_files: Maximum number of files to load (for testing)



        """



        self.data_dir = Path(data_dir)



        self.sequence_length = sequence_length



        self.stride = stride



        self.normalize = normalize



        



        # Load labels if provided



        self.labels_dict = {}



        self.label_column = label_column



        self.auto_extract_labels = False



        



        if labels_file and label_column:



            labels_df = pd.read_csv(labels_file)



            if 'subject_id' not in labels_df.columns:



                raise ValueError("labels_file must contain 'subject_id' column")



            if label_column not in labels_df.columns:



                raise ValueError(f"label_column '{label_column}' not found in labels_file")



            self.labels_dict = dict(zip(



                labels_df['subject_id'].astype(str),



                labels_df[label_column]



            ))



            print(f"Loaded {len(self.labels_dict)} labels from {labels_file}")



        elif label_column and not labels_file:



            # Auto-extract labels from folder names (for diagnosis)



            self.auto_extract_labels = True



            print(f"Will auto-extract '{label_column}' labels from folder names")



        



        # Load data (implement in subclass)



        self.sequences = []



        self.metadata = []



        self._load_data(max_files)



        



        # Per-sequence (within-subject) z-normalization is applied in __getitem__
        # when self.mean is None — avoids any cross-subject leakage.
        self.mean = None



        self.std = None



    



    def _load_data(self, max_files: Optional[int] = None):



        """



        Load data from data_dir. Implement in subclass.



        Should populate self.sequences and self.metadata.



        """



        raise NotImplementedError("Subclass must implement _load_data()")



    



    def _extract_subject_id(self, filename: str) -> str:



        """Extract subject ID from filename. Override if needed."""



        # Default: assume subject ID is first part of filename



        return Path(filename).stem.split('_')[0]



    



    def __len__(self) -> int:



        return len(self.sequences)



    



    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:



        sequence = self.sequences[idx].copy()



        



        # Normalize



        if self.normalize:



            if self.mean is not None:



                sequence = (sequence - self.mean) / self.std



            else:



                # Per-sequence (within-subject) z-normalization — leakage-free fallback
                seq_mean = np.mean(sequence, axis=0, keepdims=True)
                seq_std = np.std(sequence, axis=0, keepdims=True) + 1e-8
                sequence = (sequence - seq_mean) / seq_std



        



        # Convert to tensor [T, N_ROI]



        sequence = torch.from_numpy(sequence).float()



        



        metadata = self.metadata[idx].copy()



        



        # Add label if available



        subject_id = metadata.get('subject', '')



        if subject_id in self.labels_dict:



            metadata['label'] = self.labels_dict[subject_id]



        elif self.auto_extract_labels:



            # Try to extract label from subject_id or folder name



            label = self._extract_label_from_subject_id(subject_id, metadata)



            if label is not None:



                metadata['label'] = label



        



        return sequence, metadata



    



    def _extract_label_from_subject_id(self, subject_id: str, metadata: Dict) -> Optional[Any]:



        """Extract label from subject ID or folder name. Override in subclass if needed."""



        return None











class ABIDEDataset(BaseDownstreamDataset):



    """



    Dataset loader for ABIDE (Autism Brain Imaging Data Exchange) dataset



    



    Automatically filters for AAL116 parcellation files and prefers timeseries over



    correlation matrices. This matches the model's expected input (116 ROIs).



    



    Supports two structures:



    1. Flat structure:



        data_dir/



            ├── sub-{subject_id}_AAL116_timeseries.csv (or .npy, .txt, .mat)



            └── ...



    



    2. Nested structure (common in ABIDE):



        data_dir/



            ├── abide/  (or any subdirectory)



            │   ├── sub-001/



            │   │   ├── sub-001_AAL116_features_timeseries.mat



            │   │   ├── sub-001_AAL116_correlation_matrix.mat  (skipped)



            │   │   ├── sub-001_harvard48_*.mat  (skipped - not AAL116)



            │   │   └── ...



            │   ├── sub-002/



            │   │   └── ...



            │   └── ...



    



    File selection priority:



    1. AAL116 files with 'timeseries' or 'features_timeseries' in name



    2. Any AAL116 file (except correlation/connectivity matrices)



    3. Skips: correlation matrices, non-AAL116 parcellations



    



    labels_file: CSV with columns [subject_id, diagnosis, age, sex, site, ...]



        - diagnosis: 0 = Control, 1 = ASD



    """



    



    def _load_data(self, max_files: Optional[int] = None):



        """Implement ABIDE-specific data loading with nested directory support



        Automatically filters for AAL116 parcellation and prefers timeseries files



        """



        import glob



        import scipy.io



        



        # First, try to find files in nested structure (data/abide/abide/sub-*/)



        all_files = []



        



        # Check for nested structure: look for subdirectories with subject IDs



        subject_dirs = []



        for item in self.data_dir.iterdir():



            if item.is_dir():



                # Check if this directory contains subject subdirectories



                for sub_item in item.iterdir():



                    if sub_item.is_dir() and sub_item.name.startswith('sub-'):



                        subject_dirs.append(sub_item)



        



        # If we found nested structure, load from there



        if subject_dirs:



            print(f"Found nested ABIDE structure with {len(subject_dirs)} subject directories")



            print("Filtering for AAL116 parcellation files...")



            



            for subject_dir in sorted(subject_dirs):



                if max_files and len(all_files) >= max_files:



                    break



                



                # Look for AAL116 files in this subject's directory



                # Priority: features_timeseries > timeseries > other AAL116 files



                aal116_files = list(subject_dir.glob('*AAL116*'))



                



                if not aal116_files:



                    continue  # Skip subjects without AAL116 files



                



                # Find the best file (prefer timeseries, skip correlation matrices)



                preferred_file = None



                



                # First priority: features_timeseries or timeseries files



                for mat_file in aal116_files:



                    filename_lower = mat_file.name.lower()



                    # Skip correlation/connectivity matrices



                    if 'correlation' in filename_lower or 'connectivity' in filename_lower:



                        continue



                    # Prefer timeseries files



                    if any(keyword in filename_lower for keyword in 



                           ['features_timeseries', 'features_tim', 'timeseries', 'time_series']):



                        preferred_file = mat_file



                        break



                



                # Second priority: any AAL116 file that's not a correlation matrix



                if not preferred_file:



                    for mat_file in aal116_files:



                        filename_lower = mat_file.name.lower()



                        if 'correlation' not in filename_lower and 'connectivity' not in filename_lower:



                            preferred_file = mat_file



                            break



                



                if preferred_file:



                    all_files.append((preferred_file, subject_dir.name))  # Store with subject_dir name for label extraction



        else:



            # Flat structure: look for files directly in data_dir



            print(f"Using flat structure, searching in {self.data_dir}")



            print("Filtering for AAL116 parcellation files...")



            supported_formats = ['*.csv', '*.npy', '*.txt', '*.tsv', '*.mat']



            for fmt in supported_formats:



                # Search for AAL116 files specifically



                all_files.extend(glob.glob(str(self.data_dir / f'*AAL116*{fmt[1:]}')))



                # Also search recursively for nested files



                all_files.extend(glob.glob(str(self.data_dir / '**' / f'*AAL116*{fmt[1:]}'), recursive=True))



            



            # Filter out correlation matrices



            all_files = [f for f in all_files if 'correlation' not in Path(f).name.lower() 



                        and 'connectivity' not in Path(f).name.lower()]



            



            # Sort by priority: timeseries files first



            def file_priority(f):



                name_lower = Path(f).name.lower()



                if any(kw in name_lower for kw in ['features_timeseries', 'features_tim', 'timeseries', 'time_series']):



                    return 0



                return 1



            all_files.sort(key=file_priority)



        



        all_files = sorted(all_files)



        if max_files:



            all_files = all_files[:max_files]



        



        print(f"Found {len(all_files)} AAL116 files to load")



        if len(all_files) == 0:



            print("Warning: No AAL116 files found!")



            print("Make sure your ABIDE data contains files with 'AAL116' in the filename")



        



        for file_item in all_files:



            try:



                # Handle both tuple (file_path, subject_dir_name) and plain file_path



                if isinstance(file_item, tuple):



                    file_path, subject_dir_name = file_item



                    file_path = Path(file_path)



                else:



                    file_path = Path(file_item)



                    subject_dir_name = None



                



                # Load timeseries data based on format



                if file_path.suffix == '.npy':



                    data = np.load(file_path).astype(np.float32)



                elif file_path.suffix == '.csv':



                    df = pd.read_csv(file_path, header=None)



                    data = df.values.astype(np.float32)



                elif file_path.suffix in ['.txt', '.tsv']:



                    data = np.loadtxt(file_path, delimiter='\t').astype(np.float32)



                elif file_path.suffix == '.mat':



                    # Load MATLAB .mat file



                    mat_data = scipy.io.loadmat(str(file_path))



                    # Try common key names for timeseries data



                    data = None



                    for key in ['timeseries', 'data', 'ROI_timeseries', 'bold', 'function_matrix']:



                        if key in mat_data:



                            candidate = mat_data[key]



                            # Check if it's a 2D array (timeseries)



                            if isinstance(candidate, np.ndarray) and candidate.ndim == 2:



                                data = candidate.astype(np.float32)



                                break



                    



                    if data is None:



                        # If no standard key found, use the first non-metadata 2D array



                        for key, value in mat_data.items():



                            if not key.startswith('__') and isinstance(value, np.ndarray):



                                if value.ndim == 2:



                                    data = value.astype(np.float32)



                                    break



                    



                    if data is None:



                        print(f"Warning: Could not find 2D timeseries data in {file_path.name}")



                        continue



                else:



                    continue



                



                # Handle different data shapes



                # Expected: [T, N_ROI] or [N_ROI, T]



                if data.ndim != 2:



                    print(f"Warning: Expected 2D array, got {data.ndim}D in {file_path.name}")



                    continue



                



                if data.shape[0] < data.shape[1]:



                    data = data.T  # Transpose if ROIs are in first dimension



                



                # Create segments using sliding window



                segments = self._create_segments(data)



                subject_id = self._extract_subject_id(str(file_path))



                



                for segment in segments:



                    self.sequences.append(segment)



                    metadata_entry = {



                        'file': file_path.name,



                        'subject': subject_id,



                        'dataset': 'ABIDE'



                    }



                    # Store subject directory name for label extraction



                    if subject_dir_name:



                        metadata_entry['subject_dir'] = subject_dir_name



                    self.metadata.append(metadata_entry)



                    



            except Exception as e:



                print(f"Error loading {file_path}: {e}")



                import traceback



                traceback.print_exc()



                continue



        



        print(f"Loaded {len(self.sequences)} segments from {len(all_files)} files")



    



    def _create_segments(self, data: np.ndarray) -> List[np.ndarray]:



        """Create sliding window segments"""



        segments = []



        num_timepoints = data.shape[0]



        



        # If sequence is too short, zero-pad



        if num_timepoints < self.sequence_length:



            pad_length = self.sequence_length - num_timepoints



            data = np.pad(data, ((0, pad_length), (0, 0)), mode='edge')



            segments.append(data)



        else:



            # Sliding window



            for start in range(0, num_timepoints - self.sequence_length + 1, self.stride):



                end = start + self.sequence_length



                segments.append(data[start:end])



        



        return segments



    



    def _extract_subject_id(self, filepath: str) -> str:



        """



        Extract subject ID from ABIDE filepath



        Handles both:



        - Flat: sub-0050002_timeseries.csv



        - Nested: data/abide/abide/sub-001/function_matrix.mat



        """



        import re



        filepath = Path(filepath)



        



        # First, try to extract from directory name (for nested structure)



        # e.g., .../sub-001/function_matrix.mat



        for part in filepath.parts:



            match = re.search(r'sub-(\d+)', part)



            if match:



                return match.group(1)



        



        # Fallback: try filename



        match = re.search(r'sub-(\d+)', filepath.name)



        if match:



            return match.group(1)



        



        # Last resort: use filename stem



        return filepath.stem.split('_')[0]



    



    def _extract_label_from_subject_id(self, subject_id: str, metadata: Dict) -> Optional[Any]:



        """Extract diagnosis label from ABIDE subject folder name"""



        if self.label_column != 'diagnosis':



            return None



        



        # Check if we have subject_dir in metadata (from folder name)



        subject_dir = metadata.get('subject_dir', '')



        if not subject_dir:



            # Try to extract from subject_id if it contains the folder name pattern



            subject_dir = subject_id



        



        # Extract label from folder name: sub-control* = 0, sub-patient* = 1



        subject_dir_lower = subject_dir.lower()



        if 'control' in subject_dir_lower:



            return 0  # Control



        elif 'patient' in subject_dir_lower:



            return 1  # ASD/Patient



        



        return None











class PPMIDataset(BaseDownstreamDataset):



    """



    Dataset loader for PPMI (Parkinson's Progression Markers Initiative) dataset



    



    Expected structure:



        data_dir/



            ├── {subject_id}_timeseries.csv (or .npy, .txt)



            └── ...



        labels_file: CSV with columns [subject_id, diagnosis, age, sex, updrs_score, ...]



            - diagnosis: 0 = Control, 1 = PD (Parkinson's Disease)



            - updrs_score: Unified Parkinson's Disease Rating Scale



    """



    



    def _load_data(self, max_files: Optional[int] = None):



        """Implement PPMI-specific data loading"""



        import glob



        



        # Support multiple file formats including .mat



        supported_formats = ['*.csv', '*.npy', '*.txt', '*.tsv', '*.mat']



        all_files = []



        for fmt in supported_formats:



            all_files.extend(glob.glob(str(self.data_dir / fmt)))



        



        all_files = sorted(all_files)



        if max_files:



            all_files = all_files[:max_files]



        



        print(f"Found {len(all_files)} files in {self.data_dir}")



        



        for file_path in all_files:



            try:



                # Load timeseries data based on format



                if file_path.endswith('.npy'):



                    data = np.load(file_path).astype(np.float32)



                elif file_path.endswith('.csv'):



                    df = pd.read_csv(file_path, header=None)



                    data = df.values.astype(np.float32)



                elif file_path.endswith('.txt') or file_path.endswith('.tsv'):



                    data = np.loadtxt(file_path, delimiter='\t').astype(np.float32)



                elif file_path.endswith('.mat'):



                    # Load MATLAB .mat file



                    import scipy.io



                    mat_data = scipy.io.loadmat(file_path)



                    # Try common key names for timeseries data



                    for key in ['timeseries', 'data', 'ROI_timeseries', 'bold']:



                        if key in mat_data:



                            data = mat_data[key].astype(np.float32)



                            break



                    else:



                        # If no standard key found, use the first non-metadata key



                        for key, value in mat_data.items():



                            if not key.startswith('__') and isinstance(value, np.ndarray):



                                data = value.astype(np.float32)



                                break



                        else:



                            print(f"Warning: Could not find timeseries data in {file_path}")



                            continue



                else:



                    continue



                



                # Handle different data shapes



                if data.shape[0] < data.shape[1]:



                    data = data.T  # Transpose if ROIs are in first dimension



                



                # Create segments using sliding window



                segments = self._create_segments(data)



                subject_id = self._extract_subject_id(file_path)



                



                for segment in segments:



                    self.sequences.append(segment)



                    self.metadata.append({



                        'file': Path(file_path).name,



                        'subject': subject_id,



                        'dataset': 'PPMI'



                    })



                    



            except Exception as e:



                print(f"Error loading {file_path}: {e}")



                continue



        



        print(f"Loaded {len(self.sequences)} segments from {len(all_files)} files")



    



    def _create_segments(self, data: np.ndarray) -> List[np.ndarray]:



        """Create sliding window segments"""



        segments = []



        num_timepoints = data.shape[0]



        



        # If sequence is too short, zero-pad



        if num_timepoints < self.sequence_length:



            pad_length = self.sequence_length - num_timepoints



            data = np.pad(data, ((0, pad_length), (0, 0)), mode='edge')



            segments.append(data)



        else:



            # Sliding window



            for start in range(0, num_timepoints - self.sequence_length + 1, self.stride):



                end = start + self.sequence_length



                segments.append(data[start:end])



        



        return segments



    



    def _extract_subject_id(self, filename: str) -> str:



        """Extract subject ID from PPMI filename"""



        # Common PPMI naming: 3XXX_timeseries.csv or PPMI_3XXX.npy



        import re



        match = re.search(r'(\d{4})', Path(filename).name)



        if match:



            return match.group(1)



        # Fallback



        return Path(filename).stem.split('_')[0]











# Dataset registry - Focus on ABIDE and PPMI for downstream evaluation



DATASET_REGISTRY = {



    'ABIDE': ABIDEDataset,



    'PPMI': PPMIDataset,



}











# ============================================================================



# Model Loading



# ============================================================================







def load_model(checkpoint_path: str, device: str = 'cuda', model_type: str = 'v2') -> Tuple[torch.nn.Module, Dict]:



    """Load pre-trained foundation model with architecture mismatch handling



    



    Args:



        checkpoint_path: Path to model checkpoint



        device: Device to load model on



        model_type: 'v2' (current) or 'old' (previous architecture)



    """



    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)



    config = checkpoint['config']



    



    if model_type == 'old':



        if KoopmanBrainModelOld is None:



            raise ImportError("OLD model not available. Please ensure OLD/ directory exists.")



        model = KoopmanBrainModelOld(



            n_roi=config['n_roi'],



            d_latent=config['latent_dim'],



            d_control=config['control_dim'],



            d_episodic=config.get('episodic_dim', 64),



            d_context=config.get('context_dim', 64),



            n_classes=config.get('n_classes', 4),



            graph_type=config.get('graph_type', 'cheb'),



            n_graph_layers=config.get('n_graph_layers', 3),



            cheb_k=config.get('cheb_k', 3),



            drop_edge_p=config.get('drop_edge_p', 0.1),



            n_temporal_layers=config.get('n_temporal_layers', 3),



            n_heads=config.get('n_heads', 8),



            koopman_rank=config.get('koopman_rank', 16),



            spectral_radius=config.get('target_spectral_radius', 0.95),



            use_sde=config.get('use_sde', True),



            diffusion_mode=config.get('diffusion_mode', 'learned'),



            uncertainty_mode=config.get('uncertainty_mode', 'heteroscedastic'),



            dropout=0.0  # No dropout during evaluation



        ).to(device)



    else:



        model = BrainMech(



            n_roi=config['n_roi'],



            d_latent=config['latent_dim'],



            d_control=config['control_dim'],



            d_episodic=config.get('episodic_dim', 64),



            d_context=config.get('context_dim', 64),



            d_task=config.get('task_dim', 32),



            n_classes=config.get('n_classes', 4),



            task_type='classification',



            graph_type=config.get('graph_type', 'cheb'),



            n_graph_layers=config.get('n_graph_layers', 3),



            cheb_k=config.get('cheb_k', 3),



            drop_edge_p=config.get('drop_edge_p', 0.1),



            n_temporal_layers=config.get('n_temporal_layers', 3),



            n_heads=config.get('n_heads', 8),



            koopman_rank=config.get('koopman_rank', 16),



            spectral_radius=config.get('target_spectral_radius', 0.95),



            use_sde=config.get('use_sde', True),



            diffusion_mode=config.get('diffusion_mode', 'learned'),



            uncertainty_mode=config.get('uncertainty_mode', 'heteroscedastic'),



            fc_matrix=None,  # Don't use FC matrix for downstream (different dataset)



            dropout=0.0  # No dropout during evaluation



        ).to(device)



    



    # Handle architecture mismatches



    state_dict = checkpoint['model_state_dict']



    model_state_dict = model.state_dict()



    



    # Create a mapping for common architecture changes

    key_mapping = {}

    for old_key in state_dict.keys():

        new_key = old_key



        # Handle possible renaming between `input_proj` and `roi_embedding`

        if 'spatial_encoder.input_proj' in old_key or 'spatial_encoder.roi_embedding' in old_key:

            has_input_proj = any(k.startswith('spatial_encoder.input_proj') for k in model_state_dict.keys())

            has_roi_embedding = any(k.startswith('spatial_encoder.roi_embedding') for k in model_state_dict.keys())



            # If both checkpoint and model already use the same key, keep it

            if 'spatial_encoder.input_proj' in old_key and has_input_proj:

                new_key = old_key

            elif 'spatial_encoder.roi_embedding' in old_key and has_roi_embedding:

                new_key = old_key

            # If checkpoint uses input_proj but model expects roi_embedding

            elif 'spatial_encoder.input_proj' in old_key and has_roi_embedding:

                new_key = old_key.replace('spatial_encoder.input_proj', 'spatial_encoder.roi_embedding')

            # If checkpoint uses roi_embedding but model expects input_proj

            elif 'spatial_encoder.roi_embedding' in old_key and has_input_proj:

                new_key = old_key.replace('spatial_encoder.roi_embedding', 'spatial_encoder.input_proj')



        key_mapping[old_key] = new_key



    



    # Apply key mapping and filter by shape compatibility



    mapped_state_dict = {}



    skipped_keys = []



    shape_mismatches = []



    



    for old_key, value in state_dict.items():



        new_key = key_mapping.get(old_key, old_key)



        



        if new_key not in model_state_dict:



            # Key doesn't exist in current model



            skipped_keys.append(old_key)



            continue



        



        # Check shape compatibility



        checkpoint_shape = value.shape



        model_shape = model_state_dict[new_key].shape



        



        if checkpoint_shape == model_shape:



            # Shapes match - use it



            mapped_state_dict[new_key] = value



        else:



            # Shape mismatch - skip and let it initialize randomly



            shape_mismatches.append((old_key, checkpoint_shape, model_shape))



    



    # Report skipped keys



    if skipped_keys:



        print(f"Warning: Skipped {len(skipped_keys)} keys not in current model")



        if len(skipped_keys) <= 5:



            for key in skipped_keys:



                print(f"  - {key}")



    



    # Report shape mismatches



    if shape_mismatches:



        print(f"Warning: {len(shape_mismatches)} keys have shape mismatches (will use random initialization):")



        for old_key, ckpt_shape, model_shape in shape_mismatches[:5]:



            print(f"  - {old_key}: checkpoint {ckpt_shape} vs model {model_shape}")



        if len(shape_mismatches) > 5:



            print(f"  ... and {len(shape_mismatches) - 5} more")



    



    # Load with strict=False to allow missing/unexpected keys



    # Filter out any remaining problematic keys



    final_state_dict = {}



    for key, value in mapped_state_dict.items():



        if key in model_state_dict:



            if value.shape == model_state_dict[key].shape:



                final_state_dict[key] = value



    



    try:



        missing_keys, unexpected_keys = model.load_state_dict(final_state_dict, strict=False)



        loaded_count = len(final_state_dict)



        total_count = len(model_state_dict)



        print(f"✓ Model loaded: {loaded_count}/{total_count} layers matched")



        



        if shape_mismatches:



            print(f"  Note: {len(shape_mismatches)} layers skipped due to shape mismatches")



            print("  These layers will use random initialization (architecture changed)")



        if skipped_keys:



            print(f"  Note: {len(skipped_keys)} layers skipped (not in current model)")



        if missing_keys:



            print(f"  Note: {len(missing_keys)} layers missing (will use random initialization)")



            



    except RuntimeError as e:



        # If there's still an error, try loading layer by layer



        print(f"Warning: Error during load_state_dict: {e}")



        print("Attempting layer-by-layer load...")



        



        # Load compatible layers one by one



        loaded = 0



        for key, value in final_state_dict.items():



            try:



                model.load_state_dict({key: value}, strict=False)



                loaded += 1



            except:



                pass



        



        print(f"✓ Partially loaded: {loaded}/{len(final_state_dict)} compatible layers")



        print("  Remaining layers will use random initialization")



    



    model.eval()



    



    return model, config











# ============================================================================



# Feature Extraction



# ============================================================================







def extract_features(



    model: torch.nn.Module,



    dataset: Dataset,



    device: torch.device,



    feature_type: str = 'z_concat',



    batch_size: int = 32



) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:



    """



    Extract features from model for all samples in dataset.



    



    Args:



        model: Pre-trained model



        dataset: Downstream dataset



        device: Device to run model on



        feature_type: Type of features ('z_mean', 'z_std', 'z_concat')



        batch_size: Batch size for feature extraction



    



    Returns:



        features: [N, d_feat] array of features



        labels: [N] array of labels (if available)



        subjects: [N] array of subject IDs



    """



    features = []



    labels = []



    subjects = []



    



    print(f"Extracting {feature_type} features from {len(dataset)} samples...")



    



    model.eval()



    with torch.no_grad():



        # Process in batches for efficiency



        for i in tqdm(range(0, len(dataset), batch_size)):



            batch_end = min(i + batch_size, len(dataset))



            batch_data = []



            batch_metadata = []



            



            for j in range(i, batch_end):



                x, metadata = dataset[j]



                batch_data.append(x)



                batch_metadata.append(metadata)



            



            # Handle variable ROI counts by padding to model's expected ROI count



            if batch_data:



                # Get expected ROI from model (default 116 for AAL_116)



                expected_roi = 116  # Default for AAL_116



                if hasattr(model, 'n_roi'):



                    expected_roi = model.n_roi



                elif hasattr(model, 'config') and isinstance(model.config, dict) and 'n_roi' in model.config:



                    expected_roi = model.config['n_roi']



                elif hasattr(model, 'spatial_encoder') and hasattr(model.spatial_encoder, 'n_roi'):



                    expected_roi = model.spatial_encoder.n_roi



                



                max_seq_len = max(x.shape[0] for x in batch_data)



                



                # Pad all sequences to expected ROI count and max sequence length



                padded_batch = []



                for x in batch_data:



                    T, n_roi = x.shape



                    # Pad or crop ROI dimension to expected ROI count



                    if n_roi < expected_roi:



                        padding_roi = torch.zeros(T, expected_roi - n_roi, dtype=x.dtype)



                        x = torch.cat([x, padding_roi], dim=1)



                    elif n_roi > expected_roi:



                        # Crop if larger than expected



                        x = x[:, :expected_roi]



                    # Pad time dimension if needed



                    if T < max_seq_len:



                        padding_time = torch.zeros(max_seq_len - T, x.shape[1], dtype=x.dtype)



                        x = torch.cat([x, padding_time], dim=0)



                    padded_batch.append(x)



                



                # Stack batch



                x_batch = torch.stack(padded_batch).to(device)  # [B, T, N_ROI]



            else:



                continue



            



            # Forward pass



            outputs = model(x_batch, task_vectors=None)



            z = outputs['z_latent']  # [B, T, d_latent]



            



            # Extract features for each sample in batch

            # Ensure batch size matches metadata length

            if len(batch_metadata) != z.shape[0]:

                print(f"  ⚠️  Warning: Batch size mismatch: z.shape[0]={z.shape[0]}, len(batch_metadata)={len(batch_metadata)}")

                print(f"     Skipping this batch to avoid shape mismatches")

                continue



            for b in range(z.shape[0]):

                try:

                    if feature_type == 'z_mean':

                        feat = z[b].mean(dim=0).cpu().numpy()

                    elif feature_type == 'z_std':

                        feat = z[b].std(dim=0).cpu().numpy()

                    elif feature_type == 'z_concat':

                        z_mean = z[b].mean(dim=0).cpu().numpy()

                        z_std = z[b].std(dim=0).cpu().numpy()

                        feat = np.concatenate([z_mean, z_std])

                    else:

                        feat = z[b].mean(dim=0).cpu().numpy()

                    

                    features.append(feat)

                    

                    # Ensure subject exists in metadata

                    if b >= len(batch_metadata) or 'subject' not in batch_metadata[b]:

                        print(f"  ⚠️  Warning: Missing subject for sample {b} in batch, skipping")

                        features.pop()  # Remove the feature we just added

                        continue

                    

                    subjects.append(batch_metadata[b]['subject'])

                    

                    # Get label if available

                    if 'label' in batch_metadata[b]:

                        labels.append(batch_metadata[b]['label'])

                    else:

                        labels.append(None)

                except (IndexError, KeyError) as e:

                    print(f"  ⚠️  Warning: Error processing sample {b} in batch: {e}, skipping")

                    # Remove any partially added feature

                    if len(features) > len(subjects):

                        features.pop()

                    continue



    

    # Validate that all arrays have the same length before converting

    if len(features) != len(subjects) or len(features) != len(labels):

        raise ValueError(

            f"Array length mismatch after feature extraction: "

            f"features={len(features)}, subjects={len(subjects)}, labels={len(labels)}. "

            f"This indicates a bug in feature extraction - some samples were skipped or processed incorrectly."

        )

    

    features = np.array(features)

    subjects = np.array(subjects)

    

    # Filter out samples without labels

    if any(l is None for l in labels):

        print("Warning: Some samples don't have labels. Filtering them out...")

        valid_mask = np.array([l is not None for l in labels])

        

        # Validate mask length matches arrays

        if len(valid_mask) != len(features):

            raise ValueError(

                f"Mask length mismatch: features={len(features)}, valid_mask={len(valid_mask)}. "

                f"This indicates a bug in label filtering."

            )

        

        features = features[valid_mask]

        subjects = subjects[valid_mask]

        labels = [l for l, v in zip(labels, valid_mask) if v]

    

    labels = np.array(labels)

    

    # Final validation before returning

    if len(features) != len(subjects) or len(features) != len(labels):

        raise ValueError(

            f"Array length mismatch after filtering: "

            f"features={len(features)}, subjects={len(subjects)}, labels={len(labels)}. "

            f"This indicates a bug in data processing."

        )

    

    return features, labels, subjects











# ============================================================================



# Data Splitting



# ============================================================================







def split_by_subject(



    features: np.ndarray,



    labels: np.ndarray,



    subjects: np.ndarray,



    test_size: float = 0.2,



    random_state: int = 42,



    stratify: bool = True



) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:



    """



    Split data by subject to avoid data leakage.



    Ensures no subject appears in both train and test sets.



    Uses stratified splitting to maintain class distribution across train/test.



    Args:



        features: Feature array



        labels: Label array



        subjects: Subject ID array



        test_size: Proportion of subjects to use for testing



        random_state: Random seed



        stratify: If True, use stratified splitting to maintain class distribution



    """



    from sklearn.model_selection import GroupShuffleSplit



    from scipy import stats



    



    unique_subjects = np.unique(subjects)



    



    # Get label for each unique subject (use mode if subject has multiple samples)



    subject_labels = []



    for subj in unique_subjects:



        subj_mask = subjects == subj



        subj_labels = labels[subj_mask]



        # Use mode (most common label) for this subject



        mode_result = stats.mode(subj_labels, keepdims=True)



        subject_labels.append(mode_result.mode[0])



    subject_labels = np.array(subject_labels)



    



    # Use GroupShuffleSplit for proper subject-level splitting with stratification

    # This ensures no subject appears in both train and test sets

    from sklearn.model_selection import train_test_split

    

    # Use stratified split if requested and we have multiple classes

    if stratify and len(np.unique(subject_labels)) > 1:

        try:

            train_subjects_arr, test_subjects_arr = train_test_split(

                unique_subjects,

                test_size=test_size,

                random_state=random_state,

                stratify=subject_labels

            )

        except ValueError as e:

            # If stratification fails (e.g., too few samples per class), fall back to random

            print(f"  ⚠️  Warning: Stratified split failed ({e}), using random split")

            train_subjects_arr, test_subjects_arr = train_test_split(

                unique_subjects,

                test_size=test_size,

                random_state=random_state

            )

    else:

        # Random split without stratification

        train_subjects_arr, test_subjects_arr = train_test_split(

            unique_subjects,

            test_size=test_size,

            random_state=random_state

        )



    



    test_subjects = set(test_subjects_arr)



    train_subjects = set(train_subjects_arr)



    

    # Validate that arrays have matching lengths

    if len(features) != len(subjects) or len(features) != len(labels):

        raise ValueError(

            f"Array length mismatch: features={len(features)}, subjects={len(subjects)}, labels={len(labels)}. "

            f"This indicates a bug in feature extraction or data collection."

        )

    

    train_mask = np.array([s in train_subjects for s in subjects])

    test_mask = np.array([s in test_subjects for s in subjects])

    

    # Verify no subject leakage (critical for longitudinal data)

    train_subjects_actual = set(subjects[train_mask])

    test_subjects_actual = set(subjects[test_mask])

    overlap = train_subjects_actual & test_subjects_actual

    if overlap:

        raise ValueError(

            f"CRITICAL: {len(overlap)} subjects appear in both train and test sets! "

            f"This causes data leakage. Overlapping subjects: {list(overlap)[:5]}..."

        )

    

    # Validate mask lengths match feature array

    if len(train_mask) != len(features) or len(test_mask) != len(features):

        raise ValueError(

            f"Mask length mismatch: features={len(features)}, train_mask={len(train_mask)}, test_mask={len(test_mask)}. "

            f"This indicates a bug in mask creation."

        )



    X_train = features[train_mask]

    X_test = features[test_mask]

    y_train = labels[train_mask]

    y_test = labels[test_mask]



    



    print(f"\n✓ Subject-level split (stratified={stratify}):")



    print(f"  Train subjects: {len(train_subjects)}")



    print(f"  Test subjects:  {len(test_subjects)}")



    print(f"  Train samples:  {len(X_train)}")



    print(f"  Test samples:   {len(X_test)}")



    



    # Print class distribution in train and test



    if len(np.unique(labels)) > 1:



        print(f"  Train class distribution:")



        for cls in np.unique(y_train):



            count = np.sum(y_train == cls)



            print(f"    Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")



        print(f"  Test class distribution:")



        for cls in np.unique(y_test):



            count = np.sum(y_test == cls)



            print(f"    Class {cls}: {count} ({count/len(y_test)*100:.1f}%)")



    



    return X_train, X_test, y_train, y_test





def aggregate_predictions_per_subject(

    predictions: np.ndarray,

    subjects: np.ndarray,

    task_type: str = 'classification',

    probabilities: Optional[np.ndarray] = None

) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

    """

    Aggregate predictions per subject.

    

    For classification: uses mode (most common prediction) per subject.

    For regression: uses mean per subject.

    

    Args:

        predictions: Array of predictions (one per sample)

        subjects: Array of subject IDs (one per sample)

        task_type: 'classification' or 'regression'

        probabilities: Optional probability array (for classification)

    

    Returns:

        aggregated_predictions: One prediction per unique subject

        unique_subjects: Array of unique subject IDs

        aggregated_probs: Aggregated probabilities (if provided, None otherwise)

    """

    from scipy import stats

    

    unique_subjects = np.unique(subjects)

    aggregated_predictions = []

    aggregated_probs = [] if probabilities is not None else None

    

    for subj in unique_subjects:

        subj_mask = subjects == subj

        subj_preds = predictions[subj_mask]

        

        if task_type == 'classification':

            # Use mode (most common prediction) for classification

            mode_result = stats.mode(subj_preds, keepdims=True)

            aggregated_predictions.append(mode_result.mode[0])

            

            # For probabilities: average them per subject

            if probabilities is not None:

                subj_probs = probabilities[subj_mask]

                if len(subj_probs.shape) == 1:

                    # Binary classification: average probabilities

                    avg_prob = np.mean(subj_probs)

                    aggregated_probs.append(avg_prob)

                else:

                    # Multi-class: average probability vectors

                    avg_prob = np.mean(subj_probs, axis=0)

                    aggregated_probs.append(avg_prob)

        else:

            # Regression: use mean

            aggregated_predictions.append(np.mean(subj_preds))

            if probabilities is not None:

                # For regression, probabilities might be uncertainties

                subj_probs = probabilities[subj_mask]

                if len(subj_probs.shape) == 1:

                    aggregated_probs.append(np.mean(subj_probs))

                else:

                    aggregated_probs.append(np.mean(subj_probs, axis=0))

    

    aggregated_predictions = np.array(aggregated_predictions)

    if aggregated_probs is not None:

        aggregated_probs = np.array(aggregated_probs)

    

    return aggregated_predictions, unique_subjects, aggregated_probs











# ============================================================================



# Evaluation Functions



# ============================================================================







def evaluate_classification(



    X_train: np.ndarray,



    y_train: np.ndarray,



    X_test: np.ndarray,



    y_test: np.ndarray,



    classifier_name: str = 'LogisticRegression'



) -> Tuple[Dict, object]:



    """Evaluate classification task"""



    print(f"\nTraining {classifier_name}...")



    



    if classifier_name == 'SVM':

        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, class_weight='balanced')

    elif classifier_name == 'RandomForest':

        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')

    elif classifier_name == 'LogisticRegression':

        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    elif classifier_name == 'GradientBoosting':

        clf = GradientBoostingClassifier(

            n_estimators=200, max_depth=5, learning_rate=0.1,

            random_state=42

        )

    else:

        raise ValueError(f"Unknown classifier: {classifier_name}")



    



    clf.fit(X_train, y_train)



    



    y_pred_train = clf.predict(X_train)



    y_pred_test = clf.predict(X_test)



    



    train_acc = accuracy_score(y_train, y_pred_train)



    test_acc = accuracy_score(y_test, y_pred_test)



    test_f1_macro = f1_score(y_test, y_pred_test, average='macro')



    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')



    



    results = {



        'classifier': classifier_name,



        'train_accuracy': float(train_acc),



        'test_accuracy': float(test_acc),



        'test_f1_macro': float(test_f1_macro),



        'test_f1_weighted': float(test_f1_weighted),



        'classification_report': classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),



        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()



    }



    



    return results, clf











def evaluate_regression(



    X_train: np.ndarray,



    y_train: np.ndarray,



    X_test: np.ndarray,



    y_test: np.ndarray,



    regressor_name: str = 'Ridge'



) -> Tuple[Dict, object]:



    """Evaluate regression task"""



    print(f"\nTraining {regressor_name}...")



    



    if regressor_name == 'Ridge':

        reg = Ridge(alpha=1.0, random_state=42)

    elif regressor_name == 'MLP':

        reg = MLPRegressor(

            hidden_layer_sizes=(128, 64), activation='relu',

            max_iter=1000, random_state=42, early_stopping=True,

            learning_rate_init=0.001, alpha=0.0001

        )

    elif regressor_name == 'RandomForest':

        reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

    elif regressor_name == 'GradientBoosting':

        reg = GradientBoostingRegressor(

            n_estimators=200, max_depth=5, learning_rate=0.1,

            random_state=42

        )

    else:

        raise ValueError(f"Unknown regressor: {regressor_name}")



    



    reg.fit(X_train, y_train)



    



    y_pred_train = reg.predict(X_train)



    y_pred_test = reg.predict(X_test)



    



    train_mae = mean_absolute_error(y_train, y_pred_train)



    test_mae = mean_absolute_error(y_test, y_pred_test)



    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))



    test_r2 = r2_score(y_test, y_pred_test)



    



    results = {



        'regressor': regressor_name,



        'train_mae': float(train_mae),



        'test_mae': float(test_mae),



        'test_rmse': float(test_rmse),



        'test_r2': float(test_r2)



    }



    



    return results, reg











# ============================================================================



# Main Evaluation Script



# ============================================================================







def main():



    parser = argparse.ArgumentParser(



        description='Downstream Task Evaluation for Foundation Model',



        formatter_class=argparse.RawDescriptionHelpFormatter,



        epilog="""



Examples:



  # Classification on ABIDE



  python scripts/evaluate_downstream.py \\



    --checkpoint results/model.pt \\



    --dataset ABIDE \\



    --data_dir data/ABIDE/BOLD \\



    --task_type classification \\



    --labels_file data/ABIDE/labels.csv \\



    --label_column diagnosis







  # Regression on PPMI



  python scripts/evaluate_downstream.py \\



    --checkpoint results/model.pt \\



    --dataset PPMI \\



    --data_dir data/PPMI/BOLD \\



    --task_type regression \\



    --labels_file data/PPMI/labels.csv \\



    --label_column age



        """



    )



    



    # Model



    parser.add_argument('--checkpoint', type=str, required=True,



                        help='Path to pre-trained model checkpoint')



    



    # Dataset



    parser.add_argument('--dataset', type=str, required=True,



                        choices=list(DATASET_REGISTRY.keys()),



                        help='Dataset name (ADNI, ABIDE, PPMI, etc.)')



    parser.add_argument('--data_dir', type=str, required=True,



                        help='Path to downstream dataset BOLD data directory')



    parser.add_argument('--labels_file', type=str, default=None,



                        help='Path to CSV file with labels (must have subject_id column)')



    parser.add_argument('--label_column', type=str, default=None,



                        help='Column name in labels_file containing target labels')



    



    # Data processing



    parser.add_argument('--sequence_length', type=int, default=None,



                        help='Sequence length (default: from checkpoint config)')



    parser.add_argument('--stride', type=int, default=None,



                        help='Stride for sliding window (default: from checkpoint config)')



    parser.add_argument('--max_files', type=int, default=None,



                        help='Maximum number of files to load (for testing)')



    



    # Feature extraction



    parser.add_argument('--feature_type', type=str, default='z_concat',



                        choices=['z_mean', 'z_std', 'z_concat'],



                        help='Type of features to extract')



    parser.add_argument('--batch_size', type=int, default=32,



                        help='Batch size for feature extraction')



    



    # Task



    parser.add_argument('--task_type', type=str, required=True,



                        choices=['classification', 'regression'],



                        help='Task type: classification or regression')



    



    # Evaluation



    parser.add_argument('--classifiers', type=str, nargs='+',



                        default=['LogisticRegression', 'SVM', 'RandomForest', 'GradientBoosting'],



                        help='Classifiers to evaluate (for classification): LogisticRegression (linear), SVM (non-linear), RandomForest (ensemble), GradientBoosting (ensemble)')



    parser.add_argument('--regressors', type=str, nargs='+',



                        default=['Ridge', 'MLP', 'RandomForest', 'GradientBoosting'],



                        help='Regressors to evaluate (for regression): Ridge (linear), MLP (neural network), RandomForest (ensemble), GradientBoosting (ensemble)')



    parser.add_argument('--test_size', type=float, default=0.2,



                        help='Test set size (subject-level split)')



    



    # Output



    parser.add_argument('--output', type=str, default='outputs/downstream_eval.json',



                        help='Output JSON file path')



    



    args = parser.parse_args()



    



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    print(f"Using device: {device}")



    



    # Load model



    print("\n" + "="*60)



    print("DOWNSTREAM TASK EVALUATION")



    print("="*60)



    print(f"Dataset: {args.dataset}")



    print(f"Task: {args.task_type}")



    print("="*60)



    



    print("\nLoading pre-trained model...")



    model, config = load_model(args.checkpoint, device)



    print(f"✓ Model loaded")



    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")



    



    # Get sequence parameters



    seq_length = args.sequence_length if args.sequence_length else config['sequence_length']



    stride = args.stride if args.stride else config['stride']



    



    # Load downstream dataset



    print(f"\nLoading {args.dataset} dataset...")



    dataset_class = DATASET_REGISTRY[args.dataset]



    dataset = dataset_class(



        data_dir=args.data_dir,



        labels_file=args.labels_file,



        label_column=args.label_column,



        sequence_length=seq_length,



        stride=stride,



        normalize=True,



        max_files=args.max_files



    )



    print(f"✓ Loaded {len(dataset)} samples")



    



    # Extract features



    features, labels, subjects = extract_features(



        model, dataset, device, args.feature_type, args.batch_size



    )



    



    print(f"\n✓ Extracted features:")



    print(f"  Shape: {features.shape}")



    print(f"  Unique subjects: {len(np.unique(subjects))}")



    if args.task_type == 'classification':



        print(f"  Classes: {np.unique(labels)}")



        print(f"  Class distribution:")



        for label in np.unique(labels):



            count = np.sum(labels == label)



            print(f"    {label}: {count} ({count/len(labels)*100:.1f}%)")



    else:



        print(f"  Label range: [{labels.min():.2f}, {labels.max():.2f}]")



        print(f"  Label mean: {labels.mean():.2f} ± {labels.std():.2f}")



    



    # Split by subject



    X_train, X_test, y_train, y_test = split_by_subject(



        features, labels, subjects, test_size=args.test_size, random_state=42



    )



    



    # Evaluate



    all_results = {



        'dataset': args.dataset,



        'task_type': args.task_type,



        'feature_type': args.feature_type,



        'feature_dim': int(features.shape[1]),



        'n_samples': int(len(features)),



        'n_subjects': int(len(np.unique(subjects))),



        'train_size': int(len(X_train)),



        'test_size': int(len(X_test)),



    }



    



    print("\n" + "="*60)



    print("EVALUATION")



    print("="*60)



    



    if args.task_type == 'classification':



        all_results['classifiers'] = {}



        for clf_name in args.classifiers:



            results, clf = evaluate_classification(



                X_train, y_train, X_test, y_test, clf_name



            )



            all_results['classifiers'][clf_name] = results



            



            print(f"\n{clf_name}:")



            print(f"  Train Accuracy: {results['train_accuracy']:.4f}")



            print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")



            print(f"  Test F1 (macro): {results['test_f1_macro']:.4f}")



        



        # Best classifier



        best_clf = max(



            all_results['classifiers'].items(),



            key=lambda x: x[1]['test_accuracy']



        )



        print(f"\n{'='*60}")



        print(f"BEST CLASSIFIER: {best_clf[0]}")



        print(f"{'='*60}")



        print(f"Test Accuracy: {best_clf[1]['test_accuracy']:.4f}")



        print(f"Test F1 (macro): {best_clf[1]['test_f1_macro']:.4f}")



    



    else:  # regression



        all_results['regressors'] = {}



        for reg_name in args.regressors:



            results, reg = evaluate_regression(



                X_train, y_train, X_test, y_test, reg_name



            )



            all_results['regressors'][reg_name] = results



            



            print(f"\n{reg_name}:")



            print(f"  Train MAE: {results['train_mae']:.4f}")



            print(f"  Test MAE:  {results['test_mae']:.4f}")



            print(f"  Test RMSE: {results['test_rmse']:.4f}")



            print(f"  Test R²:   {results['test_r2']:.4f}")



        



        # Best regressor



        best_reg = min(



            all_results['regressors'].items(),



            key=lambda x: x[1]['test_mae']



        )



        print(f"\n{'='*60}")



        print(f"BEST REGRESSOR: {best_reg[0]}")



        print(f"{'='*60}")



        print(f"Test MAE:  {best_reg[1]['test_mae']:.4f}")



        print(f"Test RMSE: {best_reg[1]['test_rmse']:.4f}")



        print(f"Test R²:   {best_reg[1]['test_r2']:.4f}")



    



    # Save results



    output_path = Path(args.output)



    output_path.parent.mkdir(parents=True, exist_ok=True)



    with open(output_path, 'w') as f:



        json.dump(all_results, f, indent=2)



    



    print(f"\n✓ Results saved to {args.output}")



    print("\n" + "="*60)



    print("EVALUATION COMPLETE")



    print("="*60)











if __name__ == '__main__':



    main()








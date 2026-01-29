"""
Disease Diagnosis Evaluation Script

Evaluates the pre-trained Koopman Foundation Model on disease diagnosis tasks:
- ABIDE: Autism Spectrum Disorder (ASD) vs Control (binary)
- PPMI: 4-class classification (Control, SWEDD, Prodromal, PD)
- TAOWU: Control vs Patient (binary)
- NEUROCON: Control vs Patient (binary)

This is a specialized script for classification tasks focusing on disease diagnosis.

Usage:
    # ABIDE - Autism diagnosis
    python scripts/evaluate_disease_diagnosis.py \
        --checkpoint results/koopman_fm_20251025_193511/best_model.pt \
        --dataset ABIDE \
        --labels_file data/ABIDE/labels.csv \
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
from tqdm import tqdm

from eval.evaluate_downstream import (
    load_model, DATASET_REGISTRY, extract_features, split_by_subject, aggregate_predictions_per_subject
)
# Feature caching removed - always extract fresh features
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
try:
    from eval.frozen_mlp_head import train_classification_head
except ImportError:
    try:
        from eval.frozen_mlp_head import train_classification_head
    except ImportError:
        try:
            from frozen_mlp_head import train_classification_head
        except ImportError:
            # Fallback: define dummy function if frozen_mlp_head is not available
            def train_classification_head(*args, **kwargs):
                raise ImportError("frozen_mlp_head module not found. Please ensure eval/frozen_mlp_head.py exists.")
try:
    from scripts.manage_checkpoints import find_best_checkpoint
except ImportError:
    find_best_checkpoint = None

# Create PPMI dataset class that handles both nested and flat structures
class PPMIDatasetNested:
    """
    Enhanced PPMI dataset that handles nested structure with 4-class classification:
    - Control (label=0)
    - SWEDD (label=1) - Scans Without Evidence of Dopaminergic Deficit
    - Prodromal (label=2) - Prodromal Parkinson's cases
    - PD (label=3) - Parkinson's Disease
    
    Structure:
    data/ppmi/sub-control/patient1234/ (contains correlation matrix and feature_timeseries)
    data/ppmi/sub-swedd/patient3456/ (contains correlation matrix and feature_timeseries)
    data/ppmi/sub-prodromal/patient9012/ (contains correlation matrix and feature_timeseries)
    data/ppmi/sub-patient/patient5678/ or data/ppmi/sub-pd/patient5678/ (contains correlation matrix and feature_timeseries)
    """
    def __init__(
        self,
        data_dir: str,
        labels_file: str = None,
        label_column: str = 'diagnosis',
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        max_files: int = None,
        single_window_per_subject: bool = False
    ):
        """Override __init__ to handle nested sub-control/patient1234/ structure with 4-class labels"""
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.label_column = label_column
        self.labels_file = labels_file
        self.single_window_per_subject = single_window_per_subject
        
        # Helper function to determine label class from category folder name
        # 4-class classification: Control=0, SWEDD=1, Prodromal=2, PD=3
        def get_label_class(cat_name_lower: str) -> int:
            """Map category folder name to 4-class label"""
            if 'control' in cat_name_lower or 'sub-control' in cat_name_lower:
                return 0  # Control
            elif 'swedd' in cat_name_lower or 'sub-swedd' in cat_name_lower:
                return 1  # SWEDD
            elif 'prodromal' in cat_name_lower or 'prodrom' in cat_name_lower or 'sub-prodromal' in cat_name_lower:
                return 2  # Prodromal
            elif ('patient' in cat_name_lower or 'pd' in cat_name_lower or 'parkinson' in cat_name_lower or 
                  'sub-patient' in cat_name_lower or 'sub-pd' in cat_name_lower):
                return 3  # PD (Parkinson's Disease)
            else:
                # Default to Control if unclear
                return 0
        
        # Helper function to get class name from label
        def get_class_name(label: int) -> str:
            class_names = {0: "Control", 1: "SWEDD", 2: "Prodromal", 3: "PD"}
            return class_names.get(label, "Unknown")
        
        # First, look for category folders (sub-control, sub-patient, sub-prodromal, sub-swedd)
        # Handle both nested structure (sub-control/patient1234/) and flat structure (sub-patient3122/)
        category_folders = []
        flat_subject_folders = []
        
        for item in self.data_dir.iterdir():
            if item.is_dir():
                name_lower = item.name.lower()
                # Check if it's a flat structure: sub-patient3122 or sub-control4032 (contains MAT files directly)
                if any(keyword in name_lower for keyword in ['sub-control', 'sub-patient', 'sub-prodromal', 'sub-swedd']):
                    # Check if it contains MAT files directly (flat structure)
                    mat_files = list(item.glob('*.mat'))
                    if mat_files:
                        # Flat structure: sub-patient3122 contains MAT files directly
                        label_class = get_label_class(name_lower)
                        flat_subject_folders.append((item, label_class))
                    else:
                        # Nested structure: sub-control/ contains patient folders
                        category_folders.append(item)
        
        if not category_folders and not flat_subject_folders:
            # Fallback: look for any folders with 'sub'
            for item in self.data_dir.iterdir():
                if item.is_dir() and 'sub' in item.name.lower():
                    mat_files = list(item.glob('*.mat'))
                    if mat_files:
                        label_class = get_label_class(item.name.lower())
                        flat_subject_folders.append((item, label_class))
                    else:
                        category_folders.append(item)
        
        if not category_folders and not flat_subject_folders:
            print(f"✗ Found 0 subject folders in {self.data_dir}")
            print(f"  Expected structure (flat or nested):")
            print(f"    Flat: {self.data_dir}/sub-patient3122/*.mat")
            print(f"    Nested: {self.data_dir}/sub-control/patient1234/*.mat")
            self.sequences = []
            self.fc_matrices = []
            self.metadata = []
            return
        
        # Now find all patient folders
        subject_folders = []
        
        # Add flat structure folders
        for subject_folder, label_class in flat_subject_folders:
            subject_folders.append((subject_folder, label_class))
        
        # Add nested structure folders
        for cat_folder in category_folders:
            # Extract label class from category folder name
            cat_name_lower = cat_folder.name.lower()
            label_class = get_label_class(cat_name_lower)
            label_name = get_class_name(label_class)
            
            # Look for patient folders inside
            patient_count = 0
            for patient_folder in cat_folder.iterdir():
                if patient_folder.is_dir():
                    subject_folders.append((patient_folder, label_class))
                    patient_count += 1
            
            if patient_count > 0:
                print(f"  {cat_folder.name}: {patient_count} patient folders ({label_name}, label={label_class})")
        
        if flat_subject_folders:
            print(f"✓ Found {len(flat_subject_folders)} flat subject folders (files directly in folder)")
        if category_folders:
            print(f"✓ Found {len(category_folders)} category folders with nested structure")
        
        if max_files:
            subject_folders = subject_folders[:max_files]
            print(f"  (Limited to {max_files} folders)")
        
        print(f"✓ Found {len(subject_folders)} total patient folders")
        
        # Load labels if provided
        label_dict = {}
        age_dict = {}
        sex_dict = {}
        if self.labels_file and Path(self.labels_file).exists():
            df = pd.read_csv(self.labels_file)
            
            # Determine subject ID column (PATNO for PPMI, sub_id/subject_id for others)
            subject_id_col = None
            for col in ['PATNO', 'sub_id', 'subject_id', 'Subject', 'subjectkey']:
                if col in df.columns:
                    subject_id_col = col
                    break
            
            if subject_id_col is None:
                print(f"  ⚠️  Warning: Could not find subject ID column. Available: {df.columns.tolist()[:5]}")
            else:
                for _, row in df.iterrows():
                    sub_id = str(row.get(subject_id_col, ''))
                    if not sub_id or sub_id == 'nan':
                        continue
                    
                    # Extract diagnosis label
                    label_value = row.get(self.label_column, None)
                    if label_value is not None:
                        if isinstance(label_value, str):
                            label_lower = label_value.lower().strip()
                            if 'control' in label_lower or 'healthy' in label_lower or 'normal' in label_lower:
                                label_dict[sub_id] = 0
                            elif 'swedd' in label_lower:
                                # SWEDD = 1 (4-class classification)
                                label_dict[sub_id] = 1
                            elif 'prodromal' in label_lower or 'prodrom' in label_lower:
                                # Prodromal = 2 (4-class classification)
                                label_dict[sub_id] = 2
                            elif ('patient' in label_lower or 'pd' in label_lower or 
                                  'parkinson' in label_lower or 'disease' in label_lower):
                                # PD = 3 (4-class classification)
                                label_dict[sub_id] = 3
                            else:
                                # Unknown string label - default to Control (0) for safety
                                print(f"  ⚠️  Warning: Unknown label '{label_value}' for subject {sub_id}, defaulting to Control (0)")
                                label_dict[sub_id] = 0
                        else:
                            # Numeric label - keep as-is for 4-class: 0=Control, 1=SWEDD, 2=Prodromal, 3=PD
                            if pd.notna(label_value):
                                label_dict[sub_id] = int(max(0, min(3, label_value)))  # Clamp to [0, 3]
                            else:
                                label_dict[sub_id] = 0  # Default to Control if NaN
                    
                    # Extract age (from BIRTHDT - calculate age at scan time if INFODT available)
                    if 'BIRTHDT' in df.columns:
                        birth_date = row.get('BIRTHDT')
                        if pd.notna(birth_date):
                            # Try to calculate age
                            scan_date = row.get('INFODT', None) if 'INFODT' in df.columns else None
                            try:
                                if scan_date and pd.notna(scan_date):
                                    # Parse dates and calculate age
                                    birth_dt = pd.to_datetime(birth_date, errors='coerce')
                                    scan_dt = pd.to_datetime(scan_date, errors='coerce')
                                    if pd.notna(birth_dt) and pd.notna(scan_dt):
                                        age_years = (scan_dt - birth_dt).days / 365.25
                                        if 0 < age_years < 120:  # Sanity check
                                            age_dict[sub_id] = float(age_years)
                                else:
                                    # Fallback: try to extract year from BIRTHDT
                                    birth_str = str(birth_date)
                                    import re
                                    year_match = re.search(r'(\d{4})', birth_str)
                                    if year_match:
                                        birth_year = int(year_match.group(1))
                                        if 1900 <= birth_year <= 2024:
                                            # Approximate age (current year - birth year)
                                            current_year = datetime.now().year
                                            age_years = current_year - birth_year
                                            if 0 < age_years < 120:
                                                age_dict[sub_id] = float(age_years)
                            except Exception as e:
                                pass  # Skip if date parsing fails
                    
                    # Extract sex (SEX column: typically 1=Male, 2=Female or M/F)
                    if 'SEX' in df.columns:
                        sex_value = row.get('SEX')
                        if pd.notna(sex_value):
                            try:
                                if isinstance(sex_value, str):
                                    sex_str = sex_value.upper()
                                    if sex_str in ['M', 'MALE', '1']:
                                        sex_dict[sub_id] = 1  # Male
                                    elif sex_str in ['F', 'FEMALE', '2']:
                                        sex_dict[sub_id] = 0  # Female
                                else:
                                    # Numeric: handle different encodings
                                    try:
                                        sex_num = float(sex_value)
                                        # PPMI might use 1=Male, 2=Female OR 0=Female, 1=Male
                                        # Check range to determine encoding
                                        if sex_num == 1.0:
                                            sex_dict[sub_id] = 1  # Male
                                        elif sex_num == 2.0:
                                            sex_dict[sub_id] = 0  # Female
                                        elif sex_num == 0.0:
                                            sex_dict[sub_id] = 0  # Female
                                        else:
                                            # Try to infer: if value is 0-1 range, assume 0=F, 1=M
                                            if 0 <= sex_num <= 1:
                                                sex_dict[sub_id] = int(sex_num)
                                    except (ValueError, TypeError):
                                        pass
                            except Exception:
                                pass
                
                print(f"✓ Loaded labels for {len(label_dict)} subjects from {self.labels_file}")
                if age_dict:
                    print(f"  Age data: {len(age_dict)} subjects (range: {min(age_dict.values()):.1f}-{max(age_dict.values()):.1f} years)")
                if sex_dict:
                    male_count = sum(1 for v in sex_dict.values() if v == 1)
                    female_count = sum(1 for v in sex_dict.values() if v == 0)
                    print(f"  Sex data: {len(sex_dict)} subjects (Male={male_count}, Female={female_count})")
        
        # Store age and sex dicts for later use
        self.age_dict = age_dict
        self.sex_dict = sex_dict
        
        # Load data from each subject
        self.sequences = []
        self.fc_matrices = []
        self.metadata = []
        
        print("Loading timeseries and FC matrices...")
        for patient_folder, label_class in tqdm(subject_folders, desc="Loading subjects"):
            try:
                # Extract patient ID (extract numeric ID to match PATNO in CSV)
                patient_id = patient_folder.name
                import re
                # Extract numeric ID from folder name (e.g., "sub-patient3122" -> "3122")
                id_match = re.search(r'(\d+)', patient_id)
                if id_match:
                    patient_id = id_match.group(1)  # Use just the numeric ID to match PATNO
                else:
                    # Fallback: remove 'sub-' prefix if present
                    patient_id = patient_id.replace('sub-', '', 1)
                
                # Determine label
                if patient_id in label_dict:
                    # Label from CSV - should be 0, 1, 2, or 3 for 4-class PPMI
                    label = label_dict[patient_id]
                    if isinstance(label, (int, float)):
                        label = float(max(0, min(3, int(label))))  # Clamp to [0, 3] for 4-class
                    elif isinstance(label, str):
                        # Handle string labels for 4-class PPMI
                        label_lower = label.lower().strip()
                        if 'control' in label_lower or 'healthy' in label_lower or 'normal' in label_lower:
                            label = 0.0  # Control
                        elif 'swedd' in label_lower:
                            label = 1.0  # SWEDD
                        elif 'prodromal' in label_lower or 'prodrom' in label_lower:
                            label = 2.0  # Prodromal
                        elif 'pd' in label_lower or 'parkinson' in label_lower or 'patient' in label_lower or 'disease' in label_lower:
                            label = 3.0  # PD
                        else:
                            # Default to Control if unknown
                            label = 0.0
                    else:
                        # Default to Control if unknown
                        label = 0.0
                elif self.label_column == 'diagnosis':
                    # Use 4-class label from category folder (already 0, 1, 2, or 3 from get_label_class)
                    label = float(label_class)  # Control=0, SWEDD=1, Prodromal=2, PD=3 (4-class)
                else:
                    # Try to extract from filename if available
                    label = self._extract_label_from_filename(patient_id) if hasattr(self, '_extract_label_from_filename') else float(label_class)
                    # Ensure 4-class mapping (0, 1, 2, or 3)
                    if isinstance(label, (int, float)):
                        label = float(max(0, min(3, int(label))))  # Clamp to [0, 3]
                    else:
                        # Default based on folder
                        label = float(label_class)
                
                # Extract age and sex from labels file if available
                age = self.age_dict.get(patient_id, None) if hasattr(self, 'age_dict') else None
                sex = self.sex_dict.get(patient_id, None) if hasattr(self, 'sex_dict') else None
                
                # Load timeseries MAT file
                ts_file = None
                ts_patterns = [
                    '*features_timeseries*.mat',
                    '*timeseries*.mat',
                    'timeseries.mat',
                    '*BOLD*.mat',
                    '*bold*.mat',
                    '*.mat'
                ]
                
                for pattern in ts_patterns:
                    if ts_file is None:
                        candidates = list(patient_folder.glob(pattern))
                        # Filter out correlation/FC files
                        candidates = [f for f in candidates if 'correlation' not in f.name.lower() 
                                    and 'fc' not in f.name.lower() and 'connectivity' not in f.name.lower()]
                        if candidates:
                            ts_file = candidates[0]
                            break
                
                if ts_file is None:
                    print(f"  ⚠️  No timeseries file in {patient_folder.name}, skipping...")
                    continue
                
                # Load MAT file
                mat_data = sio.loadmat(str(ts_file))
                
                # Find timeseries array
                timeseries = None
                for key in ['timeseries', 'BOLD', 'bold', 'data', 'ts', 'features_timeseries']:
                    if key in mat_data:
                        timeseries = mat_data[key]
                        break
                
                if timeseries is None:
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            timeseries = mat_data[key]
                            break
                
                if timeseries is None:
                    print(f"  ⚠️  No timeseries array in {ts_file.name}, skipping...")
                    continue
                
                # Ensure shape is [T, N_ROI]
                timeseries = np.array(timeseries, dtype=np.float32)
                if timeseries.shape[1] > timeseries.shape[0]:
                    timeseries = timeseries.T
                
                # Load FC/correlation matrix
                fc_matrix = None
                fc_patterns = [
                    '*correlation*.mat',
                    'fc.mat',
                    '*FC*.mat',
                    '*connectivity*.mat',
                    '*corr*.mat'
                ]
                
                for pattern in fc_patterns:
                    if fc_matrix is None:
                        candidates = list(patient_folder.glob(pattern))
                        if candidates:
                            fc_file = candidates[0]
                            try:
                                fc_data = sio.loadmat(str(fc_file))
                                for key in ['correlation_matrix', 'fc', 'FC', 'connectivity', 'corr', 'R']:
                                    if key in fc_data:
                                        fc_matrix = np.array(fc_data[key], dtype=np.float32)
                                        break
                                if fc_matrix is None:
                                    for key in fc_data.keys():
                                        if not key.startswith('__'):
                                            fc_matrix = np.array(fc_data[key], dtype=np.float32)
                                            break
                            except:
                                pass
                            break
                
                if fc_matrix is None:
                    fc_matrix = np.eye(timeseries.shape[1], dtype=np.float32)
                
                # Create segments
                segments = self._create_segments(timeseries)
                
                for segment in segments:
                    self.sequences.append(segment)
                    self.fc_matrices.append(fc_matrix)
                    self.metadata.append({
                        'subject': patient_id,
                        'file': str(ts_file),
                        'label': label,
                        'labels': label,
                        'age': age,
                        'sex': sex
                    })
                    
            except Exception as e:
                print(f"  ✗ Error loading {patient_folder.name}: {e}")
                continue
        
        if self.single_window_per_subject:
            print(f"✓ Created {len(self.sequences)} sequence segments from {len(subject_folders)} subjects (single window per subject)")
        else:
            print(f"✓ Created {len(self.sequences)} sequence segments from {len(subject_folders)} subjects (sliding window)")
        
        # Normalize if requested
        if self.normalize and len(self.sequences) > 0:
            try:
                all_data = np.concatenate(self.sequences, axis=0)
                self.mean = np.mean(all_data, axis=0, keepdims=True)
                self.std = np.std(all_data, axis=0, keepdims=True) + 1e-8
            except ValueError as e:
                print(f"  ⚠️  Warning: Could not concatenate sequences for normalization: {e}")
                print(f"     Using per-sequence normalization")
                self.mean = None
                self.std = None
        else:
            self.mean = None
            self.std = None
    
    def _create_segments(self, data: np.ndarray):
        """Create sliding window segments or single window per subject"""
        segments = []
        num_timepoints = data.shape[0]
        
        if self.single_window_per_subject:
            # Single window mode: take only the first window
            if num_timepoints < self.sequence_length:
                # Pad if too short
                pad_length = self.sequence_length - num_timepoints
                padded = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')
                segments.append(padded)
            else:
                # Take first window
                segments.append(data[:self.sequence_length])
        else:
            # Sliding window mode (default)
            if num_timepoints < self.sequence_length:
                # Pad if too short (same as single window mode for short sequences)
                pad_length = self.sequence_length - num_timepoints
                padded = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')
                segments.append(padded)
            else:
                # Create overlapping segments
                for start in range(0, num_timepoints - self.sequence_length + 1, self.stride):
                    end = start + self.sequence_length
                    segments.append(data[start:end])
        
        return segments
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        fc = self.fc_matrices[idx]
        meta = self.metadata[idx].copy()
        
        # Normalize if requested
        if self.normalize:
            if self.mean is not None:
                seq = (seq - self.mean) / self.std
            else:
                # Per-sequence normalization
                seq_mean = np.mean(seq, axis=0, keepdims=True)
                seq_std = np.std(seq, axis=0, keepdims=True) + 1e-8
                seq = (seq - seq_mean) / seq_std
        
        # Convert to tensor
        sequence = torch.from_numpy(seq.astype(np.float32))
        
        # Add FC matrix to metadata
        meta['fc_matrix'] = torch.from_numpy(fc.astype(np.float32))
        
        # Return tuple (sequence, metadata) to match extract_features expectations
        return sequence, meta


class PPMISCFCCsvDataset(Dataset):
    """
    PPMI SC-FC dataset for atlas-specific CSV BOLD time series:
        .../PPMI-SC-FC/<atlas>/BOLD/sub-<PATNO>_..._time_series.csv

    Labels come from metadata.csv with subject ID (e.g., PATNO or Subject)
    and diagnosis column (e.g., Group) mapping into 4 classes:
        0: Control, 1: SWEDD, 2: Prodromal, 3: PD
    """
    def __init__(
        self,
        data_dir: str,
        labels_file: str = None,
        label_column: str = 'Group',
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        max_files: int = None,
        single_window_per_subject: bool = False,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.label_column = label_column
        self.labels_file = labels_file
        self.single_window_per_subject = single_window_per_subject

        self.sequences = []
        self.fc_matrices = []
        self.metadata = []

        if not self.data_dir.exists():
            print(f"✗ PPMI SC-FC data directory does not exist: {self.data_dir}")
            return

        # List CSV time-series files
        all_files = sorted(self.data_dir.glob("sub-*_time_series.csv"))
        if not all_files:
            # Fallback: any CSV starting with sub-
            all_files = sorted(self.data_dir.glob("sub-*.csv"))

        if max_files is not None:
            all_files = all_files[:max_files]

        print(f"Found {len(all_files)} PPMI SC-FC CSV files in {self.data_dir}")

        # Load labels
        label_dict = {}
        if labels_file and Path(labels_file).exists():
            try:
                df = pd.read_csv(labels_file)
                print(f"✓ Loaded PPMI metadata with {len(df)} rows from {labels_file}")
                # Subject ID column: try common names
                subject_col = None
                for col in ['Subject', 'PATNO', 'patno', 'PATNO_x', 'PATNO_y']:
                    if col in df.columns:
                        subject_col = col
                        break

                # Diagnosis / group column
                diag_col = None
                candidates = [self.label_column, 'Group', 'diagnosis', 'DX', 'dx']
                for col in candidates:
                    if col in df.columns:
                        diag_col = col
                        break

                if subject_col and diag_col:
                    for _, row in df.iterrows():
                        raw_id = str(row[subject_col]).strip()
                        if not raw_id or raw_id == 'nan':
                            continue
                        # Normalize to numeric core, e.g., "74743"
                        numeric = ''.join(ch for ch in raw_id if ch.isdigit())
                        keys = set()
                        if raw_id:
                            keys.add(raw_id)
                        if numeric:
                            keys.add(numeric.lstrip('0') or numeric)
                            keys.add(f"sub-{numeric.lstrip('0') or numeric}")

                        dx_val = row[diag_col]
                        label = None
                        if isinstance(dx_val, str):
                            dx_lower = dx_val.lower().strip()
                            if 'control' in dx_lower or 'healthy' in dx_lower or 'normal' in dx_lower:
                                label = 0
                            elif 'swedd' in dx_lower:
                                label = 1
                            elif 'prodromal' in dx_lower or 'prodrom' in dx_lower:
                                label = 2
                            elif ('patient' in dx_lower or 'pd' in dx_lower or
                                  'parkinson' in dx_lower or 'disease' in dx_lower):
                                label = 3
                        else:
                            if pd.notna(dx_val):
                                try:
                                    label = int(max(0, min(3, int(dx_val))))
                                except Exception:
                                    pass

                        if label is not None:
                            for k in keys:
                                label_dict[k] = label

                    print(f"✓ PPMI SC-FC labels for {len(label_dict)} subject IDs")
                else:
                    print(f"  ⚠️  Could not find subject/diagnosis columns in PPMI metadata.")
                    print(f"      Columns: {list(df.columns)[:10]}")
            except Exception as e:
                print(f"  ⚠️  Failed to load PPMI labels from {labels_file}: {e}")

        n_skipped_label = 0

        # Load each CSV and create sliding windows
        for csv_file in tqdm(all_files, desc="Loading PPMI SC-FC subjects"):
            try:
                name = csv_file.name
                # Extract subject ID from filename: sub-74743_...
                subj_core = None
                if name.startswith("sub-"):
                    core = name.split('_')[0]  # sub-74743
                    subj_core = core  # keep "sub-74743"
                if subj_core is None:
                    continue

                # Map to label
                label = None
                if subj_core in label_dict:
                    label = label_dict[subj_core]
                else:
                    # Try numeric core
                    num = ''.join(ch for ch in subj_core if ch.isdigit())
                    num = num.lstrip('0') or num
                    if num in label_dict:
                        label = label_dict[num]

                if label is None:
                    n_skipped_label += 1
                    continue

                df_ts = pd.read_csv(csv_file)
                ts = df_ts.values.astype(np.float32)
                # Ensure [T, N]
                if ts.ndim != 2:
                    continue
                if ts.shape[1] > ts.shape[0]:
                    ts = ts.T

                # Replace NaNs
                if np.isnan(ts).any():
                    ts = np.nan_to_num(ts, nan=0.0)

                T = ts.shape[0]
                if T < self.sequence_length:
                    continue

                # Sliding windows
                starts = [0] if self.single_window_per_subject else range(0, T - self.sequence_length + 1, self.stride)
                for start in starts:
                    end = start + self.sequence_length
                    if end > T:
                        break
                    window = ts[start:end].copy()
                    if self.normalize:
                        mean = window.mean(axis=0, keepdims=True)
                        std = window.std(axis=0, keepdims=True) + 1e-6
                        window = (window - mean) / std

                    # FC matrix per window
                    fc = np.corrcoef(window.T)
                    if np.isnan(fc).any():
                        fc = np.nan_to_num(fc, nan=0.0)

                    self.sequences.append(window.astype(np.float32))
                    self.fc_matrices.append(fc.astype(np.float32))
                    self.metadata.append({
                        'subject_id': subj_core,
                        'label': int(label),
                    })
            except Exception as e:
                print(f"  ⚠️  Error loading {csv_file.name}: {e}")

        print(f"✓ PPMI SC-FC: {len(self.sequences)} windows with labels")
        if n_skipped_label > 0:
            print(f"  ⚠️  Skipped {n_skipped_label} files without matching labels")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        fc = self.fc_matrices[idx]
        meta = self.metadata[idx].copy()
        # Add fc_matrix to metadata for consistency
        meta['fc_matrix'] = torch.from_numpy(fc.astype(np.float32))
        sequence = torch.from_numpy(seq.astype(np.float32))
        return sequence, meta

# Register PPMI dataset (default nested/mat implementation)
DATASET_REGISTRY['PPMI'] = PPMIDatasetNested

# Create TAOWU dataset class
class TAOWUDataset:
    """
    TAOWU dataset loader for structure:
    data/taowu/sub-control(patient)xxxx/AAL116_timeseires.mat and AAL116_correlation_matrix.mat
    data/taowu/sub-patient(patient)xxxx/AAL116_timeseires.mat and AAL116_correlation_matrix.mat
    """
    def __init__(
        self,
        data_dir: str,
        labels_file: str = None,
        label_column: str = 'diagnosis',
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        max_files: int = None,
        single_window_per_subject: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.label_column = label_column
        self.labels_file = labels_file
        self.single_window_per_subject = single_window_per_subject
        
        # Find all subject folders matching pattern: sub-control(patient)xxxx or sub-patient(patient)xxxx
        subject_folders = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                name_lower = item.name.lower()
                # Check for sub-control(patient) or sub-patient(patient) pattern
                if 'sub-control' in name_lower or 'sub-patient' in name_lower:
                    # Extract label from folder name
                    is_control = 'control' in name_lower or 'sub-control' in name_lower
                    subject_folders.append((item, is_control))
        
        if max_files:
            subject_folders = subject_folders[:max_files]
        
        print(f"Found {len(subject_folders)} subject folders")
        
        # Load labels if provided
        label_dict = {}
        if self.labels_file and Path(self.labels_file).exists():
            df = pd.read_csv(self.labels_file)
            # Try multiple possible subject ID column names
            subject_col = None
            for col in ['Subject', 'subject', 'sub_id', 'subject_id', 'PATNO']:
                if col in df.columns:
                    subject_col = col
                    break
            if subject_col:
                for _, row in df.iterrows():
                    sub_id = str(row[subject_col]).strip()
                    if sub_id:
                        label_dict[sub_id] = row.get(self.label_column, None)
                print(f"✓ Loaded labels for {len(label_dict)} subjects from {self.labels_file}")
            else:
                print(f"  ⚠️  Could not find subject column in labels file")
        
        # Load data from each subject
        self.sequences = []
        self.fc_matrices = []
        self.metadata = []
        
        print("Loading timeseries and correlation matrices...")
        for subject_folder, is_control in tqdm(subject_folders, desc="Loading subjects"):
            try:
                # Extract subject ID - extract numeric ID to match metadata CSV
                import re
                folder_name = subject_folder.name
                # Extract numeric ID from folder name (e.g., "sub-control032057" -> "032057")
                id_match = re.search(r'(\d+)', folder_name)
                if id_match:
                    subject_id = id_match.group(1)
                else:
                    # Fallback: remove 'sub-' prefix if present
                    subject_id = folder_name.replace('sub-', '', 1)
                
                # Determine label - first try label_dict, then infer from folder name
                if subject_id in label_dict:
                    raw_label = label_dict[subject_id]
                    # Convert text labels to numeric (Control=0, Patient/Disease=1)
                    if isinstance(raw_label, str):
                        label_lower = raw_label.lower().strip()
                        if 'control' in label_lower or 'healthy' in label_lower or 'normal' in label_lower:
                            label = 0.0
                        elif 'pd' in label_lower or 'parkinson' in label_lower or 'patient' in label_lower or 'disease' in label_lower:
                            label = 1.0  # PD, Patient, Disease
                        else:
                            label = 1.0  # Default to patient/disease for unknown labels
                    else:
                        label = float(raw_label) if raw_label is not None else (0.0 if is_control else 1.0)
                else:
                    # Fallback: use folder name to determine label (Control=0, Patient=1)
                    label = 0.0 if is_control else 1.0
                
                # Load timeseries MAT file (note: typo in filename "timeseires")
                ts_file = None
                ts_patterns = [
                    'AAL116_timeseires.mat',  # Original typo
                    'AAL116_timeseries.mat',  # Corrected version
                    '*timeseires*.mat',
                    '*timeseries*.mat',
                    '*.mat'
                ]
                
                for pattern in ts_patterns:
                    if ts_file is None:
                        candidates = list(subject_folder.glob(pattern))
                        # Filter out correlation files
                        candidates = [f for f in candidates if 'correlation' not in f.name.lower()]
                        if candidates:
                            ts_file = candidates[0]
                            break
                
                if ts_file is None:
                    print(f"  ⚠️  No timeseries file in {subject_folder.name}, skipping...")
                    continue
                
                # Load MAT file
                mat_data = sio.loadmat(str(ts_file))
                
                # Find timeseries array
                timeseries = None
                for key in ['timeseries', 'BOLD', 'bold', 'data', 'ts', 'AAL116_timeseires', 'AAL116_timeseries']:
                    if key in mat_data:
                        timeseries = mat_data[key]
                        break
                
                if timeseries is None:
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            timeseries = mat_data[key]
                            break
                
                if timeseries is None:
                    print(f"  ⚠️  No timeseries array in {ts_file.name}, skipping...")
                    continue
                
                # Ensure shape is [T, N_ROI]
                timeseries = np.array(timeseries, dtype=np.float32)
                if timeseries.shape[1] > timeseries.shape[0]:
                    timeseries = timeseries.T
                
                # Load correlation matrix
                fc_matrix = None
                fc_patterns = [
                    'AAL116_correlation_matrix.mat',
                    '*correlation*.mat',
                    'fc.mat',
                    '*FC*.mat',
                    '*connectivity*.mat',
                    '*corr*.mat'
                ]
                
                for pattern in fc_patterns:
                    if fc_matrix is None:
                        candidates = list(subject_folder.glob(pattern))
                        if candidates:
                            fc_file = candidates[0]
                            try:
                                fc_data = sio.loadmat(str(fc_file))
                                for key in ['correlation_matrix', 'AAL116_correlation_matrix', 'fc', 'FC', 'connectivity', 'corr', 'R']:
                                    if key in fc_data:
                                        fc_matrix = np.array(fc_data[key], dtype=np.float32)
                                        break
                                if fc_matrix is None:
                                    for key in fc_data.keys():
                                        if not key.startswith('__'):
                                            fc_matrix = np.array(fc_data[key], dtype=np.float32)
                                            break
                            except:
                                pass
                            break
                
                if fc_matrix is None:
                    fc_matrix = np.eye(timeseries.shape[1], dtype=np.float32)
                
                # Create segments
                segments = self._create_segments(timeseries)
                
                for segment in segments:
                    self.sequences.append(segment)
                    self.fc_matrices.append(fc_matrix)
                    self.metadata.append({
                        'subject': subject_id,
                        'file': str(ts_file),
                        'label': label,
                        'labels': label
                    })
                    
            except Exception as e:
                print(f"  ✗ Error loading {subject_folder.name}: {e}")
                continue
        
        if self.single_window_per_subject:
            print(f"✓ Created {len(self.sequences)} sequence segments from {len(subject_folders)} subjects (single window per subject)")
        else:
            print(f"✓ Created {len(self.sequences)} sequence segments from {len(subject_folders)} subjects (sliding window)")
        
        # Check for consistent ROI counts
        if len(self.sequences) > 0:
            roi_counts = [seq.shape[1] for seq in self.sequences]
            unique_roi_counts, counts = np.unique(roi_counts, return_counts=True)
            if len(unique_roi_counts) > 1:
                print(f"  ⚠️  Warning: Sequences have different ROI counts: {dict(zip(unique_roi_counts, counts))}")
                # Filter to keep only sequences with the most common ROI count
                most_common_roi = unique_roi_counts[np.argmax(counts)]
                print(f"     Keeping sequences with {most_common_roi} ROIs (most common)")
                filtered_sequences = []
                filtered_fc = []
                filtered_metadata = []
                removed_count = 0
                for seq, fc, meta in zip(self.sequences, self.fc_matrices, self.metadata):
                    if seq.shape[1] == most_common_roi:
                        filtered_sequences.append(seq)
                        filtered_fc.append(fc)
                        filtered_metadata.append(meta)
                    else:
                        removed_count += 1
                self.sequences = filtered_sequences
                self.fc_matrices = filtered_fc
                self.metadata = filtered_metadata
                if removed_count > 0:
                    print(f"     Removed {removed_count} sequences with mismatched ROI counts")
                self.normalize_per_sequence = False
            else:
                self.normalize_per_sequence = False
        
        # Normalize if requested
        if self.normalize and len(self.sequences) > 0 and not self.normalize_per_sequence:
            # All sequences have same ROI count, can concatenate
            try:
                all_data = np.concatenate(self.sequences, axis=0)
                self.mean = np.mean(all_data, axis=0, keepdims=True)
                self.std = np.std(all_data, axis=0, keepdims=True) + 1e-8
            except ValueError as e:
                print(f"  ⚠️  Warning: Could not concatenate sequences for normalization: {e}")
                print(f"     Falling back to per-sequence normalization.")
                self.mean = None
                self.std = None
                self.normalize_per_sequence = True
        else:
            self.mean = None
            self.std = None
            if not hasattr(self, 'normalize_per_sequence'):
                self.normalize_per_sequence = False
    
    def _create_segments(self, data: np.ndarray):
        """Create sliding window segments or single window per subject"""
        segments = []
        num_timepoints = data.shape[0]
        
        if self.single_window_per_subject:
            # Single window mode: take only the first window
            if num_timepoints < self.sequence_length:
                # Pad if too short
                pad_length = self.sequence_length - num_timepoints
                padded = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')
                segments.append(padded)
            else:
                # Take first window
                segments.append(data[:self.sequence_length])
        else:
            # Sliding window mode (default)
            for start in range(0, num_timepoints - self.sequence_length + 1, self.stride):
                end = start + self.sequence_length
                segments.append(data[start:end])
        
        return segments
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        fc = self.fc_matrices[idx]
        meta = self.metadata[idx].copy()
        
        # Normalize if requested
        if self.normalize:
            if hasattr(self, 'normalize_per_sequence') and self.normalize_per_sequence:
                # Per-sequence normalization (for variable ROI counts)
                seq_mean = np.mean(seq, axis=0, keepdims=True)
                seq_std = np.std(seq, axis=0, keepdims=True) + 1e-8
                seq = (seq - seq_mean) / seq_std
            elif self.mean is not None:
                # Global normalization (all sequences have same ROI count)
                seq = (seq - self.mean) / self.std
        
        # Convert to tensor
        sequence = torch.from_numpy(seq.astype(np.float32))
        
        # Add FC matrix to metadata (matching PPMI format)
        meta['fc_matrix'] = torch.from_numpy(fc.astype(np.float32))
        
        # Return tuple (sequence, metadata) to match extract_features expectations
        return sequence, meta

# Create NEUROCON dataset class (same structure as TAOWU)
class NEUROCONDataset:
    """
    NEUROCON dataset loader for structure:
    data/neurocon/sub-control(patient)xxxx/AAL116_timeseires.mat and AAL116_correlation_matrix.mat
    data/neurocon/sub-patient(patient)xxxx/AAL116_timeseires.mat and AAL116_correlation_matrix.mat
    """
    def __init__(
        self,
        data_dir: str,
        labels_file: str = None,
        label_column: str = 'Group',  # Changed from 'diagnosis' to match metadata format (Subject,Group,Age,Sex)
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        max_files: int = None,
        single_window_per_subject: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.label_column = label_column
        self.labels_file = labels_file
        self.single_window_per_subject = single_window_per_subject
        
        # Find all subject folders matching pattern: sub-control(patient)xxxx or sub-patient(patient)xxxx
        subject_folders = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                name_lower = item.name.lower()
                # Check for sub-control(patient) or sub-patient(patient) pattern
                if 'sub-control' in name_lower or 'sub-patient' in name_lower:
                    # Extract label from folder name
                    is_control = 'control' in name_lower or 'sub-control' in name_lower
                    subject_folders.append((item, is_control))
        
        if max_files:
            subject_folders = subject_folders[:max_files]
        
        print(f"Found {len(subject_folders)} subject folders")
        
        # Load labels if provided
        label_dict = {}
        if self.labels_file and Path(self.labels_file).exists():
            df = pd.read_csv(self.labels_file)
            # Try multiple possible subject ID column names
            subject_col = None
            for col in ['Subject', 'subject', 'sub_id', 'subject_id', 'PATNO']:
                if col in df.columns:
                    subject_col = col
                    break
            if subject_col:
                for _, row in df.iterrows():
                    sub_id = str(row[subject_col]).strip()
                    if sub_id:
                        label_dict[sub_id] = row.get(self.label_column, None)
                print(f"✓ Loaded labels for {len(label_dict)} subjects from {self.labels_file}")
            else:
                print(f"  ⚠️  Could not find subject column in labels file")
        
        # Load data from each subject
        self.sequences = []
        self.fc_matrices = []
        self.metadata = []
        
        print("Loading timeseries and correlation matrices...")
        for subject_folder, is_control in tqdm(subject_folders, desc="Loading subjects"):
            try:
                # Extract subject ID - extract numeric ID to match metadata CSV
                import re
                folder_name = subject_folder.name
                # Extract numeric ID from folder name (e.g., "sub-control032057" -> "032057")
                id_match = re.search(r'(\d+)', folder_name)
                if id_match:
                    subject_id = id_match.group(1)
                else:
                    # Fallback: remove 'sub-' prefix if present
                    subject_id = folder_name.replace('sub-', '', 1)
                
                # Determine label - first try label_dict, then infer from folder name
                if subject_id in label_dict:
                    raw_label = label_dict[subject_id]
                    # Convert text labels to numeric (Control=0, Patient/Disease=1)
                    if isinstance(raw_label, str):
                        label_lower = raw_label.lower().strip()
                        if 'control' in label_lower or 'healthy' in label_lower or 'normal' in label_lower:
                            label = 0.0
                        elif 'pd' in label_lower or 'parkinson' in label_lower or 'patient' in label_lower or 'disease' in label_lower:
                            label = 1.0  # PD, Patient, Disease
                        else:
                            label = 1.0  # Default to patient/disease for unknown labels
                    else:
                        label = float(raw_label) if raw_label is not None else (0.0 if is_control else 1.0)
                else:
                    # Fallback: use folder name to determine label (Control=0, Patient=1)
                    label = 0.0 if is_control else 1.0
                
                # Load timeseries MAT file (note: typo in filename "timeseires")
                ts_file = None
                ts_patterns = [
                    'AAL116_timeseires.mat',  # Original typo
                    'AAL116_timeseries.mat',  # Corrected version
                    '*timeseires*.mat',
                    '*timeseries*.mat',
                    '*.mat'
                ]
                
                for pattern in ts_patterns:
                    if ts_file is None:
                        candidates = list(subject_folder.glob(pattern))
                        # Filter out correlation files
                        candidates = [f for f in candidates if 'correlation' not in f.name.lower()]
                        if candidates:
                            ts_file = candidates[0]
                            break
                
                if ts_file is None:
                    print(f"  ⚠️  No timeseries file in {subject_folder.name}, skipping...")
                    continue
                
                # Load MAT file
                mat_data = sio.loadmat(str(ts_file))
                
                # Find timeseries array
                timeseries = None
                for key in ['timeseries', 'BOLD', 'bold', 'data', 'ts', 'AAL116_timeseires', 'AAL116_timeseries']:
                    if key in mat_data:
                        timeseries = mat_data[key]
                        break
                
                if timeseries is None:
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            timeseries = mat_data[key]
                            break
                
                if timeseries is None:
                    print(f"  ⚠️  No timeseries array in {ts_file.name}, skipping...")
                    continue
                
                # Ensure shape is [T, N_ROI]
                timeseries = np.array(timeseries, dtype=np.float32)
                if timeseries.shape[1] > timeseries.shape[0]:
                    timeseries = timeseries.T
                
                # Load correlation matrix
                fc_matrix = None
                fc_patterns = [
                    'AAL116_correlation_matrix.mat',
                    '*correlation*.mat',
                    'fc.mat',
                    '*FC*.mat',
                    '*connectivity*.mat',
                    '*corr*.mat'
                ]
                
                for pattern in fc_patterns:
                    if fc_matrix is None:
                        candidates = list(subject_folder.glob(pattern))
                        if candidates:
                            fc_file = candidates[0]
                            try:
                                fc_data = sio.loadmat(str(fc_file))
                                for key in ['correlation_matrix', 'AAL116_correlation_matrix', 'fc', 'FC', 'connectivity', 'corr', 'R']:
                                    if key in fc_data:
                                        fc_matrix = np.array(fc_data[key], dtype=np.float32)
                                        break
                                if fc_matrix is None:
                                    for key in fc_data.keys():
                                        if not key.startswith('__'):
                                            fc_matrix = np.array(fc_data[key], dtype=np.float32)
                                            break
                            except:
                                pass
                            break
                
                if fc_matrix is None:
                    fc_matrix = np.eye(timeseries.shape[1], dtype=np.float32)
                
                # Create segments
                segments = self._create_segments(timeseries)
                
                for segment in segments:
                    self.sequences.append(segment)
                    self.fc_matrices.append(fc_matrix)
                    self.metadata.append({
                        'subject': subject_id,
                        'file': str(ts_file),
                        'label': label,
                        'labels': label
                    })
                    
            except Exception as e:
                print(f"  ✗ Error loading {subject_folder.name}: {e}")
                continue
        
        if self.single_window_per_subject:
            print(f"✓ Created {len(self.sequences)} sequence segments from {len(subject_folders)} subjects (single window per subject)")
        else:
            print(f"✓ Created {len(self.sequences)} sequence segments from {len(subject_folders)} subjects (sliding window)")
        
        # Check for consistent ROI counts
        if len(self.sequences) > 0:
            roi_counts = [seq.shape[1] for seq in self.sequences]
            unique_roi_counts, counts = np.unique(roi_counts, return_counts=True)
            if len(unique_roi_counts) > 1:
                print(f"  ⚠️  Warning: Sequences have different ROI counts: {dict(zip(unique_roi_counts, counts))}")
                # Filter to keep only sequences with the most common ROI count
                most_common_roi = unique_roi_counts[np.argmax(counts)]
                print(f"     Keeping sequences with {most_common_roi} ROIs (most common)")
                filtered_sequences = []
                filtered_fc = []
                filtered_metadata = []
                removed_count = 0
                for seq, fc, meta in zip(self.sequences, self.fc_matrices, self.metadata):
                    if seq.shape[1] == most_common_roi:
                        filtered_sequences.append(seq)
                        filtered_fc.append(fc)
                        filtered_metadata.append(meta)
                    else:
                        removed_count += 1
                self.sequences = filtered_sequences
                self.fc_matrices = filtered_fc
                self.metadata = filtered_metadata
                if removed_count > 0:
                    print(f"     Removed {removed_count} sequences with mismatched ROI counts")
                self.normalize_per_sequence = False
            else:
                self.normalize_per_sequence = False
        
        # Normalize if requested
        if self.normalize and len(self.sequences) > 0 and not self.normalize_per_sequence:
            # All sequences have same ROI count, can concatenate
            try:
                all_data = np.concatenate(self.sequences, axis=0)
                self.mean = np.mean(all_data, axis=0, keepdims=True)
                self.std = np.std(all_data, axis=0, keepdims=True) + 1e-8
            except ValueError as e:
                print(f"  ⚠️  Warning: Could not concatenate sequences for normalization: {e}")
                print(f"     Falling back to per-sequence normalization.")
                self.mean = None
                self.std = None
                self.normalize_per_sequence = True
        else:
            self.mean = None
            self.std = None
            if not hasattr(self, 'normalize_per_sequence'):
                self.normalize_per_sequence = False
    
    def _create_segments(self, data: np.ndarray):
        """Create sliding window segments or single window per subject"""
        segments = []
        num_timepoints = data.shape[0]
        
        if self.single_window_per_subject:
            # Single window mode: take only the first window
            if num_timepoints < self.sequence_length:
                # Pad if too short
                pad_length = self.sequence_length - num_timepoints
                padded = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')
                segments.append(padded)
            else:
                # Take first window
                segments.append(data[:self.sequence_length])
        else:
            # Sliding window mode (default)
            for start in range(0, num_timepoints - self.sequence_length + 1, self.stride):
                end = start + self.sequence_length
                segments.append(data[start:end])
        
        return segments
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        fc = self.fc_matrices[idx]
        meta = self.metadata[idx].copy()
        
        # Normalize if requested
        if self.normalize:
            if hasattr(self, 'normalize_per_sequence') and self.normalize_per_sequence:
                # Per-sequence normalization (for variable ROI counts)
                seq_mean = np.mean(seq, axis=0, keepdims=True)
                seq_std = np.std(seq, axis=0, keepdims=True) + 1e-8
                seq = (seq - seq_mean) / seq_std
            elif self.mean is not None:
                # Global normalization (all sequences have same ROI count)
                seq = (seq - self.mean) / self.std
        
        # Convert to tensor
        sequence = torch.from_numpy(seq.astype(np.float32))
        
        # Add FC matrix to metadata (matching PPMI format)
        meta['fc_matrix'] = torch.from_numpy(fc.astype(np.float32))
        
        # Return tuple (sequence, metadata) to match extract_features expectations
        return sequence, meta

# Add TAOWU and NEUROCON to registry
DATASET_REGISTRY['TAOWU'] = TAOWUDataset
DATASET_REGISTRY['NEUROCON'] = NEUROCONDataset


# Create ADNI dataset class
class ADNIDataset:
    """
    ADNI dataset loader for structure:
    data/adni/BOLD/BOLD*.npz (timeseries)
    data/adni/FC/FC*.npz (functional connectivity matrices)
    
    NPZ files contain:
    - BOLD: timeseries data [T, N_ROI]
    - FC: correlation matrix [N_ROI, N_ROI]
    
    Labels from metadata.csv with columns including:
    - Subject: subject ID (e.g., "002_S_0295")
    - DX: diagnosis (CN, SMC, EMCI, LMCI, AD)
    - Age, Sex for demographics
    
    For 2-class classification: CN/SMC/EMCI → 0, LMCI/AD → 1
    
    Filename format: BOLD000000_adni_002S6652_task-rest_AAL_116.npz
    Subject ID extraction: "002S6652" -> "002_S_6652" (to match metadata format)
    """
    
    # Label mapping for 2-class classification
    LABEL_MAPPING = {
        'CN': 0, 'SMC': 0, 'EMCI': 0,  # Cognitively Normal group
        'LMCI': 1, 'AD': 1,             # Alzheimer's Disease group
    }
    
    def __init__(
        self,
        data_dir: str,
        labels_file: str = None,
        label_column: str = 'DX',
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        max_files: int = None,
        single_window_per_subject: bool = False,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.labels_file = labels_file
        self.label_column = label_column
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.normalize_per_sequence = True
        self.single_window_per_subject = single_window_per_subject
        
        self.sequences = []
        self.fc_matrices = []
        self.metadata = []
        
        # Determine FC directory (sibling to BOLD directory)
        self.fc_dir = self.data_dir.parent / 'FC'
        if not self.fc_dir.exists():
            print(f"  ⚠️  FC directory not found: {self.fc_dir}")
            print(f"     Will compute FC matrices from BOLD data")
        
        # Load labels from metadata.csv
        label_dict = {}
        age_dict = {}
        sex_dict = {}
        if labels_file and Path(labels_file).exists():
            try:
                df = pd.read_csv(labels_file)
                print(f"✓ Loaded metadata with {len(df)} rows from {labels_file}")
                
                # Find subject column
                subject_col = None
                for col in ['Subject', 'subject', 'PTID', 'SUB_ID', 'RID']:
                    if col in df.columns:
                        subject_col = col
                        break
                
                if subject_col:
                    for _, row in df.iterrows():
                        subj = str(row[subject_col]).strip()
                        if not subj or subj == 'nan':
                            continue
                        
                        # Extract label if label_column exists
                        if label_column in df.columns:
                            dx = str(row[label_column]).strip()
                            if dx in self.LABEL_MAPPING:
                                label_dict[subj] = self.LABEL_MAPPING[dx]
                        
                        # Extract age if available
                        if 'Age' in df.columns:
                            age_val = row.get('Age')
                            if pd.notna(age_val):
                                try:
                                    age_dict[subj] = float(age_val)
                                except:
                                    pass
                        
                        # Extract sex if available
                        if 'Sex' in df.columns:
                            sex_val = row.get('Sex')
                            if pd.notna(sex_val):
                                try:
                                    if isinstance(sex_val, str):
                                        sex_dict[subj] = 1 if sex_val.upper().strip() in ['M', 'MALE', '1'] else 0
                                    else:
                                        sex_dict[subj] = int(float(sex_val))
                                except:
                                    pass
                    
                    if label_dict:
                        print(f"✓ Loaded labels for {len(label_dict)} subjects (2-class: CN+SMC+EMCI=0, LMCI+AD=1)")
                    if age_dict:
                        print(f"  Age data: {len(age_dict)} subjects")
                    if sex_dict:
                        print(f"  Sex data: {len(sex_dict)} subjects")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load labels: {e}")
                import traceback
                traceback.print_exc()
        
        # Store age and sex dicts
        self.age_dict = age_dict
        self.sex_dict = sex_dict
        
        # Find all NPZ files in BOLD directory
        bold_files = sorted(self.data_dir.glob("BOLD*.npz"))
        if max_files:
            bold_files = bold_files[:max_files]
        
        print(f"Found {len(bold_files)} BOLD NPZ files in {data_dir}")
        
        # Helper function to extract and normalize subject ID from filename
        def extract_subject_id(filename):
            """
            Extract subject ID from filename and convert to metadata format.
            Example: "BOLD000000_adni_002S6652_task-rest_AAL_116.npz" -> "002_S_6652"
            """
            import re
            # Extract pattern like "002S6652" or "014S6920"
            match = re.search(r'_(\d+)S(\d+)_', filename.name)
            if match:
                # Convert "002S6652" -> "002_S_6652" to match metadata format
                part1 = match.group(1)
                part2 = match.group(2)
                return f"{part1}_S_{part2}"
            # Fallback: try to extract any numeric pattern
            match = re.search(r'(\d+)_S_(\d+)', filename.name)
            if match:
                return f"{match.group(1)}_S_{match.group(2)}"
            # Last resort: use stem without prefix
            return filename.stem.replace('BOLD', '').replace('_adni_', '').split('_')[0]
        
        # Load timeseries and FC from NPZ files
        for bold_file in tqdm(bold_files, desc="Loading ADNI subjects"):
            try:
                # Extract subject ID from filename
                subject_id = extract_subject_id(bold_file)
                
                # Load BOLD timeseries from NPZ
                try:
                    bold_data = np.load(str(bold_file))
                    # Try common keys for timeseries
                    timeseries = None
                    for key in ['bold', 'BOLD', 'timeseries', 'data', 'ts']:
                        if key in bold_data:
                            timeseries = bold_data[key].astype(np.float32)
                            break
                    if timeseries is None:
                        # Get first array that's not metadata
                        for key in bold_data.keys():
                            if not key.startswith('__'):
                                arr = bold_data[key]
                                if len(arr.shape) == 2:  # [T, ROI] format
                                    timeseries = arr.astype(np.float32)
                                    break
                    
                    if timeseries is None:
                        print(f"  ⚠️  Could not find timeseries in {bold_file.name}")
                        continue
                    
                    # Ensure shape is [T, N_ROI]
                    if timeseries.shape[1] > timeseries.shape[0]:
                        timeseries = timeseries.T
                    
                except Exception as e:
                    print(f"  ⚠️  Error loading {bold_file.name}: {e}")
                    continue
                
                if timeseries.shape[0] < 10:
                    continue  # Skip files with too little data
                
                # Handle NaN values
                if np.isnan(timeseries).any():
                    timeseries = np.nan_to_num(timeseries, nan=0.0)
                
                # Load FC matrix from FC directory
                fc_matrix = None
                # Find corresponding FC file (FC000000_adni_002S6652_task-rest_AAL_116.npz)
                fc_filename = bold_file.name.replace('BOLD', 'FC')
                fc_file = self.fc_dir / fc_filename
                
                if fc_file.exists():
                    try:
                        fc_data = np.load(str(fc_file))
                        for key in ['fc', 'FC', 'correlation', 'corr', 'connectivity']:
                            if key in fc_data:
                                fc_matrix = fc_data[key].astype(np.float32)
                                break
                        if fc_matrix is None:
                            for key in fc_data.keys():
                                if not key.startswith('__'):
                                    arr = fc_data[key]
                                    if len(arr.shape) == 2:  # [ROI, ROI] format
                                        fc_matrix = arr.astype(np.float32)
                                        break
                    except Exception as e:
                        print(f"  ⚠️  Error loading FC from {fc_file.name}: {e}")
                
                # Compute FC if not loaded
                if fc_matrix is None:
                    fc_matrix = np.corrcoef(timeseries.T)
                    fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
                
                # Get label (optional - for age/sex evaluation, labels_file may be None)
                label = None
                if label_dict:  # Only check labels if labels_file was provided
                    if subject_id in label_dict:
                        label = label_dict[subject_id]
                    else:
                        # Try partial match (handle format variations)
                        matched = False
                        for key in label_dict:
                            # Normalize both for comparison
                            key_norm = key.replace('_', '').replace('-', '').lower()
                            subj_norm = subject_id.replace('_', '').replace('-', '').lower()
                            if key_norm == subj_norm or key_norm in subj_norm or subj_norm in key_norm:
                                label = label_dict[key]
                                matched = True
                                break
                        if not matched:
                            # For age/sex evaluation, allow subjects without labels
                            # For disease diagnosis, skip subjects without labels
                            if labels_file is None:
                                label = None  # Allow loading without label
                            else:
                                continue  # Skip subjects without labels for disease diagnosis
                else:
                    # No labels_file provided (age/sex evaluation) - allow all subjects
                    label = None
                
                # Get age and sex if available
                age = self.age_dict.get(subject_id, None) if hasattr(self, 'age_dict') else None
                sex = self.sex_dict.get(subject_id, None) if hasattr(self, 'sex_dict') else None
                
                # Create segments
                segments = self._create_segments(timeseries)
                
                for segment in segments:
                    self.sequences.append(segment)
                    self.fc_matrices.append(fc_matrix.copy())
                    self.metadata.append({
                        'subject': subject_id,
                        'file': str(bold_file),
                        'label': float(label) if label is not None else None,
                        'labels': float(label) if label is not None else None,
                        'age': age,
                        'sex': sex
                    })
                    
            except Exception as e:
                print(f"  ⚠️  Error processing {bold_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary
        if self.sequences:
            labels = [m['label'] for m in self.metadata if m['label'] is not None]
            if labels:
                if self.single_window_per_subject:
                    print(f"✓ Created {len(self.sequences)} sequence segments (single window per subject)")
                else:
                    print(f"✓ Created {len(self.sequences)} sequence segments (sliding window)")
                print(f"  Class distribution: CN-group={labels.count(0.0)}, AD-group={labels.count(1.0)}")
            else:
                if self.single_window_per_subject:
                    print(f"✓ Created {len(self.sequences)} sequence segments (single window per subject, no labels - age/sex evaluation mode)")
                else:
                    print(f"✓ Created {len(self.sequences)} sequence segments (sliding window, no labels - age/sex evaluation mode)")
        else:
            print("⚠️  No sequences loaded!")
            print(f"  Check that NPZ files exist in {data_dir}")
            print(f"  Expected format: BOLD*.npz files with timeseries data")
    
    def _create_segments(self, timeseries: np.ndarray) -> list:
        """Create overlapping segments from timeseries or single window per subject"""
        segments = []
        T, n_roi = timeseries.shape
        
        if self.single_window_per_subject:
            # Single window mode: take only the first window
            if T < self.sequence_length:
                # Pad if too short
                pad_length = self.sequence_length - T
                timeseries = np.pad(timeseries, ((0, pad_length), (0, 0)), mode='constant')
                segments.append(timeseries)
            else:
                # Take first window
                segments.append(timeseries[:self.sequence_length])
        else:
            # Sliding window mode (default)
            if T < self.sequence_length:
                # Pad if too short
                pad_length = self.sequence_length - T
                timeseries = np.pad(timeseries, ((0, pad_length), (0, 0)), mode='constant')
                segments.append(timeseries)
            else:
                # Create overlapping segments
                for start in range(0, T - self.sequence_length + 1, self.stride):
                    segments.append(timeseries[start:start + self.sequence_length])
        
        return segments
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        fc = self.fc_matrices[idx]
        meta = self.metadata[idx].copy()
        
        # Normalize per sequence
        if self.normalize:
            seq_mean = np.mean(seq, axis=0, keepdims=True)
            seq_std = np.std(seq, axis=0, keepdims=True) + 1e-8
            seq = (seq - seq_mean) / seq_std
        
        # Convert to tensor
        sequence = torch.from_numpy(seq.astype(np.float32))
        meta['fc_matrix'] = torch.from_numpy(fc.astype(np.float32))
        
        return sequence, meta


class ADNISCFCCsvDataset(Dataset):
    """
    ADNI SC-FC dataset for atlas-specific CSV BOLD time series:
        .../ADNI-SC-FC/<atlas>/BOLD/sub-941S7106_..._time_series.csv

    Subject IDs are mapped to metadata 'Subject' (e.g., "941_S_7106").
    Labels come from metadata.csv with DX column (CN/SMC/EMCI vs LMCI/AD).
    """
    LABEL_MAPPING = ADNIDataset.LABEL_MAPPING

    def __init__(
        self,
        data_dir: str,
        labels_file: str = None,
        label_column: str = 'DX',
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        max_files: int = None,
        single_window_per_subject: bool = False,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.labels_file = labels_file
        self.label_column = label_column
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.normalize_per_sequence = True
        self.single_window_per_subject = single_window_per_subject

        self.sequences = []
        self.fc_matrices = []
        self.metadata = []

        if not self.data_dir.exists():
            print(f"✗ ADNI SC-FC data directory does not exist: {self.data_dir}")
            return

        # Load labels from metadata.csv
        label_dict = {}
        if labels_file and Path(labels_file).exists():
            try:
                df = pd.read_csv(labels_file)
                print(f"✓ Loaded ADNI metadata with {len(df)} rows from {labels_file}")

                # Subject column: match patterns like "941_S_7106"
                subject_col = None
                for col in ['Subject', 'subject', 'PTID', 'SUB_ID', 'RID']:
                    if col in df.columns:
                        subject_col = col
                        break

                if subject_col and label_column in df.columns:
                    for _, row in df.iterrows():
                        subj = str(row[subject_col]).strip()
                        if not subj or subj == 'nan':
                            continue
                        dx = str(row[label_column]).strip()
                        if dx in self.LABEL_MAPPING:
                            label_dict[subj] = self.LABEL_MAPPING[dx]
                    print(f"✓ ADNI SC-FC labels for {len(label_dict)} subjects")
                else:
                    print(f"  ⚠️  Could not find Subject/DX columns in ADNI metadata.")
                    print(f"      Columns: {list(df.columns)[:10]}")
            except Exception as e:
                print(f"  ⚠️  Failed to load ADNI labels from {labels_file}: {e}")

        # Helper to extract ADNI subject ID from SC-FC filename
        def extract_subject_id_from_csv(fname: Path) -> str:
            """
            Example: sub-941S7106_task-rest_...csv -> '941_S_7106'
            """
            import re
            m = re.search(r'sub-(\d{3})S(\d{4})', fname.name)
            if m:
                return f"{m.group(1)}_S_{m.group(2)}"
            # Fallback: try any pattern like 941S7106
            m2 = re.search(r'(\d{3})S(\d{4})', fname.name)
            if m2:
                return f"{m2.group(1)}_S_{m2.group(2)}"
            return fname.stem

        all_files = sorted(self.data_dir.glob("sub-*_time_series.csv"))
        if not all_files:
            all_files = sorted(self.data_dir.glob("sub-*.csv"))
        if max_files is not None:
            all_files = all_files[:max_files]

        print(f"Found {len(all_files)} ADNI SC-FC CSV files in {self.data_dir}")

        n_skipped_label = 0

        for csv_file in tqdm(all_files, desc="Loading ADNI SC-FC subjects"):
            try:
                subj_id = extract_subject_id_from_csv(csv_file)
                label = label_dict.get(subj_id, None)
                if label is None:
                    n_skipped_label += 1
                    continue

                df_ts = pd.read_csv(csv_file)
                ts = df_ts.values.astype(np.float32)
                if ts.ndim != 2:
                    continue
                if ts.shape[1] > ts.shape[0]:
                    ts = ts.T

                if np.isnan(ts).any():
                    ts = np.nan_to_num(ts, nan=0.0)

                T = ts.shape[0]
                if T < self.sequence_length:
                    continue

                starts = [0] if self.single_window_per_subject else range(0, T - self.sequence_length + 1, self.stride)
                for start in starts:
                    end = start + self.sequence_length
                    if end > T:
                        break
                    window = ts[start:end].copy()
                    if self.normalize:
                        mean = window.mean(axis=0, keepdims=True)
                        std = window.std(axis=0, keepdims=True) + 1e-6
                        window = (window - mean) / std

                    fc = np.corrcoef(window.T)
                    if np.isnan(fc).any():
                        fc = np.nan_to_num(fc, nan=0.0)

                    self.sequences.append(window.astype(np.float32))
                    self.fc_matrices.append(fc.astype(np.float32))
                    self.metadata.append({
                        'subject_id': subj_id,
                        'label': int(label),
                    })
            except Exception as e:
                print(f"  ⚠️  Error loading {csv_file.name}: {e}")

        print(f"✓ ADNI SC-FC: {len(self.sequences)} windows with labels")
        if n_skipped_label > 0:
            print(f"  ⚠️  Skipped {n_skipped_label} files without matching labels")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        fc = self.fc_matrices[idx]
        meta = self.metadata[idx].copy()
        meta['fc_matrix'] = torch.from_numpy(fc.astype(np.float32))
        sequence = torch.from_numpy(seq.astype(np.float32))
        return sequence, meta


# Register ADNI dataset (default NPZ-based implementation)
DATASET_REGISTRY['ADNI'] = ADNIDataset


# Create ABIDE dataset class
class ABIDEDataset:
    """
    ABIDE dataset loader.
    Assumes structure: data/abide/sub-controlXXXXX/ or data/abide/sub-patientXXXXX/
    Files: sub-control50030_AAL116_features_timeseries.mat, sub-control50030_AAL116_correlation_matrix.mat
    
    Labels from metadata.csv with Group column ("Autism" or "Control")
    Metadata format: Image Data ID,Subject,Group,Sex,Age,Visit,Modality,Description
    """
    
    def __init__(
        self,
        data_dir: str,
        labels_file: str = None,
        label_column: str = 'Group',  # Default to 'Group' to match metadata format
        sequence_length: int = 100,
        stride: int = 50,
        normalize: bool = True,
        max_files: int = None,
        single_window_per_subject: bool = False,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.labels_file = labels_file
        self.label_column = label_column
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.normalize_per_sequence = True
        self.single_window_per_subject = single_window_per_subject
        
        self.sequences = []
        self.fc_matrices = []
        self.metadata = []
        
        # Load labels from metadata.csv
        label_dict = {}
        age_dict = {}
        sex_dict = {}
        if labels_file and Path(labels_file).exists():
            try:
                df = pd.read_csv(labels_file)
                print(f"✓ Loaded metadata with {len(df)} rows from {labels_file}")
                
                # Find subject column (should be 'Subject')
                subject_col = None
                for col in ['Subject', 'SUB_ID', 'subject', 'SUBJ_ID', 'subject_id']:
                    if col in df.columns:
                        subject_col = col
                        break
                
                # Find label column (try 'Group' first, then 'DX_GROUP')
                label_col = None
                for col in [label_column, 'Group', 'DX_GROUP', 'diagnosis', 'Diagnosis']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if subject_col and label_col:
                    for _, row in df.iterrows():
                        subj = str(row[subject_col]).strip()
                        if not subj or subj == 'nan':
                            continue
                        
                        # Extract label
                        dx_val = row[label_col]
                        if isinstance(dx_val, str):
                            # Text labels: "Autism" -> 1, "Control" -> 0
                            dx_lower = dx_val.lower().strip()
                            if 'autism' in dx_lower or 'asd' in dx_lower:
                                label = 1
                            elif 'control' in dx_lower or 'typical' in dx_lower or 'td' in dx_lower:
                                label = 0
                            else:
                                continue  # Skip unknown labels
                        else:
                            # Numeric labels: 1=Autism, 2=Control (or 0=Control, 1=Autism)
                            try:
                                dx = int(float(dx_val))
                                # Handle both encodings: 1=Autism,2=Control OR 0=Control,1=Autism
                                if dx == 1:
                                    # Check if it's Autism or Control by looking at other rows
                                    label = 1  # Assume 1=Autism (common encoding)
                                elif dx == 2:
                                    label = 0  # 2=Control
                                elif dx == 0:
                                    label = 0  # 0=Control
                                else:
                                    continue
                            except:
                                continue
                        
                        label_dict[subj] = label
                        
                        # Extract age if available
                        if 'Age' in df.columns:
                            age_val = row.get('Age')
                            if pd.notna(age_val):
                                try:
                                    age_dict[subj] = float(age_val)
                                except:
                                    pass
                        
                        # Extract sex if available
                        if 'Sex' in df.columns:
                            sex_val = row.get('Sex')
                            if pd.notna(sex_val):
                                try:
                                    if isinstance(sex_val, str):
                                        sex_dict[subj] = 1 if sex_val.upper().strip() in ['M', 'MALE', '1'] else 0
                                    else:
                                        sex_dict[subj] = int(float(sex_val))
                                except:
                                    pass
                    
                    print(f"✓ Loaded labels for {len(label_dict)} subjects (Autism=1, Control=0)")
                    if age_dict:
                        print(f"  Age data: {len(age_dict)} subjects")
                    if sex_dict:
                        print(f"  Sex data: {len(sex_dict)} subjects")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load labels: {e}")
                import traceback
                traceback.print_exc()
        
        # Store age and sex dicts
        self.age_dict = age_dict
        self.sex_dict = sex_dict
        
        # Find data files - check nested structure (sub-controlXXXXX/ or sub-patientXXXXX/)
        subject_folders = sorted(list(self.data_dir.glob("sub-*")))
        
        if not subject_folders:
            print(f"⚠️  No subject folders found in {data_dir}")
            print(f"  Expected structure: {data_dir}/sub-control50030/AAL116_*.mat")
            return
        
        print(f"Found {len(subject_folders)} subject folders")
        
        # Load data from each subject folder
        data_files_info = []  # List of (file_path, subject_id, is_timeseries, is_fc)
        
        for folder in subject_folders:
            # Extract subject ID from folder name (e.g., "sub-control50030" -> "50030")
            folder_name = folder.name
            import re
            # Extract numeric ID from folder name
            id_match = re.search(r'(\d+)', folder_name)
            if id_match:
                subject_id = id_match.group(1)
            else:
                # Fallback: remove 'sub-' prefix
                subject_id = folder_name.replace('sub-', '').replace('control', '').replace('patient', '').strip()
            
            # Look for AAL116 files
            # Timeseries file: *features_timeseries*.mat or *timeseries*.mat
            ts_files = list(folder.glob("*AAL116*features_timeseries*.mat")) + \
                      list(folder.glob("*AAL116*timeseries*.mat")) + \
                      list(folder.glob("*features_timeseries*.mat")) + \
                      list(folder.glob("*timeseries*.mat"))
            
            # FC/correlation file: *correlation*.mat or *corr*.mat
            fc_files = list(folder.glob("*AAL116*correlation*.mat")) + \
                      list(folder.glob("*AAL116*corr*.mat")) + \
                      list(folder.glob("*correlation*.mat")) + \
                      list(folder.glob("*corr*.mat"))
            
            # Also check for CSV/TXT files
            csv_files = list(folder.glob("*.csv")) + list(folder.glob("*.txt"))
            
            # Prioritize MAT files, but also include CSV/TXT
            if ts_files:
                data_files_info.append((ts_files[0], subject_id, True, False))
            elif csv_files:
                data_files_info.append((csv_files[0], subject_id, True, False))
            
            if fc_files:
                data_files_info.append((fc_files[0], subject_id, False, True))
        
        if max_files:
            data_files_info = data_files_info[:max_files]
        
        print(f"Found {len(data_files_info)} data files to process")
        
        # Group files by subject
        subject_files = {}
        for file_path, subject_id, is_ts, is_fc in data_files_info:
            if subject_id not in subject_files:
                subject_files[subject_id] = {'timeseries': None, 'fc': None}
            if is_ts:
                subject_files[subject_id]['timeseries'] = file_path
            if is_fc:
                subject_files[subject_id]['fc'] = file_path
        
        # Load timeseries and FC for each subject
        for subject_id, files in tqdm(subject_files.items(), desc="Loading ABIDE subjects"):
            try:
                # Get label
                label = label_dict.get(subject_id)
                if label is None:
                    # Try partial match
                    for key in label_dict:
                        if key == subject_id or key in subject_id or subject_id in key:
                            label = label_dict[key]
                            break
                
                if label is None:
                    # Auto-extract from folder name if no labels file
                    if not label_dict:
                        # Extract from folder name: sub-control* -> 0, sub-patient* -> 1
                        folder_name = files['timeseries'].parent.name if files['timeseries'] else ''
                        if 'control' in folder_name.lower():
                            label = 0
                        elif 'patient' in folder_name.lower() or 'autism' in folder_name.lower():
                            label = 1
                        else:
                            continue  # Skip if can't determine label
                    else:
                        continue  # Skip if no label found and we have labels file
                
                # Load timeseries
                timeseries = None
                if files['timeseries']:
                    file_path = files['timeseries']
                    if file_path.suffix == '.mat':
                        # Load MATLAB file
                        mat_data = sio.loadmat(str(file_path))
                        # Try common keys
                        for key in ['timeseries', 'BOLD', 'bold', 'data', 'ts', 'features_timeseries']:
                            if key in mat_data:
                                timeseries = np.array(mat_data[key], dtype=np.float32)
                                break
                        if timeseries is None:
                            # Get first non-metadata key
                            for key in mat_data.keys():
                                if not key.startswith('__'):
                                    timeseries = np.array(mat_data[key], dtype=np.float32)
                                    break
                    else:
                        # Load CSV/TXT
                        try:
                            timeseries = pd.read_csv(file_path, header=0).values.astype(np.float32)
                        except:
                            timeseries = np.loadtxt(file_path, delimiter=',').astype(np.float32)
                
                if timeseries is None or timeseries.shape[0] < 10:
                    continue
                
                # Ensure shape is [T, N_ROI]
                if timeseries.shape[1] > timeseries.shape[0]:
                    timeseries = timeseries.T
                
                # Handle NaN
                timeseries = np.nan_to_num(timeseries, nan=0.0)
                
                # Load FC matrix
                fc_matrix = None
                if files['fc']:
                    fc_path = files['fc']
                    if fc_path.suffix == '.mat':
                        fc_data = sio.loadmat(str(fc_path))
                        for key in ['correlation_matrix', 'fc', 'FC', 'connectivity', 'corr', 'R']:
                            if key in fc_data:
                                fc_matrix = np.array(fc_data[key], dtype=np.float32)
                                break
                        if fc_matrix is None:
                            for key in fc_data.keys():
                                if not key.startswith('__'):
                                    fc_matrix = np.array(fc_data[key], dtype=np.float32)
                                    break
                    else:
                        try:
                            fc_matrix = pd.read_csv(fc_path, header=0).values.astype(np.float32)
                        except:
                            fc_matrix = np.loadtxt(fc_path, delimiter=',').astype(np.float32)
                
                # Compute FC if not loaded
                if fc_matrix is None:
                    fc_matrix = np.corrcoef(timeseries.T)
                    fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
                
                # Create segments
                segments = self._create_segments(timeseries)
                
                # Get age and sex
                age = self.age_dict.get(subject_id, None) if hasattr(self, 'age_dict') else None
                sex = self.sex_dict.get(subject_id, None) if hasattr(self, 'sex_dict') else None
                
                for segment in segments:
                    self.sequences.append(segment)
                    self.fc_matrices.append(fc_matrix.copy())
                    self.metadata.append({
                        'subject': subject_id,
                        'file': str(files['timeseries'] or files['fc']),
                        'label': float(label),
                        'labels': float(label),
                        'age': age,
                        'sex': sex
                    })
                    
            except Exception as e:
                print(f"  ⚠️  Error loading subject {subject_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if self.sequences:
            labels = [m['label'] for m in self.metadata]
            if self.single_window_per_subject:
                print(f"✓ Created {len(self.sequences)} segments from {len(subject_files)} subjects (single window per subject)")
            else:
                print(f"✓ Created {len(self.sequences)} segments from {len(subject_files)} subjects (sliding window)")
            print(f"  Autism={labels.count(1.0)}, Control={labels.count(0.0)}")
        else:
            print("⚠️  No sequences loaded! Check data structure and subject ID matching.")
    
    def _create_segments(self, timeseries: np.ndarray) -> list:
        """Create sliding window segments or single window per subject"""
        segments = []
        T, n_roi = timeseries.shape
        
        if self.single_window_per_subject:
            # Single window mode: take only the first window
            if T < self.sequence_length:
                pad_length = self.sequence_length - T
                timeseries = np.pad(timeseries, ((0, pad_length), (0, 0)), mode='constant')
                segments.append(timeseries)
            else:
                # Take first window
                segments.append(timeseries[:self.sequence_length])
        else:
            # Sliding window mode (default)
            if T < self.sequence_length:
                pad_length = self.sequence_length - T
                timeseries = np.pad(timeseries, ((0, pad_length), (0, 0)), mode='constant')
                segments.append(timeseries)
            else:
                for start in range(0, T - self.sequence_length + 1, self.stride):
                    segments.append(timeseries[start:start + self.sequence_length])
        
        return segments
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        fc = self.fc_matrices[idx]
        meta = self.metadata[idx].copy()
        
        if self.normalize:
            seq_mean = np.mean(seq, axis=0, keepdims=True)
            seq_std = np.std(seq, axis=0, keepdims=True) + 1e-8
            seq = (seq - seq_mean) / seq_std
        
        sequence = torch.from_numpy(seq.astype(np.float32))
        meta['fc_matrix'] = torch.from_numpy(fc.astype(np.float32))
        
        return sequence, meta


# Register ABIDE dataset
DATASET_REGISTRY['ABIDE'] = ABIDEDataset

# Register ABCD dataset (for age/sex evaluation)
try:
    from utils.data_loader import ABCDBOLDWithFCDataset
    DATASET_REGISTRY['ABCD'] = ABCDBOLDWithFCDataset
except ImportError:
    pass  # ABCD dataset not available


def custom_collate_fn(batch, expected_roi=116):
    """
    Custom collate function that handles None values in metadata and variable ROI counts.
    
    Args:
        batch: List of (sequence, metadata) tuples
        expected_roi: Expected ROI count (default 116 for AAL_116). All sequences will be padded to this.
    """
    # Handle empty batch
    if len(batch) == 0:
        # Return empty tensors with correct shape
        empty_seq = torch.zeros(1, expected_roi, dtype=torch.float32)
        empty_meta = {'subject': 'unknown', 'label': None, 'age': None, 'sex': None}
        return empty_seq, [empty_meta]
    
    sequences = []
    metadata_list = []
    
    # Use expected ROI count (model's expected ROI) instead of batch max
    # This ensures compatibility with the pretrained model
    target_roi = expected_roi
    max_seq_len = max(item[0].shape[0] for item in batch if item[0] is not None and item[0].numel() > 0)
    
    # Handle case where all sequences are empty
    if max_seq_len == 0:
        max_seq_len = 100  # Default sequence length
    
    for seq, meta in batch:
        # Pad sequence if needed to match expected ROI and max sequence length
        T, n_roi = seq.shape
        
        # Pad ROI dimension to expected ROI count (crop if larger)
        if n_roi < target_roi:
            padding_roi = torch.zeros(T, target_roi - n_roi, dtype=seq.dtype)
            seq = torch.cat([seq, padding_roi], dim=1)
        elif n_roi > target_roi:
            # Crop if larger than expected (shouldn't happen often)
            seq = seq[:, :target_roi]
        
        # Pad time dimension if needed (shouldn't happen often, but handle it)
        if T < max_seq_len:
            padding_time = torch.zeros(max_seq_len - T, seq.shape[1], dtype=seq.dtype)
            seq = torch.cat([seq, padding_time], dim=0)
        
        sequences.append(seq)
        
        # Handle FC matrix - pad/crop to expected ROI count
        if 'fc_matrix' in meta and meta['fc_matrix'] is not None:
            fc = meta['fc_matrix']
            if fc.dim() == 2:
                # Single FC matrix [ROI, ROI]
                if fc.shape[0] != target_roi or fc.shape[1] != target_roi:
                    # Pad or crop FC matrix to target ROI
                    if fc.shape[0] < target_roi:
                        pad_rows = torch.zeros(target_roi - fc.shape[0], fc.shape[1], dtype=fc.dtype)
                        fc = torch.cat([fc, pad_rows], dim=0)
                    elif fc.shape[0] > target_roi:
                        fc = fc[:target_roi, :]
                    
                    if fc.shape[1] < target_roi:
                        pad_cols = torch.zeros(fc.shape[0], target_roi - fc.shape[1], dtype=fc.dtype)
                        fc = torch.cat([fc, pad_cols], dim=1)
                    elif fc.shape[1] > target_roi:
                        fc = fc[:, :target_roi]
                    
                    meta['fc_matrix'] = fc
        
        # Convert None to NaN for age/sex to avoid collation errors
        clean_meta = meta.copy()
        if 'age' in clean_meta and clean_meta['age'] is None:
            clean_meta['age'] = float('nan')
        if 'sex' in clean_meta and clean_meta['sex'] is None:
            clean_meta['sex'] = float('nan')
        if 'label' in clean_meta and clean_meta['label'] is None:
            clean_meta['label'] = float('nan')
        
        metadata_list.append(clean_meta)
    
    # Stack sequences
    batch_x = torch.stack(sequences)
    
    # Return metadata as list (don't try to collate dicts with None/NaN)
    return batch_x, metadata_list


def extract_features_with_demographics(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    feature_type: str = 'z_concat',
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features along with age and sex from metadata.
    
    Returns:
        features: [N, d_feat] array of features
        labels: [N] array of diagnosis labels
        ages: [N] array of ages (NaN if not available)
        sexes: [N] array of sexes (NaN if not available)
        subjects: [N] array of subject IDs
    """
    features = []
    labels = []
    ages = []
    sexes = []
    subjects = []
    
    print(f"Extracting {feature_type} features with demographics...")
    
    model.eval()
    with torch.no_grad():
        from torch.utils.data import DataLoader
        from functools import partial
        # Use custom collate function to handle None values and variable ROI counts
        # Get expected ROI from model config (default 116 for AAL_116)
        # Try to get from model's n_roi attribute or config
        expected_roi = 116  # Default for AAL_116
        if hasattr(model, 'n_roi'):
            expected_roi = model.n_roi
        elif hasattr(model, 'config') and isinstance(model.config, dict) and 'n_roi' in model.config:
            expected_roi = model.config['n_roi']
        elif hasattr(model, 'spatial_encoder') and hasattr(model.spatial_encoder, 'n_roi'):
            expected_roi = model.spatial_encoder.n_roi
        collate_fn = partial(custom_collate_fn, expected_roi=expected_roi)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        for batch_x, batch_metadata in tqdm(dataloader, desc="Extracting features"):
            # batch_metadata is a list of dicts from custom_collate_fn
            # Skip empty batches
            if batch_x.shape[0] == 0:
                print(f"  ⚠️  Warning: Skipping empty batch")
                continue
            
            batch_x = batch_x.to(device)
            
            # Validate batch size matches metadata
            if len(batch_metadata) != batch_x.shape[0]:
                print(f"  ⚠️  Warning: Batch size mismatch: batch_x.shape[0]={batch_x.shape[0]}, len(batch_metadata)={len(batch_metadata)}")
                print(f"     Skipping this batch to avoid shape mismatches")
                continue
            
            # Get FC matrices if available (handle variable sizes)
            fc_matrices = []
            batch_roi_count = batch_x.shape[2]  # [B, T, ROI]
            
            for meta in batch_metadata:
                if 'fc_matrix' in meta:
                    fc = meta['fc_matrix']
                    # Ensure FC matrix matches sequence ROI count
                    if fc.shape[0] != batch_roi_count or fc.shape[1] != batch_roi_count:
                        # Pad or crop to match
                        if fc.shape[0] < batch_roi_count:
                            # Pad
                            pad_rows = torch.zeros(batch_roi_count - fc.shape[0], fc.shape[1], dtype=fc.dtype, device=fc.device)
                            fc = torch.cat([fc, pad_rows], dim=0)
                        elif fc.shape[0] > batch_roi_count:
                            # Crop
                            fc = fc[:batch_roi_count, :]
                        
                        if fc.shape[1] < batch_roi_count:
                            # Pad columns
                            pad_cols = torch.zeros(fc.shape[0], batch_roi_count - fc.shape[1], dtype=fc.dtype, device=fc.device)
                            fc = torch.cat([fc, pad_cols], dim=1)
                        elif fc.shape[1] > batch_roi_count:
                            # Crop columns
                            fc = fc[:, :batch_roi_count]
                    
                    fc_matrices.append(fc)
                else:
                    # Create identity matrix if no FC provided
                    fc_matrices.append(torch.eye(batch_roi_count, dtype=torch.float32))
            
            # Stack FC matrices (all should be same size now)
            if len(fc_matrices) > 0:
                try:
                    fc_matrix = torch.stack(fc_matrices).to(device)
                except RuntimeError as e:
                    # If stacking fails, use identity matrices
                    print(f"  ⚠️  Warning: Could not stack FC matrices: {e}")
                    print(f"     Using identity matrices instead")
                    fc_matrix = torch.eye(batch_roi_count, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_x.shape[0], 1, 1)
            else:
                fc_matrix = None
            
            # Forward pass
            outputs = model(batch_x, task_vectors=None, fc_matrix=fc_matrix, return_uncertainty=False)
            z = outputs['z_latent']  # [B, T, d_latent]
            
            # Extract features for each sample in batch
            for b in range(z.shape[0]):
                if feature_type == 'z_mean':
                    feat = z[b].mean(dim=0).cpu().numpy()
                elif feature_type == 'z_std':
                    feat = z[b].std(dim=0).cpu().numpy()
                elif feature_type == 'z_concat':
                    z_mean = z[b].mean(dim=0).cpu().numpy()
                    z_std = z[b].std(dim=0).cpu().numpy()
                    feat = np.concatenate([z_mean, z_std])
                elif feature_type == 'z_all':
                    z_mean = z[b].mean(dim=0).cpu().numpy()
                    z_std = z[b].std(dim=0).cpu().numpy()
                    z_max = z[b].max(dim=0)[0].cpu().numpy()
                    z_min = z[b].min(dim=0)[0].cpu().numpy()
                    feat = np.concatenate([z_mean, z_std, z_max, z_min])
                else:
                    feat = z[b].mean(dim=0).cpu().numpy()
                
                features.append(feat)
                
                # Extract metadata (batch_metadata is a list of dicts)
                meta = batch_metadata[b]
                subjects.append(meta.get('subject', 'unknown'))
                
                # Handle NaN values (convert back to None for consistency)
                label_val = meta.get('label', None)
                if label_val is not None and (isinstance(label_val, float) and np.isnan(label_val)):
                    label_val = None
                labels.append(label_val)
                
                age_val = meta.get('age', None)
                if age_val is not None and (isinstance(age_val, float) and np.isnan(age_val)):
                    age_val = None
                ages.append(age_val)
                
                sex_val = meta.get('sex', None)
                if sex_val is not None and (isinstance(sex_val, float) and np.isnan(sex_val)):
                    sex_val = None
                sexes.append(sex_val)
    
    features = np.array(features)
    subjects = np.array(subjects)
    
    # Convert labels, ages, sexes to arrays, handling None values
    labels_arr = np.array([l if l is not None else np.nan for l in labels])
    ages_arr = np.array([a if a is not None else np.nan for a in ages])
    sexes_arr = np.array([s if s is not None else np.nan for s in sexes])
    
    return features, labels_arr, ages_arr, sexes_arr, subjects


def evaluate_age_prediction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mlp_params: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate age prediction (regression task)
    
    Returns:
        results: Dictionary with all metrics
    """
    print(f"\nTraining age predictor...")
    
    # Validate array lengths match
    if len(X_train) != len(y_train):
        print(f"  ⚠️  Warning: X_train length ({len(X_train)}) != y_train length ({len(y_train)})")
        print(f"     Truncating to match...")
        min_len = min(len(X_train), len(y_train))
        X_train = X_train[:min_len]
        y_train = y_train[:min_len]
    
    if len(X_test) != len(y_test):
        print(f"  ⚠️  Warning: X_test length ({len(X_test)}) != y_test length ({len(y_test)})")
        print(f"     Truncating to match...")
        min_len = min(len(X_test), len(y_test))
        X_test = X_test[:min_len]
        y_test = y_test[:min_len]
    
    # Filter out missing ages
    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test)
    
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("  ⚠️  No valid age data available")
        return {
            'test_mse': None,
            'test_mae': None,
            'test_r2': None,
            'test_pearson': None,
        }
    
    X_train_valid = X_train[train_mask]
    y_train_valid = y_train[train_mask]
    X_test_valid = X_test[test_mask]
    y_test_valid = y_test[test_mask]
    
    # Additional validation: ensure we have valid data after filtering
    if len(X_train_valid) == 0 or len(X_test_valid) == 0:
        print("  ⚠️  No valid data after filtering")
        return {
            'test_mse': None,
            'test_mae': None,
            'test_r2': None,
            'test_pearson': None,
        }
    
    print(f"  Train: {len(X_train_valid)} samples, Test: {len(X_test_valid)} samples")
    print(f"  Age range - Train: [{y_train_valid.min():.1f}, {y_train_valid.max():.1f}] years")
    print(f"  Age range - Test: [{y_test_valid.min():.1f}, {y_test_valid.max():.1f}] years")
    
    # Use FrozenMLP if available, otherwise Ridge regression
    if mlp_params is not None:
        try:
            from eval.frozen_mlp_head import train_regression_head
            # Additional check before calling MLP training
            if len(X_test_valid) == 0:
                print("  ⚠️  Test set is empty, skipping MLP training")
                raise ValueError("Empty test set")
            results = train_regression_head(
                X_train_valid, y_train_valid, X_test_valid, y_test_valid,
                **mlp_params
            )
            return results
        except (ImportError, ValueError) as e:
            if isinstance(e, ValueError) and "Empty test set" in str(e):
                # Return early with None values
                return {
                    'test_mse': None,
                    'test_mae': None,
                    'test_r2': None,
                    'test_pearson': None,
                }
            pass
    
    # Fallback to Ridge regression
    from sklearn.preprocessing import StandardScaler
    import warnings
    from scipy.linalg import LinAlgWarning
    
    # Suppress ill-conditioned matrix warnings for Ridge regression
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=LinAlgWarning)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_valid)
        X_test_scaled = scaler.transform(X_test_valid)
        
        # Hyperparameter search with more stable solver
        from sklearn.model_selection import GridSearchCV
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
        # Use 'lsqr' or 'sag' solver for better numerical stability with ill-conditioned matrices
        ridge = Ridge(solver='lsqr', max_iter=1000)
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_valid)
    
    model = grid_search.best_estimator_
    print(f"  Best alpha: {grid_search.best_params_['alpha']}")
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_mse = mean_squared_error(y_train_valid, y_pred_train)
    test_mse = mean_squared_error(y_test_valid, y_pred_test)
    train_mae = mean_absolute_error(y_train_valid, y_pred_train)
    test_mae = mean_absolute_error(y_test_valid, y_pred_test)
    train_r2 = r2_score(y_train_valid, y_pred_train)
    test_r2 = r2_score(y_test_valid, y_pred_test)
    train_pearson, _ = pearsonr(y_train_valid, y_pred_train)
    test_pearson, _ = pearsonr(y_test_valid, y_pred_test)
    
    results = {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_pearson': float(train_pearson),
        'test_pearson': float(test_pearson),
    }
    
    return results


def evaluate_sex_prediction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mlp_params: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate sex prediction (binary classification)
    
    Returns:
        results: Dictionary with all metrics
    """
    print(f"\nTraining sex classifier...")
    
    # Validate array lengths match
    if len(X_train) != len(y_train):
        print(f"  ⚠️  Warning: X_train length ({len(X_train)}) != y_train length ({len(y_train)})")
        print(f"     Truncating to match...")
        min_len = min(len(X_train), len(y_train))
        X_train = X_train[:min_len]
        y_train = y_train[:min_len]
    
    if len(X_test) != len(y_test):
        print(f"  ⚠️  Warning: X_test length ({len(X_test)}) != y_test length ({len(y_test)})")
        print(f"     Truncating to match...")
        min_len = min(len(X_test), len(y_test))
        X_test = X_test[:min_len]
        y_test = y_test[:min_len]
    
    # Filter out missing sexes
    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test)
    
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("  ⚠️  No valid sex data available")
        return {
            'test_accuracy': None,
            'test_f1': None,
            'test_auc': None,
        }
    
    X_train_valid = X_train[train_mask]
    y_train_valid = y_train[train_mask].astype(int)
    X_test_valid = X_test[test_mask]
    y_test_valid = y_test[test_mask].astype(int)
    
    # Additional validation: ensure we have valid data after filtering
    if len(X_train_valid) == 0 or len(X_test_valid) == 0:
        print("  ⚠️  No valid data after filtering")
        return {
            'train_accuracy': None,
            'test_accuracy': None,
            'train_f1': None,
            'test_f1': None,
            'test_auc': None,
        }
    
    print(f"  Train: {len(X_train_valid)} samples, Test: {len(X_test_valid)} samples")
    print(f"  Sex distribution - Train: Female={np.sum(y_train_valid==0)}, Male={np.sum(y_train_valid==1)}")
    print(f"  Sex distribution - Test: Female={np.sum(y_test_valid==0)}, Male={np.sum(y_test_valid==1)}")
    
    # Check if we have both classes
    unique_train = np.unique(y_train_valid)
    unique_test = np.unique(y_test_valid)
    if len(unique_train) < 2 or len(unique_test) < 2:
        print("  ⚠️  Warning: Only one class present in data. Cannot train binary classifier.")
        return {
            'train_accuracy': None,
            'test_accuracy': None,
            'train_f1': None,
            'test_f1': None,
            'test_auc': None,
        }
    
    # Use FrozenMLP if available, otherwise Logistic Regression
    if mlp_params is not None:
        try:
            from eval.frozen_mlp_head import train_classification_head
            # Additional check before calling MLP training
            if len(X_test_valid) == 0:
                print("  ⚠️  Test set is empty, skipping MLP training")
                raise ValueError("Empty test set")
            results = train_classification_head(
                X_train_valid, y_train_valid, X_test_valid, y_test_valid,
                **mlp_params
            )
            return results
        except (ImportError, ValueError) as e:
            if isinstance(e, ValueError) and "Empty test set" in str(e):
                # Return early with None values
                return {
                    'train_accuracy': None,
                    'test_accuracy': None,
                    'train_f1': None,
                    'test_f1': None,
                    'test_auc': None,
                }
            pass
    
    # Fallback to Logistic Regression
    from sklearn.preprocessing import StandardScaler
    import warnings
    from scipy.linalg import LinAlgWarning
    
    # Suppress ill-conditioned matrix warnings for LogisticRegression
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=LinAlgWarning)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_valid)
        X_test_scaled = scaler.transform(X_test_valid)
        
        # Hyperparameter search with more stable solver
        from sklearn.model_selection import GridSearchCV
        param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
        # Use 'lbfgs' or 'liblinear' solver for better numerical stability
        logreg = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            solver='lbfgs',  # More stable than default 'lbfgs' for ill-conditioned problems
            tol=1e-4  # Slightly relaxed tolerance
        )
        # Use accuracy scoring if only one class, otherwise use roc_auc
        scoring = 'accuracy' if len(np.unique(y_train_valid)) < 2 else 'roc_auc'
        grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_valid)
    
    model = grid_search.best_estimator_
    print(f"  Best C: {grid_search.best_params_['C']}")
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train_valid, y_pred_train)
    test_acc = accuracy_score(y_test_valid, y_pred_test)
    train_f1 = f1_score(y_train_valid, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test_valid, y_pred_test, zero_division=0)
    
    # ROC AUC (only if both classes present)
    try:
        if len(np.unique(y_test_valid)) >= 2:
            test_auc = roc_auc_score(y_test_valid, y_pred_proba)
        else:
            test_auc = None
    except:
        test_auc = None
    
    results = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_f1': float(train_f1),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc) if test_auc is not None else None,
    }
    
    return results


def evaluate_disease_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_name: str = 'LogisticRegression',
    cv_folds: int = 5,
    class_names: Optional[List[str]] = None,
    mlp_params: Optional[Dict] = None,
    test_subjects: Optional[np.ndarray] = None,
    per_subject_eval: bool = False,
) -> Tuple[Dict, object]:
    """
    Evaluate disease classification with comprehensive metrics
    
    Returns:
        results: Dictionary with all metrics
        classifier: Trained model
    """
    print(f"\nTraining {classifier_name}...")
    
    # Validate inputs
    if len(X_train) == 0 or len(X_test) == 0:
        print("  ⚠️  Warning: Empty train or test set")
        return {
            'train_accuracy': None,
            'test_accuracy': None,
            'train_balanced_accuracy': None,
            'test_balanced_accuracy': None,
            'test_f1_macro': None,
            'test_f1_weighted': None,
            'test_precision': None,
            'test_recall': None,
            'test_auc': None,
            'classifier': None,
            'n_classes': 0,
            'class_names': [],
        }, None
    
    # Select classifier
    import warnings
    from scipy.linalg import LinAlgWarning
    
    # Suppress ill-conditioned matrix warnings during classifier training
    if classifier_name == 'LogisticRegression':
        clf = LogisticRegression(
            max_iter=1000, random_state=42, 
            class_weight='balanced', C=1.0,
            solver='lbfgs',  # More stable solver
            tol=1e-4  # Slightly relaxed tolerance
        )
    elif classifier_name == 'RandomForest':
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, class_weight='balanced', n_jobs=-1
        )
    elif classifier_name == 'GradientBoosting':
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42
        )
    elif classifier_name == 'SVM':
        clf = SVC(
            kernel='rbf', C=1.0, gamma='scale',
            random_state=42, class_weight='balanced',
            probability=True
        )
    elif classifier_name == 'MLP':
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation='relu',
            max_iter=1000, random_state=42, early_stopping=True
        )
    elif classifier_name.lower() in {'frozenmlp', 'frozen_mlp'}:
        mlp_params = mlp_params or {}
        unique_classes = np.unique(y_train)
        inferred_names = [str(cls) for cls in unique_classes]
        # Additional check before calling MLP training
        if len(X_test) == 0:
            print("  ⚠️  Test set is empty, skipping MLP training")
            return {
                'train_accuracy': None,
                'test_accuracy': None,
                'train_balanced_accuracy': None,
                'test_balanced_accuracy': None,
                'test_f1_macro': None,
                'test_f1_weighted': None,
                'test_precision': None,
                'test_recall': None,
                'test_auc': None,
                'classifier': 'FrozenMLP',
                'n_classes': int(len(unique_classes)),
                'class_names': class_names if class_names else inferred_names,
            }, None
        try:
            mlp_results = train_classification_head(
                X_train, y_train, X_test, y_test, **mlp_params
            )
            mlp_results.update({
                'classifier': 'FrozenMLP',
                'n_classes': int(len(unique_classes)),
                'class_names': class_names if class_names else inferred_names,
            })
            return mlp_results, None
        except (ValueError, RuntimeError) as e:
            if "batch_size" in str(e) or "empty" in str(e).lower():
                print(f"  ⚠️  Error during MLP training: {e}")
                return {
                    'train_accuracy': None,
                    'test_accuracy': None,
                    'train_balanced_accuracy': None,
                    'test_balanced_accuracy': None,
                    'test_f1_macro': None,
                    'test_f1_weighted': None,
                    'test_precision': None,
                    'test_recall': None,
                    'test_auc': None,
                    'classifier': 'FrozenMLP',
                    'n_classes': int(len(unique_classes)),
                    'class_names': class_names if class_names else inferred_names,
                }, None
            raise
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    # Train with warnings suppressed for ill-conditioned matrices
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=LinAlgWarning)
            clf.fit(X_train, y_train)
    except Exception as e:
        print(f"  ✗ Error training {classifier_name}: {e}")
        print(f"     Returning None results...")
        return {
            'classifier': classifier_name,
            'train_accuracy': None,
            'test_accuracy': None,
            'train_balanced_accuracy': None,
            'test_balanced_accuracy': None,
            'test_f1_macro': None,
            'test_f1_weighted': None,
            'test_precision': None,
            'test_recall': None,
            'test_auc': None,
            'n_classes': int(len(np.unique(y_train))),
            'class_names': class_names if class_names else [f"Class_{i}" for i in range(len(np.unique(y_train)))],
        }, None
    
    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    # Probabilities (for ROC/AUC)
    if hasattr(clf, 'predict_proba'):
        y_prob_train = clf.predict_proba(X_train)
        y_prob_test = clf.predict_proba(X_test)
    else:
        y_prob_train = None
        y_prob_test = None
    
    # Per-subject aggregation if requested
    if per_subject_eval and test_subjects is not None:
        from scipy import stats
        
        # Aggregate predictions per subject
        y_pred_test_agg, test_subjects_unique, y_prob_test_agg = aggregate_predictions_per_subject(
            y_pred_test, test_subjects, task_type='classification', probabilities=y_prob_test
        )
        
        # Aggregate labels per subject (use mode for classification)
        y_test_agg = []
        for subj in test_subjects_unique:
            subj_mask = test_subjects == subj
            subj_labels = y_test[subj_mask]
            mode_result = stats.mode(subj_labels, keepdims=True)
            y_test_agg.append(mode_result.mode[0])
        y_test_agg = np.array(y_test_agg)
        
        # Use aggregated predictions and labels for metrics
        y_pred_test = y_pred_test_agg
        y_test = y_test_agg
        y_prob_test = y_prob_test_agg
        
        print(f"  ✓ Aggregated to {len(y_test_agg)} subjects (from {len(test_subjects)} samples)")
    
    # Basic metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
    
    # Multi-class metrics
    n_classes = len(np.unique(y_train))
    average_mode = 'binary' if n_classes == 2 else 'macro'
    
    test_f1_macro = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    
    # Per-class F1 scores
    per_class_f1 = f1_score(y_test, y_pred_test, average=None, zero_division=0)
    
    # ROC AUC (if probabilities available)
    if y_prob_test is not None:
        if n_classes == 2:
            # Binary classification
            test_auc = roc_auc_score(y_test, y_prob_test[:, 1])
        else:
            # Multi-class (one-vs-rest)
            try:
                test_auc = roc_auc_score(
                    y_test, y_prob_test, 
                    average='macro', multi_class='ovr'
                )
            except:
                test_auc = None
    else:
        test_auc = None
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(
            clf, X_train, y_train, cv=cv,
            scoring='balanced_accuracy', n_jobs=-1
        )
        cv_acc = cv_scores.mean()
        cv_acc_std = cv_scores.std()
    except Exception as e:
        print(f"  ⚠️  Warning: Cross-validation failed for {classifier_name}: {e}")
        print(f"     Continuing without CV scores...")
        cv_acc = None
        cv_acc_std = None
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Classification report
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    # Validate class_names length matches n_classes
    if len(class_names) != n_classes:
        print(f"  ⚠️  Warning in evaluate_disease_classification: class_names length ({len(class_names)}) doesn't match n_classes ({n_classes})")
        print(f"     Expected class_names: {class_names}, but data has {n_classes} classes")
        print(f"     Using generic class names instead.")
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    clf_report = classification_report(
        y_test, y_pred_test, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Sensitivity and specificity (for binary classification)
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = None
        specificity = None
    
    results = {
        'classifier': classifier_name,
        'n_classes': int(n_classes),
        'class_names': class_names,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_balanced_accuracy': float(train_balanced_acc),
        'test_balanced_accuracy': float(test_balanced_acc),
        'test_f1_macro': float(test_f1_macro),
        'test_f1_weighted': float(test_f1_weighted),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'per_class_f1': [float(x) for x in per_class_f1],
        'test_auc': float(test_auc) if test_auc is not None else None,
        'cv_balanced_accuracy': float(cv_acc) if cv_acc is not None else None,
        'cv_balanced_accuracy_std': float(cv_acc_std) if cv_acc_std is not None else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': clf_report,
        'sensitivity': float(sensitivity) if sensitivity is not None else None,
        'specificity': float(specificity) if specificity is not None else None,
        'predictions': {
            'test_true': y_test.tolist(),
            'test_pred': y_pred_test.tolist(),
            'test_probs': y_prob_test.tolist() if y_prob_test is not None else None
        }
    }
    
    return results, clf


def plot_disease_diagnosis(
    results: Dict,
    output_dir: Path,
    dataset_name: str = ''
):
    """Create comprehensive visualization of disease diagnosis results"""
    y_true = np.array(results['predictions']['test_true'])
    y_pred = np.array(results['predictions']['test_pred'])
    y_probs = results['predictions']['test_probs']
    class_names = results['class_names']
    n_classes = results['n_classes']
    
    # Create figure
    if n_classes == 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
    
    # 1. Confusion Matrix
    cm = np.array(results['confusion_matrix'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    ax = axes[0] if n_classes == 2 else axes[0]
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    acc = results.get("test_accuracy")
    bal_acc = results.get("test_balanced_accuracy")
    acc_str = f"{acc:.3f}" if acc is not None else "N/A"
    bal_acc_str = f"{bal_acc:.3f}" if bal_acc is not None else "N/A"
    ax.set_title(f'{dataset_name} Confusion Matrix\n'
                 f'Accuracy={acc_str}, '
                 f'Balanced Acc={bal_acc_str}',
                 fontsize=14)
    
    # 2. ROC Curve (if binary or probabilities available)
    if n_classes == 2 and y_probs is not None:
        y_probs = np.array(y_probs)
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        ax = axes[1]
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    elif n_classes > 2 and y_probs is not None:
        # Multi-class ROC (one-vs-rest)
        y_probs = np.array(y_probs)
        ax = axes[1]
        
        # Binarize labels for OvR
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        
        for i, class_name in enumerate(class_names):
            if i < y_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC={roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Multi-class ROC Curves', fontsize=14)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 3. Per-class F1 scores
    ax = axes[2] if n_classes == 2 else axes[2]
    per_class_f1 = results['per_class_f1']
    x_pos = np.arange(len(class_names))
    bars = ax.bar(x_pos, per_class_f1, color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores', fontsize=14)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, f1 in zip(bars, per_class_f1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Class distribution (for multi-class)
    if n_classes > 2:
        ax = axes[3]
        unique, counts = np.unique(y_true, return_counts=True)
        # Convert to int for indexing (handle numpy float64)
        class_labels = [class_names[int(i)] for i in unique]
        ax.bar(class_labels, counts, color='coral', alpha=0.7)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Test Set Class Distribution', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for i, (label, count) in enumerate(zip(class_labels, counts)):
            ax.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / f'diagnosis_{dataset_name}_{results["classifier"]}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Disease Diagnosis Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pre-trained model checkpoint (default: auto-find best checkpoint)')
    
    # Single dataset mode
    parser.add_argument('--dataset', type=str, default=None,
                       choices=list(DATASET_REGISTRY.keys()),
                       help='Single dataset name')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset BOLD data directory')
    parser.add_argument('--labels_file', type=str, default=None,
                       help='Path to labels CSV file')
    parser.add_argument('--label_column', type=str, default='diagnosis',
                       help='Column name for labels in CSV (default: diagnosis)')
    parser.add_argument('--subject_column', type=str, default='Subject',
                       help='Column name for subject IDs in CSV (default: Subject)')
    
    # Multi-dataset mode
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Multiple dataset names')
    parser.add_argument('--data_dirs', type=str, nargs='+', default=None,
                       help='Paths to dataset BOLD data directories')
    parser.add_argument('--labels_files', type=str, nargs='+', default=None,
                       help='Paths to labels CSV files')
    
    # Class names (optional)
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='Class names for visualization (e.g., Control ASD)')
    
    # Data processing
    parser.add_argument('--sequence_length', type=int, default=None,
                       help='Sequence length (default: from checkpoint)')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride for sliding window (default: from checkpoint)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files per dataset (for testing)')
    parser.add_argument('--single_window_per_subject', action='store_true',
                       help='Use only one window per subject (no sliding window). Results in N samples = N subjects instead of multiple samples per subject.')
    
    # Feature extraction
    parser.add_argument('--feature_type', type=str, default='z_concat',
                       choices=['z_mean', 'z_std', 'z_concat', 'z_all', 'cognitive'],
                       help='Type of features to extract')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    
    # Evaluation
    parser.add_argument('--classifiers', type=str, nargs='+',
                       default=['FrozenMLP', 'LogisticRegression', 'RandomForest', 'SVM'],
                       help='Classifiers to evaluate (FrozenMLP = fine-tuning with frozen backbone)')
    parser.add_argument('--use_linear_probes', action='store_true',
                       help='Use linear probes instead of fine-tuning (overrides --classifiers default)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (subject-level split)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--n_runs', type=int, default=30,
                       help='Number of evaluation runs with different random seeds (default: 30 for fair comparison. Uses seeds 42, 43, 44, ..., 42+n_runs-1)')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base random seed for evaluation runs (default: 42). Seeds will be base_seed, base_seed+1, ..., base_seed+n_runs-1')
    parser.add_argument('--per_subject_eval', action='store_true',
                       help='Aggregate predictions per subject before computing metrics (one prediction per subject)')
    parser.add_argument('--fc_mode', type=str, default='dynamic', choices=['static', 'dynamic', 'hybrid'],
                       help='FC computation mode: static (one FC per subject), dynamic (FC per window), hybrid (weighted combination). Note: Custom datasets (PPMI, TAOWU, etc.) load pre-computed FC from files, so this may not apply.')
    parser.add_argument('--hybrid_weight', type=float, default=0.3,
                       help='Weight for dynamic FC in hybrid mode (0.0 = fully static, 1.0 = fully dynamic)')
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[256, 128],
                       help='Hidden dimensions for Frozen MLP head (default: [256, 128] for better regularization)')
    parser.add_argument('--mlp_dropout', type=float, default=0.3,
                       help='Dropout for Frozen MLP head (default: 0.3 for better regularization)')
    parser.add_argument('--mlp_lr', type=float, default=5e-4,
                       help='Learning rate for Frozen MLP head (default: 5e-4 for more stable training)')
    parser.add_argument('--mlp_weight_decay', type=float, default=1e-3,
                       help='Weight decay for Frozen MLP head (default: 1e-3 for stronger regularization)')
    parser.add_argument('--mlp_epochs', type=int, default=200,
                       help='Max epochs for Frozen MLP head')
    parser.add_argument('--mlp_batch_size', type=int, default=64,
                       help='Batch size for Frozen MLP head')
    parser.add_argument('--mlp_patience', type=int, default=15,
                       help='Early stopping patience for Frozen MLP head (default: 15 to prevent overfitting)')
    parser.add_argument('--mlp_use_robust_norm', action='store_true', default=True,
                       help='Use robust normalization (RobustScaler with median/IQR) for MLP head (default: True, matches paper methodology)')
    parser.add_argument('--mlp_use_standard_norm', action='store_true', default=False,
                       help='Use standard normalization (StandardScaler with mean/std) instead of robust normalization')
    
    # Output
    parser.add_argument('--output', type=str, default='outputs/disease_diagnosis.json',
                       help='Output JSON file path')
    parser.add_argument('--plot_dir', type=str, default='outputs/diagnosis_plots',
                       help='Directory for plots')
    
    args = parser.parse_args()
    
    # Adjust default classifiers if user wants linear probes only
    if args.use_linear_probes and 'FrozenMLP' in args.classifiers:
        # Remove FrozenMLP and use only linear probes
        args.classifiers = [c for c in args.classifiers if c != 'FrozenMLP']
        if len(args.classifiers) == 0:
            args.classifiers = ['LogisticRegression', 'RandomForest', 'SVM']
        print("Using linear probes only (FrozenMLP excluded)")
    
    # Validate arguments
    if args.dataset is None and args.datasets is None:
        parser.error("Must provide either --dataset or --datasets")
    
    # Auto-detect default paths for common datasets (using relative paths like test_abide.py/test_ppmi.py)
    default_paths = {
        'ABIDE': {
            'data_dir': '../data/abide',
            'labels_file': None,  # Optional - can auto-extract from folder names
        },
        'PPMI': {
            'data_dir': '../data/ppmi',
            'labels_file': '../data/ppmi/labels.csv',  # PPMI labels with age/sex
        },
        'MATAI': {
            'data_dir': '../data/matai',
            'labels_file': None,  # Optional - can auto-extract from folder names
        },
        'NEUROCON': {
            'data_dir': '../data/neurocon',
            'labels_file': None,  # Optional - can auto-extract from folder names
        },
        'TAOWU': {
            'data_dir': '../data/taowu',
            'labels_file': None,  # Optional - can auto-extract from folder names
        }
    }
    
    # Setup dataset list
    if args.dataset:
        datasets = [args.dataset]
        
        # Auto-fill missing arguments with defaults
        dataset_name = args.dataset.upper()
        if dataset_name in default_paths:
            data_dir = args.data_dir if args.data_dir else default_paths[dataset_name]['data_dir']
            # labels_file can be None (auto-extract from folder names)
            labels_file = args.labels_file if args.labels_file is not None else default_paths[dataset_name]['labels_file']
            
            if args.data_dir is None:
                print(f"Auto-selected data_dir: {data_dir}")
            if args.labels_file is None:
                if labels_file:
                    print(f"Auto-selected labels_file: {labels_file}")
                else:
                    print("No labels_file - will auto-extract labels from folder names")
        else:
            data_dir = args.data_dir
            labels_file = args.labels_file
        
        # Use provided label column (default: 'diagnosis')
        label_column = args.label_column
        
        data_dirs = [data_dir]
        labels_files = [labels_file]
        label_columns = [label_column]
    else:
        datasets = args.datasets
        data_dirs = args.data_dirs
        labels_files = args.labels_files
        # Convert 'None' strings to None (for compatibility with wrapper scripts)
        if labels_files:
            labels_files = [None if f == 'None' or f == '' else f for f in labels_files]
        
        # Use provided label column for all datasets
        label_columns = [args.label_column] * len(datasets)
        
        if not all([
            len(datasets) == len(data_dirs),
            len(datasets) == len(labels_files),
        ]):
            parser.error("All dataset arguments must have same length")
    
    # Auto-find checkpoint if not provided
    if args.checkpoint is None:
        if find_best_checkpoint is None:
            parser.error("--checkpoint is required when manage_checkpoints.py is not available")
        print("Auto-finding best checkpoint...")
        # Try to infer dataset type from first dataset
        dataset_for_checkpoint = datasets[0].upper() if datasets else 'UKB'
        try:
            checkpoint_path = find_best_checkpoint(results_dir='results', dataset=dataset_for_checkpoint, quiet=True)
        except Exception as e:
            parser.error(f"Auto-find failed: {e}. Please specify --checkpoint manually")
        if checkpoint_path:
            args.checkpoint = checkpoint_path
            print(f"✓ Auto-selected best checkpoint: {args.checkpoint}\n")
        else:
            parser.error("Could not auto-find checkpoint. Please specify --checkpoint")
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\n✗ Error: Checkpoint not found: {args.checkpoint}")
        print("  Please update --checkpoint path or run:")
        dataset_for_checkpoint = datasets[0].upper() if datasets else 'UKB'
        print(f"    python scripts/manage_checkpoints.py list --dataset {dataset_for_checkpoint}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Determine normalization type (robust by default to match paper)
    use_robust_norm = args.mlp_use_robust_norm and not args.mlp_use_standard_norm
    
    mlp_params = {
        'hidden_dims': args.mlp_hidden_dims,
        'dropout': args.mlp_dropout,
        'lr': args.mlp_lr,
        'weight_decay': args.mlp_weight_decay,
        'batch_size': args.mlp_batch_size,
        'epochs': args.mlp_epochs,
        'patience': args.mlp_patience,
        'device': device,
        'use_robust_norm': use_robust_norm,  # Robust normalization (median/IQR) as per paper
    }
    
    # Load model
    print("\n" + "="*70)
    print("DISEASE DIAGNOSIS EVALUATION")
    print("="*70)
    
    # Indicate evaluation method
    if 'FrozenMLP' in args.classifiers:
        print("\n✓ Using fine-tuning method (Frozen MLP head with frozen backbone)")
        print("  Methodology matches paper:")
        print("    - Freeze pretrained backbone, train only MLP head")
        print("    - MLP: Linear → LayerNorm → GELU → Dropout (p=0.1) → ... → Linear")
        print("    - Loss: MSE for regression, CrossEntropy for classification")
        print("    - Optimizer: Adam with ReduceLROnPlateau scheduler")
        print(f"    - Normalization: {'Robust (median/IQR)' if use_robust_norm else 'Standard (mean/std)'}")
        print("  This is more robust than linear probes for cross-dataset evaluation")
    else:
        print("\n✓ Using linear probe method (scikit-learn classifiers)")
        print("  For better performance, consider using --classifiers FrozenMLP")
    
    print("\nLoading pre-trained model...")
    model, config = load_model(args.checkpoint, device)
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # Get sequence parameters
    seq_length = args.sequence_length if args.sequence_length else config['sequence_length']
    stride = args.stride if args.stride else config['stride']
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate data directories and labels files (like test_abide.py/test_ppmi.py)
    # Note: label_column is always 'diagnosis' (hardcoded for simplicity)
    for i, (dataset_name, data_dir, labels_file, label_col) in enumerate(zip(
        datasets, data_dirs, labels_files, label_columns
    )):
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"\n✗ Error: Data directory not found: {data_dir}")
            print(f"  Please update --data_dir path for {dataset_name}")
            return
        
        # Check labels file if provided
        if labels_file and not Path(labels_file).exists():
            print(f"\n⚠️  Warning: Labels file not found: {labels_file}")
            print(f"  {dataset_name} can auto-extract labels from folder names if labels_file is not provided")
            print(f"  Continuing without labels file...")
            labels_files[i] = None  # Update to None for auto-extraction
    
    # Evaluate on each dataset
    all_results = {
        'checkpoint': args.checkpoint,
        'feature_type': args.feature_type,
        'classifiers': args.classifiers,
        'datasets': {}
    }
    
    for dataset_name, data_dir, labels_file, label_col in zip(
        datasets, data_dirs, labels_files, label_columns
    ):
        print("\n" + "="*70)
        print(f"DATASET: {dataset_name}")
        print("="*70)
        
        # Load dataset
        print(f"\nLoading {dataset_name} dataset...")
        if labels_file and Path(labels_file).exists():
            print(f"Using labels file: {labels_file}")
        else:
            print("No labels file provided - will auto-extract labels from folder names")
            if dataset_name == 'ABIDE':
                print("  (sub-control* = Control, sub-patient* = ASD)")
            elif dataset_name == 'PPMI':
                print("  (2-class: sub-control* = Control (0), sub-patient*/sub-prodromal*/sub-swedd* = Patient (1))")
            elif dataset_name == 'TAOWU':
                print("  (sub-control(patient)xxxx = Control, sub-patient(patient)xxxx = Patient)")
            elif dataset_name == 'NEUROCON':
                print("  (sub-control(patient)xxxx = Control, sub-patient(patient)xxxx = Patient)")
            labels_file = None  # Ensure it's None for auto-extraction
        
        # For TAOWU and NEUROCON, verify folder structure exists
        if dataset_name in ['TAOWU', 'NEUROCON']:
            data_path = Path(data_dir)
            # Check for sub-control(patient)xxxx or sub-patient(patient)xxxx folders
            subject_folders = []
            for item in data_path.iterdir():
                if item.is_dir():
                    name_lower = item.name.lower()
                    if 'sub-control' in name_lower or 'sub-patient' in name_lower:
                        subject_folders.append(item)
            
            if len(subject_folders) == 0:
                print(f"\n✗ Error: No subject folders found in {data_dir}")
                print(f"  {dataset_name} expects structure:")
                print(f"    {data_dir}/")
                print(f"    ├── sub-control(patient)xxxx/")
                print(f"    │   ├── AAL116_timeseires.mat")
                print(f"    │   └── AAL116_correlation_matrix.mat")
                print(f"    ├── sub-patient(patient)xxxx/")
                print(f"    │   ├── AAL116_timeseires.mat")
                print(f"    │   └── AAL116_correlation_matrix.mat")
                print(f"    └── ...")
                print(f"  Skipping {dataset_name}...")
                continue
            else:
                # Check if required files exist in at least one folder
                found_files = 0
                for sub_folder in subject_folders[:5]:  # Check first 5
                    ts_files = list(sub_folder.glob('AAL116_timeseires*.mat')) + list(sub_folder.glob('AAL116_timeseries*.mat'))
                    corr_files = list(sub_folder.glob('AAL116_correlation*.mat'))
                    if ts_files and corr_files:
                        found_files += 1
                
                if found_files == 0:
                    print(f"⚠️  Warning: No AAL116_timeseires.mat or AAL116_correlation_matrix.mat found in sample folders")
                    print(f"  Will try to load with flexible pattern matching...")
                else:
                    print(f"✓ Found {len(subject_folders)} subject folders")
                    print(f"  Sample folders contain required files: {found_files}/{min(5, len(subject_folders))}")
        
        # For PPMI, verify nested folder structure exists (legacy layout only).
        # Skip this check for SC-FC CSV layouts (PPMI-SC-FC/...), which use a different loader.
        if dataset_name == 'PPMI' and 'PPMI-SC-FC' not in str(data_dir):
            data_path = Path(data_dir)
            # Check for sub-control/ and sub-patient/ folders
            category_folders = []
            for item in data_path.iterdir():
                if item.is_dir():
                    name_lower = item.name.lower()
                    if name_lower == 'sub-control' or name_lower == 'sub-patient' or 'sub-control' in name_lower or 'sub-patient' in name_lower:
                        category_folders.append(item)
            
            if len(category_folders) == 0:
                print(f"\n✗ Error: No category folders (sub-control/, sub-swedd/, sub-prodromal/, or sub-patient/) found in {data_dir}")
                print(f"  PPMI expects nested structure (4-class classification):")
                print(f"    {data_dir}/")
                print(f"    ├── sub-control/      (Control, label=0)")
                print(f"    ├── sub-swedd/         (SWEDD, label=1)")
                print(f"    ├── sub-prodromal/    (Prodromal, label=2)")
                print(f"    ├── sub-patient/      (PD, label=3)")
                print(f"    │   ├── patient1234/")
                print(f"    │   │   ├── *features_timeseries*.mat")
                print(f"    │   │   └── *correlation*.mat")
                print(f"    │   └── ...")
                print(f"    ├── sub-prodromal/    (Prodromal, label=2)")
                print(f"    └── sub-swedd/         (SWEDD, label=3)")
                print(f"  Skipping {dataset_name}...")
                continue
            else:
                # Count patient folders inside category folders
                total_patients = 0
                for cat_folder in category_folders:
                    patients = [d for d in cat_folder.iterdir() if d.is_dir()]
                    total_patients += len(patients)
                print(f"✓ Found {len(category_folders)} category folders ({[f.name for f in category_folders]})")
                print(f"  Total patient folders: {total_patients}")
        
        dataset_class = DATASET_REGISTRY[dataset_name]

        # Override dataset class for SC-FC atlas-specific CSV layouts
        data_dir_str = str(data_dir)
        if dataset_name == 'PPMI' and 'PPMI-SC-FC' in data_dir_str:
            print(f"Using PPMISCFCCsvDataset for PPMI SC-FC data: {data_dir}")
            dataset_class = PPMISCFCCsvDataset
        if dataset_name == 'ADNI' and 'ADNI-SC-FC' in data_dir_str:
            print(f"Using ADNISCFCCsvDataset for ADNI SC-FC data: {data_dir}")
            dataset_class = ADNISCFCCsvDataset
        
        # Build dataset arguments
        # Special handling for ABCD which uses bold_dir instead of data_dir
        if dataset_name == 'ABCD':
            dataset_kwargs = {
                'bold_dir': data_dir,  # ABCD uses bold_dir, not data_dir
                'fc_dir': None,  # Will auto-detect
                'sequence_length': seq_length,
                'stride': stride,
                'normalize': True,
                'task_filter': ['REST'],  # ABCD-specific
                'max_files': args.max_files,
                'single_window_per_subject': args.single_window_per_subject
            }
        else:
            dataset_kwargs = {
                'data_dir': data_dir,
                'labels_file': labels_file,  # None = auto-extract from folder names
                'label_column': label_col,
                'sequence_length': seq_length,
                'stride': stride,
                'normalize': True,
                'max_files': args.max_files,
                'single_window_per_subject': args.single_window_per_subject
            }
        
        # Add FC mode parameters if dataset supports them
        # (Most custom datasets load pre-computed FC, but pass for consistency)
        if hasattr(dataset_class, '__init__'):
            import inspect
            sig = inspect.signature(dataset_class.__init__)
            if 'fc_mode' in sig.parameters:
                dataset_kwargs['fc_mode'] = args.fc_mode
                dataset_kwargs['hybrid_weight'] = args.hybrid_weight
        
        dataset = dataset_class(**dataset_kwargs)
        if args.single_window_per_subject:
            print(f"✓ Loaded {len(dataset)} samples (single window per subject mode)")
        else:
            print(f"✓ Loaded {len(dataset)} samples (sliding window mode)")
        
        # Extract features with demographics (age, sex) - always fresh (no caching)
        print("Extracting features from model...")
        try:
            features, labels, ages, sexes, subjects = extract_features_with_demographics(
                model, dataset, device, args.feature_type, args.batch_size
            )
            has_age = np.sum(~np.isnan(ages)) > 0
            has_sex = np.sum(~np.isnan(sexes)) > 0
        except Exception as e:
            print(f"  ⚠️  Could not extract demographics: {e}")
            print("  Falling back to standard feature extraction...")
            features, labels, subjects = extract_features(
                model, dataset, device, args.feature_type, args.batch_size
            )
            ages = None
            sexes = None
            has_age = False
            has_sex = False
        
        # Validate we have data
        # Handle labels properly: check if None, empty, or has length
        labels_len = 0
        if labels is None:
            labels_len = 0
        elif isinstance(labels, np.ndarray):
            labels_len = labels.size
        elif isinstance(labels, (list, tuple)):
            labels_len = len(labels)
        
        if len(features) == 0 or labels_len == 0:
            print(f"\n✗ Error: No samples with labels found for {dataset_name}!")
            print(f"  This usually means:")
            print(f"    1. Data directory structure doesn't match expected format")
            print(f"    2. No subject folders found in {data_dir}")
            if dataset_name == 'PPMI':
                print(f"    3. PPMI expects nested structure: data/ppmi/sub-*/timeseries.mat")
                print(f"    4. Check that subject folders contain 'sub' in their name")
            elif dataset_name == 'ABIDE':
                print(f"    3. ABIDE expects AAL116 files in nested or flat structure")
            print(f"  Skipping {dataset_name}...")
            continue
        
        print(f"\n✓ Extracted features:")
        print(f"  Shape: {features.shape}")
        print(f"  Unique subjects: {len(np.unique(subjects))}")
        
        # Check for valid labels (not all NaN)
        valid_labels_mask = ~np.isnan(labels) if np.issubdtype(labels.dtype, np.floating) else np.ones(len(labels), dtype=bool)
        n_valid_labels = valid_labels_mask.sum()
        
        if n_valid_labels == 0:
            print(f"\n✗ Error: No valid labels found for {dataset_name}!")
            print(f"  All {len(labels)} labels are NaN or missing.")
            print(f"  This usually means:")
            print(f"    1. Subject IDs in labels file don't match folder names")
            print(f"    2. Labels file format is incorrect")
            print(f"    3. Labels file path is wrong: {labels_file}")
            print(f"  Skipping {dataset_name}...")
            continue
        
        # Filter out samples with NaN labels
        if n_valid_labels < len(labels):
            print(f"  ⚠️  Filtering out {len(labels) - n_valid_labels} samples with NaN labels")
            features = features[valid_labels_mask]
            labels = labels[valid_labels_mask]
            subjects = subjects[valid_labels_mask]
            if ages is not None:
                ages = ages[valid_labels_mask]
            if sexes is not None:
                sexes = sexes[valid_labels_mask]
            print(f"  Remaining samples: {len(labels)}")
        
        # Calculate n_classes and class_names AFTER filtering
        n_classes = len(np.unique(labels))
        
        # Set default class names if not provided
        if args.class_names is None:
            if dataset_name == 'PPMI':
                # PPMI is 4-class: Control, SWEDD, Prodromal, PD
                if n_classes == 4:
                    class_names = ['Control', 'SWEDD', 'Prodromal', 'PD']
                else:
                    # Fallback: use generic names if number of classes doesn't match
                    class_names = [f"Class_{i}" for i in range(n_classes)]
                    print(f"  ⚠️  Warning: PPMI expected 4 classes but found {n_classes}. Using generic class names.")
            elif dataset_name == 'ABIDE':
                class_names = ['Control', 'ASD']
            elif dataset_name == 'ADNI':
                class_names = ['CN-group', 'AD-group']
            else:
                # Generic class names
                class_names = [f"Class_{i}" for i in range(n_classes)]
        else:
            class_names = args.class_names
        
        # Validate class_names length matches n_classes
        if len(class_names) != n_classes:
            print(f"  ⚠️  Warning: class_names length ({len(class_names)}) doesn't match n_classes ({n_classes})")
            print(f"     Using generic class names instead.")
            class_names = [f"Class_{i}" for i in range(n_classes)]
        
        if len(np.unique(labels)) > 0:
            unique_labels = np.unique(labels)
            print(f"  Classes: {unique_labels}")
            print(f"  Class distribution:")
            for label in unique_labels:
                count = np.sum(labels == label)
                # Handle NaN safely (shouldn't happen after filtering, but be safe)
                try:
                    if np.isnan(label):
                        continue
                except (TypeError, ValueError):
                    pass
                label_idx = int(label) if isinstance(label, (int, float, np.integer, np.floating)) else 0
                class_name = class_names[label_idx] if 0 <= label_idx < len(class_names) else f"Class_{label}"
                print(f"    {class_name} (label={label}): {count} ({count/len(labels)*100:.1f}%)")
        else:
            print(f"  ⚠️  Warning: No valid labels found")
        
        # Split by subject - get masks for reuse with ages/sexes
        X_train, X_test, y_train, y_test = split_by_subject(
            features, labels, subjects, test_size=args.test_size, random_state=42
        )
        
        # Get train/test masks from the split for reuse with ages/sexes
        # We need to reconstruct the masks from the split results. If too few
        # subjects are available for a stratified split, fall back to a simple
        # split or mark as insufficient data.
        from sklearn.model_selection import train_test_split
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        if n_subjects < 2:
            print(f"\n✗ Error: Too few subjects for train/test split (n_subjects={n_subjects}).")
            print("  Skipping classification for this dataset due to insufficient data.")
            results[dataset_name] = {
                'status': 'error',
                'error': f'Too few subjects for disease evaluation (n_subjects={n_subjects})',
                'n_samples': int(len(features)),
                'n_subjects': int(n_subjects),
            }
            continue
        # Get subject labels for stratification
        from scipy import stats
        subject_labels = []
        for subj in unique_subjects:
            subj_mask = subjects == subj
            subj_labels = labels[subj_mask]
            mode_result = stats.mode(subj_labels, keepdims=True)
            subject_labels.append(mode_result.mode[0])
        subject_labels = np.array(subject_labels)
        
        # Recreate the same split
        try:
            train_subjects_arr, test_subjects_arr = train_test_split(
                unique_subjects,
                test_size=min(args.test_size, 0.5),
                random_state=42,
                stratify=subject_labels if len(np.unique(subject_labels)) > 1 else None
            )
        except (ValueError, TypeError):
            train_subjects_arr, test_subjects_arr = train_test_split(
                unique_subjects,
                test_size=min(args.test_size, 0.5),
                random_state=42
            )
        
        train_subjects_set = set(train_subjects_arr)
        test_subjects_set = set(test_subjects_arr)
        train_mask = np.array([s in train_subjects_set for s in subjects])
        test_mask = np.array([s in test_subjects_set for s in subjects])
        
        dataset_results = {
            'n_samples': int(len(features)),
            'n_subjects': int(len(np.unique(subjects))),
            'n_classes': int(n_classes),
            'class_distribution': {
                str(cls): int(np.sum(labels == cls)) 
                for cls in np.unique(labels)
            },
            'n_runs': args.n_runs,
            'classifiers': {},
            'has_age': has_age if 'has_age' in locals() else False,
            'has_sex': has_sex if 'has_sex' in locals() else False,
            'single_window_per_subject': args.single_window_per_subject,
        }
        
        def aggregate_metrics(run_results_list, metric_keys):
            """Aggregate numeric metrics across runs"""
            def is_valid_value(val):
                """Check if value is valid (not None, not NaN, and scalar)"""
                if val is None:
                    return False
                # Skip arrays - only aggregate scalar values
                if isinstance(val, np.ndarray):
                    return False
                # Check for NaN (works for scalar floats)
                try:
                    if np.isnan(val):
                        return False
                except (TypeError, ValueError):
                    # Not a numeric type that can be NaN
                    return False
                return True
            
            aggregated = {}
            for key in metric_keys:
                values = [r[key] for r in run_results_list if is_valid_value(r.get(key))]
                if len(values) > 0:
                    aggregated[key] = float(np.mean(values))
                    aggregated[f'{key}_std'] = float(np.std(values))
                else:
                    aggregated[key] = None
                    aggregated[f'{key}_std'] = None
            return aggregated
        
        print("\n" + "-"*70)
        print(f"EVALUATION ({args.n_runs} runs with different random seeds)")
        print("-"*70)
        if dataset_name == 'PPMI':
            print(f"  Performing 4-class classification: {class_names} (Control, SWEDD, Prodromal, PD)")
        
        # Store results from all runs
        all_runs_results = {clf_name: [] for clf_name in args.classifiers}
        all_runs_age_results = []
        all_runs_sex_results = []
        
        # Run evaluation N times with different random seeds
        # IMPORTANT: For fair comparison, use same seed sequence (base_seed, base_seed+1, ..., base_seed+n_runs-1)
        # Default: seeds 42, 43, 44, ..., 42+n_runs-1
        print(f"\n{'='*70}")
        print(f"EVALUATION SETTINGS FOR FAIR COMPARISON:")
        print(f"  - Number of runs: {args.n_runs}")
        print(f"  - Base seed: {args.base_seed}")
        print(f"  - Seed sequence: {args.base_seed}, {args.base_seed+1}, ..., {args.base_seed+args.n_runs-1}")
        print(f"  - Feature type: {args.feature_type}")
        print(f"  - FC mode: {args.fc_mode}")
        print(f"  - Hybrid weight: {args.hybrid_weight}")
        print(f"{'='*70}")
        
        for run_idx in range(args.n_runs):
            random_seed = args.base_seed + run_idx  # Consistent seed sequence for fair comparison
            print(f"\n{'='*70}")
            print(f"RUN {run_idx + 1}/{args.n_runs} (random_seed={random_seed})")
            print(f"{'='*70}")
            
            # Split by subject with different random seed for each run
            X_train, X_test, y_train, y_test = split_by_subject(
                features, labels, subjects, test_size=args.test_size, random_state=random_seed
            )
            
            # Recreate masks for this run
            unique_subjects = np.unique(subjects)
            subject_labels = []
            for subj in unique_subjects:
                subj_mask = subjects == subj
                subj_labels = labels[subj_mask]
                mode_result = stats.mode(subj_labels, keepdims=True)
                subject_labels.append(mode_result.mode[0])
            subject_labels = np.array(subject_labels)
            
            if len(unique_subjects) < 2:
                raise ValueError(f"Too few subjects for train/test split (n_subjects={len(unique_subjects)})")
            try:
                train_subjects_arr, test_subjects_arr = train_test_split(
                    unique_subjects,
                    test_size=min(args.test_size, 0.5),
                    random_state=random_seed,
                    stratify=subject_labels if len(np.unique(subject_labels)) > 1 else None
                )
            except (ValueError, TypeError):
                train_subjects_arr, test_subjects_arr = train_test_split(
                    unique_subjects,
                    test_size=min(args.test_size, 0.5),
                    random_state=random_seed
                )
            
            train_subjects_set = set(train_subjects_arr)
            test_subjects_set = set(test_subjects_arr)
            train_mask = np.array([s in train_subjects_set for s in subjects])
            test_mask = np.array([s in test_subjects_set for s in subjects])
            
            # Store train/test sizes for first run
            if run_idx == 0:
                dataset_results['train_size'] = int(len(X_train))
                dataset_results['test_size'] = int(len(X_test))
            
            # Get test subjects for per-subject evaluation
            test_subjects_eval = None
            if args.per_subject_eval:
                test_subjects_eval = subjects[test_mask]
            
            # Evaluate each classifier
            for clf_name in args.classifiers:
                try:
                    results, clf = evaluate_disease_classification(
                        X_train, y_train, X_test, y_test,
                        clf_name, cv_folds=args.cv_folds,
                        class_names=class_names,
                        mlp_params=mlp_params,
                        test_subjects=test_subjects_eval,
                        per_subject_eval=args.per_subject_eval
                    )
                    # Only append if results are valid (not all None)
                    if results.get('test_accuracy') is not None or results.get('train_accuracy') is not None:
                        all_runs_results[clf_name].append(results)
                    else:
                        print(f"  ⚠️  Warning: {clf_name} returned None results, skipping this run")
                except Exception as e:
                    print(f"  ✗ Error evaluating {clf_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Still append None results to maintain run count consistency
                    all_runs_results[clf_name].append({
                        'classifier': clf_name,
                        'test_accuracy': None,
                        'train_accuracy': None,
                        'error': str(e)
                    })
                
                if run_idx == 0:  # Print detailed results only for first run
                    print(f"\n{clf_name}:")
                    acc = results.get('test_accuracy')
                    bal_acc = results.get('test_balanced_accuracy')
                    f1_macro = results.get('test_f1_macro')
                    f1_weighted = results.get('test_f1_weighted')
                    
                    if acc is not None:
                        print(f"  Accuracy:          {acc:.4f}")
                    else:
                        print(f"  Accuracy:          N/A")
                    if bal_acc is not None:
                        print(f"  Balanced Accuracy: {bal_acc:.4f}")
                    else:
                        print(f"  Balanced Accuracy: N/A")
                    if f1_macro is not None:
                        print(f"  F1 (macro):        {f1_macro:.4f}")
                    else:
                        print(f"  F1 (macro):        N/A")
                    if f1_weighted is not None:
                        print(f"  F1 (weighted):     {f1_weighted:.4f}")
                    else:
                        print(f"  F1 (weighted):     N/A")
                    if results.get('test_auc') is not None:
                        print(f"  ROC AUC:           {results['test_auc']:.4f}")
                    if results.get('sensitivity') is not None:
                        print(f"  Sensitivity:       {results['sensitivity']:.4f}")
                        print(f"  Specificity:       {results['specificity']:.4f}")
                    cv_mean = results.get('cv_balanced_accuracy')
                    cv_std = results.get('cv_balanced_accuracy_std')
                    if cv_mean is None or (isinstance(cv_mean, float) and np.isnan(cv_mean)):
                        print("  CV Balanced Acc:   N/A")
                    else:
                        print(f"  CV Balanced Acc:   {cv_mean:.4f} ± {cv_std:.4f}")
                else:
                    # Brief summary for subsequent runs
                    acc = results.get('test_accuracy')
                    bal_acc = results.get('test_balanced_accuracy')
                    f1_macro = results.get('test_f1_macro')
                    acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                    bal_acc_str = f"{bal_acc:.4f}" if bal_acc is not None else "N/A"
                    f1_str = f"{f1_macro:.4f}" if f1_macro is not None else "N/A"
                    print(f"  {clf_name}: Acc={acc_str}, BalAcc={bal_acc_str}, F1={f1_str}")
            
            # Evaluate age prediction if available
            if has_age and ages is not None:
                # Align ages if needed (same logic as before)
                ages_aligned = ages.copy()
                if len(ages_aligned) != len(features):
                    if len(ages_aligned) < len(features):
                        ages_aligned = np.full(len(features), np.nan)
                        ages_aligned[:len(ages)] = ages
                    else:
                        ages_aligned = ages_aligned[:len(features)]
                
                if len(train_mask) == len(ages_aligned):
                    ages_train = ages_aligned[train_mask]
                    ages_test = ages_aligned[test_mask]
                    age_results = evaluate_age_prediction(
                        X_train, ages_train, X_test, ages_test, mlp_params=mlp_params
                    )
                    all_runs_age_results.append(age_results)
            
            # Evaluate sex prediction if available
            if has_sex and sexes is not None:
                # Align sexes if needed (same logic as before)
                sexes_aligned = sexes.copy()
                if len(sexes_aligned) != len(features):
                    if len(sexes_aligned) < len(features):
                        sexes_aligned = np.full(len(features), np.nan)
                        sexes_aligned[:len(sexes)] = sexes
                    else:
                        sexes_aligned = sexes_aligned[:len(features)]
                
                if len(train_mask) == len(sexes_aligned):
                    sexes_train = sexes_aligned[train_mask]
                    sexes_test = sexes_aligned[test_mask]
                    sex_results = evaluate_sex_prediction(
                        X_train, sexes_train, X_test, sexes_test, mlp_params=mlp_params
                    )
                    all_runs_sex_results.append(sex_results)
        
        # Aggregate results across runs
        print(f"\n{'='*70}")
        print("AGGREGATED RESULTS (Mean ± Std across all runs)")
        print(f"{'='*70}")
        
        def aggregate_metrics(run_results_list, metric_keys):
            """Aggregate numeric metrics across runs"""
            def is_valid_value(val):
                """Check if value is valid (not None, not NaN, and scalar)"""
                if val is None:
                    return False
                # Skip arrays - only aggregate scalar values
                if isinstance(val, np.ndarray):
                    return False
                # Check for NaN (works for scalar floats)
                try:
                    if np.isnan(val):
                        return False
                except (TypeError, ValueError):
                    # Not a numeric type that can be NaN
                    return False
                return True
            
            aggregated = {}
            for key in metric_keys:
                values = [r[key] for r in run_results_list if is_valid_value(r.get(key))]
                if len(values) > 0:
                    aggregated[key] = float(np.mean(values))
                    aggregated[f'{key}_std'] = float(np.std(values))
                else:
                    aggregated[key] = None
                    aggregated[f'{key}_std'] = None
            return aggregated
        
        # Aggregate classifier results
        for clf_name in args.classifiers:
            if len(all_runs_results[clf_name]) == 0:
                continue
            
            # Get all metric keys from first run
            metric_keys = [k for k in all_runs_results[clf_name][0].keys() 
                          if k not in ['classifier', 'n_classes', 'class_names', 'confusion_matrix', 
                                      'classification_report', 'predictions', 'cv_balanced_accuracy_std']]
            
            aggregated = aggregate_metrics(all_runs_results[clf_name], metric_keys)
            aggregated['classifier'] = clf_name
            aggregated['n_classes'] = all_runs_results[clf_name][0]['n_classes']
            aggregated['class_names'] = all_runs_results[clf_name][0]['class_names']
            aggregated['runs'] = all_runs_results[clf_name]  # Store individual runs
            
            dataset_results['classifiers'][clf_name] = aggregated
            
            print(f"\n{clf_name} (n={args.n_runs} runs):")
            acc = aggregated.get('test_accuracy')
            bal_acc = aggregated.get('test_balanced_accuracy')
            f1_macro = aggregated.get('test_f1_macro')
            acc_std = aggregated.get('test_accuracy_std', 0)
            bal_acc_std = aggregated.get('test_balanced_accuracy_std', 0)
            f1_std = aggregated.get('test_f1_macro_std', 0)
            
            if acc is not None:
                print(f"  Accuracy:          {acc:.4f} ± {acc_std:.4f}")
            else:
                print(f"  Accuracy:          N/A")
            if bal_acc is not None:
                print(f"  Balanced Accuracy: {bal_acc:.4f} ± {bal_acc_std:.4f}")
            else:
                print(f"  Balanced Accuracy: N/A")
            if f1_macro is not None:
                print(f"  F1 (macro):        {f1_macro:.4f} ± {f1_std:.4f}")
            else:
                print(f"  F1 (macro):        N/A")
            f1_weighted = aggregated.get('test_f1_weighted')
            f1_weighted_std = aggregated.get('test_f1_weighted_std', 0)
            if f1_weighted is not None:
                print(f"  F1 (weighted):     {f1_weighted:.4f} ± {f1_weighted_std:.4f}")
            else:
                print(f"  F1 (weighted):     N/A")
            if aggregated.get('test_auc') is not None:
                auc = aggregated.get('test_auc')
                auc_std = aggregated.get('test_auc_std', 0)
                print(f"  ROC AUC:           {auc:.4f} ± {auc_std:.4f}")
        
        # Find best classifier based on mean test accuracy
        best_classifier = None
        best_mean_acc = -1
        for clf_name, aggregated in dataset_results['classifiers'].items():
            acc = aggregated.get('test_accuracy')
            if acc is not None and acc > best_mean_acc:
                best_mean_acc = acc
                best_classifier = clf_name
        
        dataset_results['best_classifier'] = best_classifier
        if best_classifier:
            best_aggregated = dataset_results['classifiers'][best_classifier]
            dataset_results['best_test_accuracy'] = float(best_mean_acc)
            dataset_results['best_test_accuracy_std'] = float(best_aggregated.get('test_accuracy_std', 0))
            # Also keep balanced accuracy for backward compatibility
            dataset_results['best_balanced_accuracy'] = float(best_aggregated.get('test_balanced_accuracy', 0))
            dataset_results['best_balanced_accuracy_std'] = float(best_aggregated.get('test_balanced_accuracy_std', 0))
            # Store weighted F1
            dataset_results['best_test_f1_weighted'] = float(best_aggregated.get('test_f1_weighted', 0))
            dataset_results['best_test_f1_weighted_std'] = float(best_aggregated.get('test_f1_weighted_std', 0))
            
            print(f"\n{'='*70}")
            print(f"BEST CLASSIFIER: {best_classifier}")
            print(f"{'='*70}")
            acc = best_aggregated.get('test_accuracy')
            bal_acc = best_aggregated.get('test_balanced_accuracy')
            f1_macro = best_aggregated.get('test_f1_macro')
            acc_std = best_aggregated.get('test_accuracy_std', 0)
            bal_acc_std = best_aggregated.get('test_balanced_accuracy_std', 0)
            f1_std = best_aggregated.get('test_f1_macro_std', 0)
            
            if acc is not None:
                print(f"Accuracy:          {acc:.4f} ± {acc_std:.4f}")
            else:
                print(f"Accuracy:          N/A")
            if bal_acc is not None:
                print(f"Balanced Accuracy: {bal_acc:.4f} ± {bal_acc_std:.4f}")
            else:
                print(f"Balanced Accuracy: N/A")
            if f1_macro is not None:
                print(f"F1 (macro):        {f1_macro:.4f} ± {f1_std:.4f}")
            else:
                print(f"F1 (macro):        N/A")
            f1_weighted = best_aggregated.get('test_f1_weighted')
            f1_weighted_std = best_aggregated.get('test_f1_weighted_std', 0)
            if f1_weighted is not None:
                print(f"F1 (weighted):     {f1_weighted:.4f} ± {f1_weighted_std:.4f}")
            else:
                print(f"F1 (weighted):     N/A")
            if best_aggregated.get('test_auc') is not None:
                auc = best_aggregated.get('test_auc')
                auc_std = best_aggregated.get('test_auc_std', 0)
                print(f"ROC AUC:           {auc:.4f} ± {auc_std:.4f}")
            
            # Use first run's best results for plotting (or could use aggregated)
            best_results = all_runs_results[best_classifier][0] if all_runs_results[best_classifier] else None
        else:
            best_results = None
        
        # Aggregate age prediction results
        if len(all_runs_age_results) > 0:
            print(f"\n{'='*70}")
            print("AGE PREDICTION (Aggregated)")
            print(f"{'='*70}")
            age_metric_keys = ['test_mse', 'test_mae', 'test_r2', 'test_pearson']
            aggregated_age = aggregate_metrics(all_runs_age_results, age_metric_keys)
            aggregated_age['runs'] = all_runs_age_results
            dataset_results['age_prediction'] = aggregated_age
            
            if aggregated_age.get('test_pearson') is not None:
                print(f"  Test MSE:     {aggregated_age['test_mse']:.3f} ± {aggregated_age.get('test_mse_std', 0):.3f} years²")
                print(f"  Test MAE:     {aggregated_age['test_mae']:.3f} ± {aggregated_age.get('test_mae_std', 0):.3f} years")
                print(f"  Test R²:      {aggregated_age['test_r2']:.3f} ± {aggregated_age.get('test_r2_std', 0):.3f}")
                print(f"  Test Pearson: {aggregated_age['test_pearson']:.3f} ± {aggregated_age.get('test_pearson_std', 0):.3f}")
        
        # Aggregate sex prediction results
        if len(all_runs_sex_results) > 0:
            print(f"\n{'='*70}")
            print("SEX PREDICTION (Aggregated)")
            print(f"{'='*70}")
            sex_metric_keys = ['test_accuracy', 'test_f1', 'test_auc']
            aggregated_sex = aggregate_metrics(all_runs_sex_results, sex_metric_keys)
            aggregated_sex['runs'] = all_runs_sex_results
            dataset_results['sex_prediction'] = aggregated_sex
            
            if aggregated_sex.get('test_accuracy') is not None:
                print(f"  Test Accuracy: {aggregated_sex['test_accuracy']:.3f} ± {aggregated_sex.get('test_accuracy_std', 0):.3f}")
                print(f"  Test F1:       {aggregated_sex['test_f1']:.3f} ± {aggregated_sex.get('test_f1_std', 0):.3f}")
                if aggregated_sex.get('test_auc') is not None:
                    print(f"  Test AUC:      {aggregated_sex['test_auc']:.3f} ± {aggregated_sex.get('test_auc_std', 0):.3f}")
        
        # Plot results
        plot_disease_diagnosis(best_results, plot_dir, dataset_name)
        
        all_results['datasets'][dataset_name] = dataset_results
    
    # Clean results to remove non-JSON-serializable objects
    def clean_for_json(obj):
        """Recursively clean dictionary/list to remove non-JSON-serializable objects"""
        # Handle basic types
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle numpy types (NumPy 2.0 compatible)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        # Also handle specific numeric types for NumPy 2.0
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle PyTorch tensors
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        
        # Handle dictionaries
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                # Skip known non-serializable keys
                if key in ['model_state_dict', 'scaler']:
                    continue
                # Recursively clean nested structures
                cleaned_value = clean_for_json(value)
                # Only add if cleaning didn't return None (for objects we can't serialize)
                if cleaned_value is not None:
                    cleaned[key] = cleaned_value
            return cleaned
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            cleaned_list = []
            for item in obj:
                cleaned_item = clean_for_json(item)
                if cleaned_item is not None:
                    cleaned_list.append(cleaned_item)
            return cleaned_list
        
        # Try to serialize directly (for basic JSON types)
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If it fails, it's likely a complex object (like scikit-learn objects)
            # Return None to skip it
            return None
    
    # Clean all results before saving
    all_results_cleaned = clean_for_json(all_results)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results_cleaned, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Results saved to {args.output}")
    print(f"✓ Plots saved to {args.plot_dir}/")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Dataset':<15} {'Best Classifier':<20} {'Test Acc':<12} {'F1 (weighted)':<15}")
    print("-"*70)
    for dataset_name, results in all_results['datasets'].items():
        best_clf = results['best_classifier']
        test_acc = results['classifiers'][best_clf]['test_accuracy']
        f1_weighted = results['classifiers'][best_clf]['test_f1_weighted']
        print(f"{dataset_name:<15} {best_clf:<20} {test_acc:<12.4f} {f1_weighted:<15.4f}")
    print("="*70)


if __name__ == '__main__':
    main()


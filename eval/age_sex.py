"""

Evaluate Age and Sex Prediction using Pretrained Koopman Model



This script:

1. Loads a pretrained Koopman model

2. Extracts features from BOLD time series

3. Trains linear probes for age (regression) and sex (classification)

4. Evaluates performance on held-out test set

5. Prints results to console



Usage:

    # Automatic (recommended)

    python evaluation/evaluate_age_sex.py --dataset HCPA

    python evaluation/evaluate_age_sex.py --dataset UKB



    # Overrides (optional)

    python evaluation/evaluate_age_sex.py --dataset HCPA --data_dir <custom_bold> --demographics_csv <custom_demo>

"""



import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader

import numpy as np

import pandas as pd

from scipy.stats import pearsonr

from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, f1_score

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

import argparse



from models.brainmech import BrainMech

from utils.data_loader import HCPABOLDDataset, UKBiobankBOLDDataset, UKBiobankBOLDWithFCDataset

try:

    from scripts.manage_checkpoints import find_best_checkpoint

except ImportError:

    find_best_checkpoint = None

try:

    from evaluation.frozen_mlp_head import (

        train_regression_head,

        train_classification_head,

    )

except ImportError:

    try:

        from eval.frozen_mlp_head import (

            train_regression_head,

            train_classification_head,

        )

    except ImportError:

        from frozen_mlp_head import (

            train_regression_head,

            train_classification_head,

        )

try:

    from eval.feature_cache import get_cache_key, load_cached_features, save_cached_features, get_model_params

except ImportError:

    from feature_cache import get_cache_key, load_cached_features, save_cached_features, get_model_params





def load_demographics(csv_path: str, dataset_type: str) -> pd.DataFrame:

    """

    Load demographics CSV file

    

    Args:

        csv_path: Path to demographics CSV file

        dataset_type: 'HCPA' or 'UKB'

    

    Returns:

        DataFrame with subject_id, age, sex columns

    """

    print(f"Loading demographics from {csv_path}...")

    df = pd.read_csv(csv_path)

    

    if dataset_type.upper() == 'HCPA':

        # HCPA format: has 'Subject', 'Age', 'Sex' columns

        # Subject column format might be like 'sub-123' or just the ID

        if 'Subject' in df.columns:

            subject_col = 'Subject'

        elif 'subjectkey' in df.columns:

            subject_col = 'subjectkey'

        else:

            raise ValueError(f"Could not find subject ID column. Available columns: {df.columns.tolist()}")

        

        # Extract relevant columns

        demo_df = df[[subject_col, 'Age', 'Sex']].copy()

        demo_df.columns = ['subject_id', 'age', 'sex']

        

        # Clean up subject IDs (remove 'sub-' prefix if present)

        demo_df['subject_id'] = demo_df['subject_id'].astype(str).str.replace('sub-', '', regex=False)

        

        # Convert sex to binary (0=Female, 1=Male)

        # HCPA typically uses 'M'/'F' or 1/2

        if demo_df['sex'].dtype == 'object':

            demo_df['sex'] = demo_df['sex'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})

        else:

            # If numeric, assume 1=Male, 2=Female or 0=Female, 1=Male

            if demo_df['sex'].min() == 1 and demo_df['sex'].max() == 2:

                demo_df['sex'] = demo_df['sex'] - 1  # Convert 1/2 to 0/1

        

    elif dataset_type.upper() == 'UKB':

        # UKB format: eid, p31 (sex), p34 (birth year), p53_i2 (scan date)

        # Age = p53_i2 - p34

        # NOTE: p34 often has missing values - we'll drop those

        

        if 'eid' not in df.columns:

            raise ValueError(f"UKB data must have 'eid' column. Available columns: {df.columns.tolist()}")

        

        subject_col = 'eid'

        

        # Check if age needs to be computed

        if 'age' in df.columns or 'Age' in df.columns:

            # Age already computed

            age_col = 'age' if 'age' in df.columns else 'Age'

            demo_df = df[[subject_col, age_col, 'p31']].copy()

            demo_df.columns = ['subject_id', 'age', 'sex']

            print("  Using pre-computed age column")

        elif 'p34' in df.columns and 'p53_i2' in df.columns:

            # Need to compute age from birth year and scan date

            print("  Computing age from p53_i2 (scan date) - p34 (birth year)")

            

            # First, filter out missing p34 values

            df_filtered = df[['eid', 'p34', 'p53_i2', 'p31']].copy()

            n_total = len(df_filtered)

            

            # Drop rows with missing p34 (birth year)

            df_filtered = df_filtered.dropna(subset=['p34'])

            n_missing_p34 = n_total - len(df_filtered)

            if n_missing_p34 > 0:

                print(f"  ⚠️  Dropped {n_missing_p34} subjects with missing birth year (p34)")

            

            # Drop rows with missing p53_i2 (scan date)

            df_filtered = df_filtered.dropna(subset=['p53_i2'])

            n_missing_scan = len(df_filtered)

            

            # Extract year from p53_i2 (scan date)

            def extract_year(date_str):

                if pd.isna(date_str):

                    return np.nan

                date_str = str(date_str)

                # Try to extract first 4 digits (year)

                import re

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

            

            # Sanity check: age should be reasonable (e.g., 18-100)

            df_filtered = df_filtered[(df_filtered['age'] >= 18) & (df_filtered['age'] <= 100)]

            

            demo_df = df_filtered[[subject_col, 'age', 'p31']].copy()

            demo_df.columns = ['subject_id', 'age', 'sex']

            

            print(f"  ✓ Successfully computed age for {len(demo_df)} subjects (dropped {n_total - len(demo_df)} incomplete/invalid entries)")

        else:

            raise ValueError(f"UKB data must have either 'age' or ('p34' and 'p53_i2'). Available columns: {df.columns.tolist()}")

        

        # Clean subject IDs

        demo_df['subject_id'] = demo_df['subject_id'].astype(str)

        

        # p31 is already 0/1 (0=Female, 1=Male) in UKB

        demo_df['sex'] = pd.to_numeric(demo_df['sex'], errors='coerce')

    

    elif dataset_type.upper() == 'PPMI':

        # PPMI format: Image Data ID,Subject,Group,Sex,Age,Visit,Modality,Description

        # Subject IDs are numeric like "3101"

        print(f"  Processing PPMI demographics...")

        

        col_map = {c.lower(): c for c in df.columns}

        

        subject_col = col_map.get('subject', col_map.get('subject_id', None))

        age_col = col_map.get('age', None)

        sex_col = col_map.get('sex', col_map.get('gender', None))

        

        if subject_col is None:

            raise ValueError(f"PPMI: Could not find subject column. Available: {df.columns.tolist()}")

        if age_col is None:

            raise ValueError(f"PPMI: Could not find age column. Available: {df.columns.tolist()}")

        if sex_col is None:

            raise ValueError(f"PPMI: Could not find sex column. Available: {df.columns.tolist()}")

        

        demo_df = df[[subject_col, age_col, sex_col]].copy()

        demo_df.columns = ['subject_id', 'age', 'sex']

        

        # Clean subject IDs (PPMI uses numeric IDs like "3101")

        demo_df['subject_id'] = demo_df['subject_id'].astype(str).str.strip()

        # Create normalized version for flexible matching

        demo_df['subject_id_norm'] = demo_df['subject_id'].str.lstrip('0')

        

        # Convert sex from F/M to 0/1 (0=Female, 1=Male)

        sex_mapping = {'F': 0, 'f': 0, 'Female': 0, 'female': 0, 

                       'M': 1, 'm': 1, 'Male': 1, 'male': 1}

        demo_df['sex'] = demo_df['sex'].map(sex_mapping)

        

        print(f"  ✓ Loaded {len(demo_df)} subjects from PPMI metadata")

        print(f"    Subject ID examples: {demo_df['subject_id'].head(3).tolist()}")

    

    elif dataset_type.upper() == 'ABCD':

        # ABCD format: src_subject_id, eventname, sex, age

        # Multiple rows per subject (different timepoints)

        print(f"  Processing ABCD demographics...")

        

        if 'src_subject_id' not in df.columns:

            raise ValueError(f"ABCD data must have 'src_subject_id' column. Available columns: {df.columns.tolist()}")

        if 'eventname' not in df.columns:

            raise ValueError(f"ABCD data must have 'eventname' column. Available columns: {df.columns.tolist()}")

        

        # Extract relevant columns

        demo_df = df[['src_subject_id', 'eventname', 'sex', 'age']].copy()

        demo_df.columns = ['subject_id', 'eventname', 'sex', 'age']

        

        # Clean subject IDs (remove 'sub-' prefix if present, handle NDAR format)

        demo_df['subject_id'] = demo_df['subject_id'].astype(str).str.replace('sub-', '', regex=False)

        

        # Normalize eventname (lowercase, handle variations)

        demo_df['eventname'] = demo_df['eventname'].astype(str).str.lower().str.strip()

        

        # Convert sex to binary (0=Female, 1=Male)

        # ABCD typically uses 1/2 (1=Male, 2=Female) or 1.0/2.0

        demo_df['sex'] = pd.to_numeric(demo_df['sex'], errors='coerce')

        if demo_df['sex'].min() == 1 and demo_df['sex'].max() == 2:

            # Convert 1/2 to 0/1 (1=Male -> 1, 2=Female -> 0)

            demo_df['sex'] = demo_df['sex'] - 1

        elif demo_df['sex'].min() == 0 and demo_df['sex'].max() == 1:

            # Already 0/1 format

            pass

        else:

            print(f"  ⚠️  Warning: Unexpected sex encoding. Values: {demo_df['sex'].unique()}")

        

        # Convert age to numeric

        demo_df['age'] = pd.to_numeric(demo_df['age'], errors='coerce')

        

        # Drop rows with missing age or sex

        n_before = len(demo_df)

        demo_df = demo_df.dropna(subset=['age', 'sex'])

        n_dropped = n_before - len(demo_df)

        if n_dropped > 0:

            print(f"  ⚠️  Dropped {n_dropped} rows with missing age or sex")

        

        print(f"  ✓ Loaded {len(demo_df)} subject-timepoint pairs from ABCD metadata")

        print(f"    Unique subjects: {demo_df['subject_id'].nunique()}")

        print(f"    Eventnames: {demo_df['eventname'].unique()[:5].tolist()}...")

    

    else:

        # Generic format - try to auto-detect columns

        print(f"  Processing generic demographics format for {dataset_type}...")

        

        col_map = {c.lower(): c for c in df.columns}

        

        # Try to find subject column

        subject_col = None

        for candidate in ['subject', 'subject_id', 'subjectkey', 'id', 'eid', 'participant_id']:

            if candidate in col_map:

                subject_col = col_map[candidate]

                break

        

        # Try to find age column

        age_col = None

        for candidate in ['age', 'age_years', 'years']:

            if candidate in col_map:

                age_col = col_map[candidate]

                break

        

        # Try to find sex column

        sex_col = None

        for candidate in ['sex', 'gender', 'sex_at_birth']:

            if candidate in col_map:

                sex_col = col_map[candidate]

                break

        

        if subject_col is None or age_col is None or sex_col is None:

            raise ValueError(

                f"Could not auto-detect columns for {dataset_type}. "

                f"Found: subject={subject_col}, age={age_col}, sex={sex_col}. "

                f"Available columns: {df.columns.tolist()}"

            )

        

        demo_df = df[[subject_col, age_col, sex_col]].copy()

        demo_df.columns = ['subject_id', 'age', 'sex']

        

        # Clean subject IDs

        demo_df['subject_id'] = demo_df['subject_id'].astype(str).str.strip()

        # Create normalized version for flexible matching

        demo_df['subject_id_norm'] = demo_df['subject_id'].str.lstrip('0')

        

        # Try to convert sex to numeric

        if demo_df['sex'].dtype == object:

            sex_mapping = {'F': 0, 'f': 0, 'Female': 0, 'female': 0, 

                           'M': 1, 'm': 1, 'Male': 1, 'male': 1,

                           '0': 0, '1': 1}

            demo_df['sex'] = demo_df['sex'].map(sex_mapping)

        

        print(f"  ✓ Loaded {len(demo_df)} subjects using auto-detected columns")

    

    # Remove missing values

    demo_df = demo_df.dropna()

    

    # Convert types

    demo_df['age'] = pd.to_numeric(demo_df['age'], errors='coerce')

    demo_df['sex'] = pd.to_numeric(demo_df['sex'], errors='coerce')

    demo_df = demo_df.dropna()

    

    # Check if age is in months (for HCPA - pediatric/adolescent data)

    # UKB ages are always in years (computed from birth year and scan date)

    if demo_df['age'].notna().any():

        age_max = demo_df['age'].max()

        age_min = demo_df['age'].min()

        age_mean = demo_df['age'].mean()

        

        # Heuristics to detect age in months:

        # 1. Max age > 150 (impossible in years, likely months)

        # 2. Mean age > 100 and dataset is HCPA (likely months for pediatric data)

        # 3. All ages > 100 (definitely months)

        is_likely_months = False

        

        if age_max > 150:

            is_likely_months = True

            print(f"  ⚠️  Age max ({age_max:.1f}) > 150 - definitely in months")

        elif dataset_type.upper() == 'HCPA' and age_mean > 100:

            is_likely_months = True

            print(f"  ⚠️  HCPA dataset with mean age ({age_mean:.1f}) > 100 - likely in months")

        elif age_min > 100:

            is_likely_months = True

            print(f"  ⚠️  Age min ({age_min:.1f}) > 100 - definitely in months")

        

        if is_likely_months:

            print(f"  Converting age from months (range: {age_min:.1f}-{age_max:.1f}) to years...")

            demo_df['age'] = demo_df['age'] / 12.0

            print(f"  Converted age range: {demo_df['age'].min():.1f} - {demo_df['age'].max():.1f} years")

        

        # Final sanity check: ages should be in reasonable range (0-120 years)

        age_max_final = demo_df['age'].max()

        age_min_final = demo_df['age'].min()

        if age_max_final > 120 or age_min_final < 0:

            print(f"  ⚠️  WARNING: Age range ({age_min_final:.1f}-{age_max_final:.1f}) seems invalid!")

            print(f"     Check your demographics CSV for data issues.")

    

    print(f"✓ Loaded demographics for {len(demo_df)} subjects")

    print(f"  Age range: {demo_df['age'].min():.1f} - {demo_df['age'].max():.1f} years")

    print(f"  Sex distribution: Female={len(demo_df[demo_df['sex']==0])}, Male={len(demo_df[demo_df['sex']==1])}")

    

    return demo_df





def extract_features(model, dataloader, device, feature_type='z_concat'):

    """

    Extract features from pretrained model

    

    Args:

        model: Pretrained Koopman model

        dataloader: DataLoader for BOLD data

        device: torch device

        feature_type: Type of features to extract

            - 'z_mean': Mean pooling over time [B, D]

            - 'z_std': Std pooling over time [B, D]

            - 'z_concat': Concatenate mean and std [B, 2*D] (RECOMMENDED)

            - 'z_max': Max pooling over time [B, D]

            - 'z_all': Concatenate mean, std, max, min [B, 4*D]

    

    Returns:

        features: [N, D] numpy array

        subjects: List of subject IDs (or dicts with subject_id and eventname for ABCD)

    """

    model.eval()

    all_features = []

    all_subjects = []

    all_eventnames = []  # For ABCD: store eventname separately

    

    print(f"Extracting {feature_type} features from pretrained model...")

    print(f"  DataLoader has {len(dataloader)} batches")

    

    if len(dataloader) == 0:

        raise ValueError(

            "DataLoader is empty - no batches to process. "

            "This usually means the dataset loaded but has no valid samples. "

            "Check that:\n"

            "  1. Files in the data directory match expected format\n"

            "  2. Files are not corrupted or empty\n"

            "  3. Sequence length requirements are met"

        )

    

    with torch.no_grad():

        batch_count = 0

        for batch_x, metadata in tqdm(dataloader, desc="Extracting features"):

            batch_x = batch_x.to(device)

            batch_count += 1

            

            # Get FC matrix if available (handle both dict and list metadata formats)

            fc_matrix = None

            batch_roi_count = batch_x.shape[2]  # [B, T, ROI]

            

            if isinstance(metadata, dict) and 'fc_matrix' in metadata:

                fc_matrix = metadata['fc_matrix'].to(device)

                # Ensure FC matrix matches batch ROI count

                if fc_matrix.dim() == 2:

                    # Single FC matrix - ensure it matches

                    if fc_matrix.shape[0] != batch_roi_count or fc_matrix.shape[1] != batch_roi_count:

                        # Resize to match

                        if fc_matrix.shape[0] < batch_roi_count:

                            pad = torch.zeros(batch_roi_count - fc_matrix.shape[0], fc_matrix.shape[1],

                                            dtype=fc_matrix.dtype, device=fc_matrix.device)

                            fc_matrix = torch.cat([fc_matrix, pad], dim=0)

                        else:

                            fc_matrix = fc_matrix[:batch_roi_count, :]

                        if fc_matrix.shape[1] < batch_roi_count:

                            pad = torch.zeros(fc_matrix.shape[0], batch_roi_count - fc_matrix.shape[1],

                                            dtype=fc_matrix.dtype, device=fc_matrix.device)

                            fc_matrix = torch.cat([fc_matrix, pad], dim=1)

                        else:

                            fc_matrix = fc_matrix[:, :batch_roi_count]

                elif fc_matrix.dim() == 3:

                    # Batch of FC matrices [B, ROI, ROI]

                    if fc_matrix.shape[1] != batch_roi_count or fc_matrix.shape[2] != batch_roi_count:

                        # Resize each FC matrix

                        batch_fc = []

                        for i in range(fc_matrix.shape[0]):

                            fc = fc_matrix[i]

                            if fc.shape[0] < batch_roi_count:

                                pad = torch.zeros(batch_roi_count - fc.shape[0], fc.shape[1],

                                                dtype=fc.dtype, device=fc.device)

                                fc = torch.cat([fc, pad], dim=0)

                            else:

                                fc = fc[:batch_roi_count, :]

                            if fc.shape[1] < batch_roi_count:

                                pad = torch.zeros(fc.shape[0], batch_roi_count - fc.shape[1],

                                                dtype=fc.dtype, device=fc.device)

                                fc = torch.cat([fc, pad], dim=1)

                            else:

                                fc = fc[:, :batch_roi_count]

                            batch_fc.append(fc)

                        fc_matrix = torch.stack(batch_fc)

            elif isinstance(metadata, list):

                # Custom collate format - FC matrices are in individual metadata dicts

                fc_list = []

                for meta in metadata:

                    if isinstance(meta, dict) and 'fc_matrix' in meta:

                        fc = meta['fc_matrix'].to(device)

                        # Ensure FC matches batch ROI count

                        if fc.shape[0] != batch_roi_count or fc.shape[1] != batch_roi_count:

                            if fc.shape[0] < batch_roi_count:

                                pad = torch.zeros(batch_roi_count - fc.shape[0], fc.shape[1],

                                                dtype=fc.dtype, device=fc.device)

                                fc = torch.cat([fc, pad], dim=0)

                            else:

                                fc = fc[:batch_roi_count, :]

                            if fc.shape[1] < batch_roi_count:

                                pad = torch.zeros(fc.shape[0], batch_roi_count - fc.shape[1],

                                                dtype=fc.dtype, device=fc.device)

                                fc = torch.cat([fc, pad], dim=1)

                            else:

                                fc = fc[:, :batch_roi_count]

                        fc_list.append(fc)

                    else:

                        # No FC - use identity

                        fc_list.append(torch.eye(batch_roi_count, dtype=torch.float32, device=device))

                

                if fc_list:

                    try:

                        fc_matrix = torch.stack(fc_list)

                    except RuntimeError as e:

                        print(f"  ⚠️  Warning: Could not stack FC matrices (shape mismatch): {e}")

                        print(f"     Batch ROI count: {batch_roi_count}, FC shapes: {[f.shape for f in fc_list[:3]]}")

                        # Use identity matrices as fallback

                        fc_matrix = None

            

            # Forward pass through model to get latent features

            outputs = model(batch_x, task_vectors=None, fc_matrix=fc_matrix, return_uncertainty=False)

            

            # Extract latent features (temporal pooling)

            z_latent = outputs['z_latent']  # [B, T, D]

            

            # Pool over time dimension based on feature_type

            if feature_type == 'z_mean':

                z_pooled = z_latent.mean(dim=1)  # [B, D]

            elif feature_type == 'z_std':

                z_pooled = z_latent.std(dim=1)  # [B, D]

            elif feature_type == 'z_max':

                z_pooled = z_latent.max(dim=1)[0]  # [B, D]

            elif feature_type == 'z_concat':

                # Concatenate mean and std for richer representation

                z_mean = z_latent.mean(dim=1)  # [B, D]

                z_std = z_latent.std(dim=1)   # [B, D]

                z_pooled = torch.cat([z_mean, z_std], dim=1)  # [B, 2*D]

            elif feature_type == 'z_all':

                # Concatenate mean, std, max, min

                z_mean = z_latent.mean(dim=1)

                z_std = z_latent.std(dim=1)

                z_max = z_latent.max(dim=1)[0]

                z_min = z_latent.min(dim=1)[0]

                z_pooled = torch.cat([z_mean, z_std, z_max, z_min], dim=1)  # [B, 4*D]

            else:

                raise ValueError(f"Unknown feature_type: {feature_type}")

            

            all_features.append(z_pooled.cpu().numpy())

            

            # Extract subject IDs and eventnames - handle different metadata formats

            if isinstance(metadata, dict):

                # Standard format: dict with lists

                subjects = metadata.get('subject', metadata.get('file', ['unknown'] * len(batch_x)))

                eventnames = metadata.get('eventname', [None] * len(batch_x))

            elif isinstance(metadata, list):

                # Custom collate format: list of dicts (from custom_collate_fn)

                subjects = [m.get('subject', m.get('file', 'unknown')) if isinstance(m, dict) else 'unknown' for m in metadata]

                eventnames = [m.get('eventname', None) if isinstance(m, dict) else None for m in metadata]

            else:

                subjects = ['unknown'] * len(batch_x)

                eventnames = [None] * len(batch_x)

            

            # Handle both string and list formats

            if isinstance(subjects, list):

                all_subjects.extend(subjects)

            else:

                # Might be a tensor or batch of strings

                if isinstance(subjects, torch.Tensor):

                    subjects = subjects.cpu().numpy()

                all_subjects.extend([str(s) for s in subjects])

            

            # Store eventnames for ABCD matching

            if isinstance(eventnames, list):

                all_eventnames.extend(eventnames)

            else:

                all_eventnames.extend([None] * len(batch_x))

    

    if len(all_features) == 0:

        raise ValueError(

            f"No features extracted after processing {batch_count} batches. "

            "This suggests the model forward pass failed or returned empty results. "

            "Check model checkpoint compatibility and data format."

        )

    

    features = np.concatenate(all_features, axis=0)

    print(f"✓ Extracted features: {features.shape} from {batch_count} batches")

    

    # Return eventnames along with subjects for ABCD matching

    if any(e is not None for e in all_eventnames):

        return features, all_subjects, all_eventnames

    else:

        return features, all_subjects, None





def train_age_predictor(X_train, y_train, X_test, y_test):

    """Train and evaluate age regression model with hyperparameter search"""

    from sklearn.model_selection import GridSearchCV

    from sklearn.decomposition import PCA

    from sklearn.feature_selection import SelectKBest, f_regression

    

    # Standardize features

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    

    # Check feature quality

    print(f"  Feature stats: mean={X_train_scaled.mean():.3f}, std={X_train_scaled.std():.3f}")

    print(f"  Feature range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")

    

    # Feature selection: Select top features correlated with age

    # This helps when features are high-dimensional and many are uninformative

    print("  Performing feature selection...")

    n_features = min(500, X_train_scaled.shape[1])  # Select top 500 or all if fewer

    selector = SelectKBest(score_func=f_regression, k=n_features)

    X_train_selected = selector.fit_transform(X_train_scaled, y_train)

    X_test_selected = selector.transform(X_test_scaled)

    print(f"  Selected {n_features} features from {X_train_scaled.shape[1]} (kept {100*n_features/X_train_scaled.shape[1]:.1f}%)")

    

    # Optional: Apply PCA for further dimensionality reduction if still high-dim

    # PCA requires n_components <= min(n_samples, n_features)

    if X_train_selected.shape[1] > 200:

        print("  Applying PCA for dimensionality reduction...")

        # Limit n_components to available samples and features

        # Use min(n_samples, n_features) - 1 for numerical stability, but at least 1

        max_components = min(X_train_selected.shape[0], X_train_selected.shape[1], 200)

        # For very small datasets, use fewer components to avoid numerical issues

        if max_components >= X_train_selected.shape[0]:

            max_components = max(1, X_train_selected.shape[0] - 1)

        

        if max_components < 1 or X_train_selected.shape[0] < 2:

            # Not enough samples for PCA, skip it

            print(f"  ⚠️  Skipping PCA: only {X_train_selected.shape[0]} samples available (need at least 2)")

            X_train_final = X_train_selected

            X_test_final = X_test_selected

        else:

            pca = PCA(n_components=max_components, random_state=42)

            X_train_final = pca.fit_transform(X_train_selected)

            X_test_final = pca.transform(X_test_selected)

            print(f"  PCA: {X_train_selected.shape[1]} -> {X_train_final.shape[1]} features (explained variance: {pca.explained_variance_ratio_.sum():.3f})")

    else:

        X_train_final = X_train_selected

        X_test_final = X_test_selected

    

    # Train Ridge regression with hyperparameter search

    print("Training age predictor (Ridge regression with CV)...")

    # Wider alpha range, but focus on lower values first

    # Very high alpha (1000+) suggests features aren't informative - try lower regularization

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]}

    ridge = Ridge(solver='cholesky')  # Use cholesky solver for better numerical stability

    # Use neg_mean_squared_error for better alignment with MSE metric we report

    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(X_train_final, y_train)

    

    model = grid_search.best_estimator_

    best_alpha = grid_search.best_params_['alpha']

    print(f"  Best alpha: {best_alpha}")

    

    # Diagnostic: Check if we're at the boundary of the search space

    if best_alpha == max(param_grid['alpha']):

        print(f"  ⚠️  Warning: Best alpha is at maximum ({best_alpha}). Features may not be informative.")

        print(f"     Consider: (1) Using --feature_type z_all for richer features")

        print(f"               (2) Using --head_type frozen_mlp instead of linear")

        print(f"               (3) Checking if model checkpoint is optimal")

    elif best_alpha == min(param_grid['alpha']):

        print(f"  ⚠️  Warning: Best alpha is at minimum ({best_alpha}). Model might be overfitting.")

    

    # Diagnostic: Show CV scores to assess generalization

    cv_scores = grid_search.cv_results_['mean_test_score']

    best_idx = grid_search.best_index_

    print(f"  CV MSE: {-cv_scores[best_idx]:.2f} years² (best), range: [{-cv_scores.max():.2f}, {-cv_scores.min():.2f}]")

    

    # Predictions

    y_pred_train = model.predict(X_train_final)

    y_pred_test = model.predict(X_test_final)

    

    # Diagnostic: Check prediction ranges

    print(f"  Train predictions: [{y_pred_train.min():.1f}, {y_pred_train.max():.1f}] years")

    print(f"  Train ground truth: [{y_train.min():.1f}, {y_train.max():.1f}] years")

    print(f"  Test predictions: [{y_pred_test.min():.1f}, {y_pred_test.max():.1f}] years")

    print(f"  Test ground truth: [{y_test.min():.1f}, {y_test.max():.1f}] years")

    

    # Check for extreme predictions (potential overfitting indicator)

    train_outliers = np.abs(y_pred_train - y_train) > 20  # More than 20 years off

    test_outliers = np.abs(y_pred_test - y_test) > 20

    if train_outliers.sum() > 0:

        print(f"  ⚠️  Train: {train_outliers.sum()}/{len(y_train)} predictions off by >20 years")

    if test_outliers.sum() > 0:

        print(f"  ⚠️  Test: {test_outliers.sum()}/{len(y_test)} predictions off by >20 years")

    

    # Metrics

    # Pearson correlation

    train_pearson, _ = pearsonr(y_train, y_pred_train)

    test_pearson, _ = pearsonr(y_test, y_pred_test)

    

    train_mse = mean_squared_error(y_train, y_pred_train)

    test_mse = mean_squared_error(y_test, y_pred_test)

    

    # Check for overfitting (large gap between train and test MSE)

    mse_ratio = test_mse / train_mse if train_mse > 0 else float('inf')

    if mse_ratio > 2.0:

        print(f"  ⚠️  Overfitting detected: Test MSE is {mse_ratio:.1f}x higher than train MSE")

        print(f"     Consider using --head_type frozen_mlp for better generalization")

    

    results = {

        'train_mae': mean_absolute_error(y_train, y_pred_train),

        'train_rmse': np.sqrt(train_mse),

        'train_r2': r2_score(y_train, y_pred_train),

        'train_mse': train_mse,

        'train_pearson': train_pearson,

        'test_mae': mean_absolute_error(y_test, y_pred_test),

        'test_rmse': np.sqrt(test_mse),

        'test_r2': r2_score(y_test, y_pred_test),

        'test_mse': test_mse,

        'test_pearson': test_pearson,

        'best_alpha': grid_search.best_params_['alpha'],

        'mse_ratio': mse_ratio

    }

    

    return model, scaler, y_pred_test, results





def train_sex_classifier(X_train, y_train, X_test, y_test, use_feature_selection=False):

    """Train and evaluate sex classification model with hyperparameter search"""

    from sklearn.model_selection import GridSearchCV

    from sklearn.feature_selection import SelectKBest, f_classif

    

    # Standardize features

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    

    # Feature selection for sex (separate from age selection)

    if use_feature_selection:

        print("  Performing feature selection for sex classification...")

        n_features = min(500, X_train_scaled.shape[1])

        selector = SelectKBest(score_func=f_classif, k=n_features)

        X_train_selected = selector.fit_transform(X_train_scaled, y_train)

        X_test_selected = selector.transform(X_test_scaled)

        print(f"  Selected {n_features} features from {X_train_scaled.shape[1]} (kept {100*n_features/X_train_scaled.shape[1]:.1f}%)")

        X_train_final = X_train_selected

        X_test_final = X_test_selected

    else:

        X_train_final = X_train_scaled

        X_test_final = X_test_scaled

    

    # Train Logistic Regression with hyperparameter search

    print("Training sex classifier (Logistic Regression with CV)...")

    param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}

    # Increase max_iter to avoid convergence warnings, especially for large datasets like ABCD

    logreg = LogisticRegression(

        max_iter=5000,  # Increased from 1000 to allow convergence on large datasets

        class_weight='balanced',

        solver='lbfgs',  # Explicitly set solver

        tol=1e-4  # Tolerance for convergence

    )

    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

    grid_search.fit(X_train_final, y_train)

    

    model = grid_search.best_estimator_

    print(f"  Best C: {grid_search.best_params_['C']}")

    

    # Predictions

    y_pred_train = model.predict(X_train_final)

    y_pred_test = model.predict(X_test_final)

    y_pred_proba = model.predict_proba(X_test_final)[:, 1]

    

    # Metrics

    results = {

        'train_accuracy': accuracy_score(y_train, y_pred_train),

        'train_balanced_accuracy': balanced_accuracy_score(y_train, y_pred_train),

        'train_f1': f1_score(y_train, y_pred_train),

        'test_accuracy': accuracy_score(y_test, y_pred_test),

        'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),

        'test_f1': f1_score(y_test, y_pred_test),

        'test_auc': roc_auc_score(y_test, y_pred_proba),

        'best_C': grid_search.best_params_['C']

    }

    

    # Confusion matrix

    cm = confusion_matrix(y_test, y_pred_test)

    

    return model, scaler, y_pred_test, y_pred_proba, cm, results





def main():

    parser = argparse.ArgumentParser(description='Evaluate Age and Sex Prediction')

    

    # Model and data

    parser.add_argument('--checkpoint', type=str, default=None,

                        help='Path to pretrained model checkpoint (default: auto-find best checkpoint)')

    parser.add_argument('--dataset', type=str, default=None,

                        help='Dataset name (HCPA, UKB, PPMI, etc.)')

    parser.add_argument('--data_dir', type=str, default=None,

                        help='Path to BOLD directory (default: dataset-specific)')

    parser.add_argument('--demographics_csv', type=str, default=None,

                        help='Path to demographics CSV (default: dataset-specific)')

    parser.add_argument('--dataset_type', type=str, default=None,

                        help='Dataset type for data loader (HCPA or UKB-style, default: infer from --dataset)')

    

    # Evaluation settings

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--max_subjects', type=int, default=None,

                        help='Maximum number of subjects to evaluate (for testing)')

    parser.add_argument('--test_split', type=float, default=0.2,

                        help='Fraction of data to use for testing')

    

    # Output

    parser.add_argument('--feature_type', type=str, default='z_all',

                        choices=['z_mean', 'z_std', 'z_max', 'z_concat', 'z_all'],

                        help='Type of features to extract (z_all=mean+std+max+min is recommended for age prediction)')

    parser.add_argument('--head_type', type=str, default='linear',

                        choices=['linear', 'frozen_mlp'],

                        help='Choose between linear probes (recommended for small datasets) and frozen MLP head')

    parser.add_argument('--min_subjects', type=int, default=50,

                        help='Minimum number of matched subjects required (default: 50, lower for small datasets)')

    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[128, 64],

                        help='Hidden dimensions for frozen MLP head (default: [128, 64] for small datasets)')

    parser.add_argument('--mlp_dropout', type=float, default=0.3,

                        help='Dropout for frozen MLP head (default: 0.3 for better regularization)')

    parser.add_argument('--mlp_lr', type=float, default=5e-4,

                        help='Learning rate for frozen MLP head (default: 5e-4 for more stable training)')

    parser.add_argument('--mlp_weight_decay', type=float, default=1e-3,

                        help='Weight decay for frozen MLP head (default: 1e-3 for stronger regularization)')

    parser.add_argument('--mlp_epochs', type=int, default=200,

                        help='Max epochs for frozen MLP head training')

    parser.add_argument('--mlp_batch_size', type=int, default=64,

                        help='Batch size for frozen MLP head training')

    parser.add_argument('--mlp_patience', type=int, default=20,

                        help='Early stopping patience for MLP head')

    parser.add_argument('--fc_mode', type=str, default='dynamic', choices=['static', 'dynamic', 'hybrid'],

                        help='FC computation mode: static (one FC per subject), dynamic (FC per window), hybrid (weighted combination)')

    parser.add_argument('--hybrid_weight', type=float, default=0.3,

                        help='Weight for dynamic FC in hybrid mode (0.0 = fully static, 1.0 = fully dynamic)')

    parser.add_argument('--n_runs', type=int, default=10,

                        help='Number of evaluation runs with different random seeds (default: 10 for rigorous results)')

    

    args = parser.parse_args()

    

    project_root = Path(__file__).parent.parent

    default_paths = {

        'HCPA': {

            'data_dir': project_root / '../data/hcpa/AAL_116/BOLD',

            'demographics_csv': project_root / '../data/hcpa/HCPA_metadata.csv'

        },

        'UKB': {

            'data_dir': project_root / '../data/ukb/AAL_116/BOLD',

            'demographics_csv': project_root / '../data/ukb/UKB_phenotype_sex_age.csv'

        }

    }

    

    dataset_name = args.dataset or args.dataset_type

    if dataset_name is None:

        parser.error("Please specify --dataset")

    dataset_name = dataset_name.upper()

    

    # Set dataset_type for data loader (HCPA-style vs UKB-style)

    # PPMI etc. use similar structure to one of the main types

    if args.dataset_type is None:

        if dataset_name in ['HCPA']:

            args.dataset_type = 'HCPA'

        elif dataset_name in ['UKB']:

            args.dataset_type = 'UKB'

        else:

            # For other datasets, infer based on available info or default to generic

            args.dataset_type = dataset_name  # Will use generic loader

    

    # Auto-select paths for known datasets, require explicit paths for others

    if dataset_name in default_paths:

        if args.data_dir is None:

            args.data_dir = str(default_paths[dataset_name]['data_dir'].resolve())

            print(f"Auto-selected data_dir: {args.data_dir}")

        if args.demographics_csv is None:

            args.demographics_csv = str(default_paths[dataset_name]['demographics_csv'].resolve())

            print(f"Auto-selected demographics_csv: {args.demographics_csv}")

    else:

        # For unknown datasets, require explicit paths

        # Convert to absolute paths if provided

        if args.data_dir is not None:

            args.data_dir = str(Path(args.data_dir).resolve())

        if args.demographics_csv is not None:

            args.demographics_csv = str(Path(args.demographics_csv).resolve())

        

        if args.data_dir is None or args.demographics_csv is None:

            parser.error(f"Dataset '{dataset_name}' not in default paths. "

                        f"Please specify --data_dir and --demographics_csv")

    

    # Auto-find best checkpoint if not provided

    if args.checkpoint is None:

        if find_best_checkpoint is None:

            print("✗ Could not auto-find checkpoint. Please specify --checkpoint")

            print("  manage_checkpoints.py is not available.")

            return

        print("Auto-finding best checkpoint...")

        try:

            checkpoint_path = find_best_checkpoint(results_dir='results', dataset=dataset_name, quiet=True)

        except Exception as e:

            print(f"⚠️  Auto-find failed: {e}")

            print("  Please specify --checkpoint manually")

            print(f"  Try: python scripts/manage_checkpoints.py list --dataset {dataset_name}")

            return

        if checkpoint_path:

            args.checkpoint = checkpoint_path

            print(f"✓ Auto-selected best checkpoint: {args.checkpoint}\n")

        else:

            print("✗ Could not auto-find checkpoint. Please specify --checkpoint")

            print(f"  Try: python scripts/manage_checkpoints.py list --dataset {dataset_name}")

            return

    

    print("=" * 60)

    print("Age and Sex Prediction Evaluation")

    print("=" * 60)

    print(f"Dataset: {args.dataset_type}")

    print(f"Checkpoint: {args.checkpoint}")

    print(f"Data directory: {args.data_dir}")

    print(f"Demographics CSV: {args.demographics_csv}\n")

    

    # Device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}\n")

    

    # Check checkpoint exists

    if not Path(args.checkpoint).exists():

        print(f"\n✗ Error: Checkpoint not found: {args.checkpoint}")

        print("  Please update --checkpoint path or run:")

        dataset_for_checkpoint = args.dataset if args.dataset else args.dataset_type

        print(f"    python scripts/manage_checkpoints.py list --dataset {dataset_for_checkpoint}")

        return

    

    # Load demographics

    demographics = load_demographics(args.demographics_csv, args.dataset_type)

    

    # Load checkpoint

    print(f"\nLoading checkpoint from {args.checkpoint}...")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    

    # Load model configuration

    if 'config' in checkpoint:

        model_config = checkpoint['config']

    else:

        # Try to infer architecture from state_dict

        print("  ⚠️  No 'config' in checkpoint, inferring architecture from state_dict...")

        state_dict = checkpoint.get('model_state_dict', checkpoint)

        

        # Infer latent_dim from spatial_encoder.input_proj.weight shape

        latent_dim = 128  # Default

        if 'spatial_encoder.input_proj.weight' in state_dict:

            # Shape is [d_latent, 1] or [d_latent, n_input]

            latent_dim = state_dict['spatial_encoder.input_proj.weight'].shape[0]

            print(f"  Inferred latent_dim={latent_dim} from spatial_encoder.input_proj.weight")

        

        # Infer control_dim from episodic_feedback or control_fusion

        control_dim = 64  # Default

        if 'episodic_feedback.input_proj.weight' in state_dict:

            # Shape is [d_episodic, d_latent]

            control_dim = state_dict['episodic_feedback.input_proj.weight'].shape[0]

            print(f"  Inferred episodic_dim={control_dim} from episodic_feedback.input_proj.weight")

        elif 'control_fusion.fusion.4.weight' in state_dict:

            # Shape is [d_control, d_latent]

            control_dim = state_dict['control_fusion.fusion.4.weight'].shape[0]

            print(f"  Inferred control_dim={control_dim} from control_fusion.fusion.4.weight")

        

        # Infer koopman_rank from koopman_core.U shape

        koopman_rank = 16  # Default

        if 'koopman_core.U' in state_dict:

            # Shape is [d_latent, koopman_rank]

            koopman_rank = state_dict['koopman_core.U'].shape[1]

            print(f"  Inferred koopman_rank={koopman_rank} from koopman_core.U")

        

        # Infer n_roi from spatial_encoder.output_proj.0.weight shape

        n_roi = 116  # Default

        if 'spatial_encoder.output_proj.0.weight' in state_dict:

            # Shape is [d_latent, n_roi * something]

            weight_shape = state_dict['spatial_encoder.output_proj.0.weight'].shape

            # Try to infer from the second dimension (usually n_roi * some_factor)

            # Common patterns: n_roi * 256, n_roi * 128, etc.

            # For AAL_116: 116 * 256 = 29696

            # For smaller: 116 * 128 = 14848

            if weight_shape[1] % 256 == 0:

                n_roi = weight_shape[1] // 256

            elif weight_shape[1] % 128 == 0:

                n_roi = weight_shape[1] // 128

            print(f"  Inferred n_roi={n_roi} from spatial_encoder.output_proj.0.weight (shape: {weight_shape})")

        

        # Use inferred/default config

        model_config = {

            'n_roi': n_roi,

            'latent_dim': latent_dim,

            'control_dim': control_dim,

            'episodic_dim': control_dim,  # Usually same as control_dim

            'context_dim': control_dim,  # Usually same as control_dim

            'task_dim': 32,

            'n_classes': 4,

            'graph_type': 'cheb',

            'n_graph_layers': 3,

            'cheb_k': 3,

            'drop_edge_p': 0.1,

            'n_temporal_layers': 3,

            'n_heads': 8,

            'koopman_rank': koopman_rank,

            'target_spectral_radius': 0.95,

            'use_sde': True,

            'diffusion_mode': 'learned',

            'uncertainty_mode': 'heteroscedastic',

            'dropout': 0.1

        }

        print(f"  Using inferred config: latent_dim={latent_dim}, control_dim={control_dim}, koopman_rank={koopman_rank}, n_roi={n_roi}")

    

    # Extract wavelet parameters from config (with backward compatibility)

    use_wavelet_ms = model_config.get('use_wavelet_ms', model_config.get('use_wavelet_branch', False))

    wavelet_levels = model_config.get('wavelet_levels', 3)

    wavelet_dropout = model_config.get('wavelet_dropout', 0.1)

    wavelet_input = model_config.get('wavelet_input', 'raw')

    

    # Build model kwargs, filtering out unsupported parameters

    import inspect

    model_signature = inspect.signature(BrainMech.__init__)

    valid_params = {k for k in model_signature.parameters.keys() if k != 'self'}

    

    model_kwargs = {

        'n_roi': model_config.get('n_roi', 116),

        'd_latent': model_config.get('latent_dim', 128),

        'd_control': model_config.get('control_dim', 64),

        'd_episodic': model_config.get('episodic_dim', 64),

        'd_context': model_config.get('context_dim', 64),

        'd_task': model_config.get('task_dim', 32),

        'n_classes': model_config.get('n_classes', 4),

        'graph_type': model_config.get('graph_type', 'cheb'),

        'n_graph_layers': model_config.get('n_graph_layers', 3),

        'cheb_k': model_config.get('cheb_k', 3),

        'drop_edge_p': model_config.get('drop_edge_p', 0.1),

        'n_temporal_layers': model_config.get('n_temporal_layers', 3),

        'n_heads': model_config.get('n_heads', 8),

        'koopman_rank': model_config.get('koopman_rank', 16),

        'spectral_radius': model_config.get('target_spectral_radius', 0.95),

        'use_sde': model_config.get('use_sde', True),

        'diffusion_mode': model_config.get('diffusion_mode', 'learned'),

        'uncertainty_mode': model_config.get('uncertainty_mode', 'heteroscedastic'),

        'dropout': model_config.get('dropout', 0.1),

    }

    

    # Only add wavelet parameters if the model supports them

    if 'use_wavelet_ms' in valid_params:

        model_kwargs.update({

            'use_wavelet_ms': use_wavelet_ms,

            'wavelet_levels': wavelet_levels,

            'wavelet_dropout': wavelet_dropout,

            'wavelet_input': wavelet_input

        })

    elif use_wavelet_ms:

        print(f"  ⚠️  Warning: Checkpoint has use_wavelet_ms=True, but model doesn't support it. Ignoring wavelet parameters.")

    

    # Filter to only valid parameters (in case of version mismatch)

    model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}

    

    # Create model

    model = BrainMech(**model_kwargs).to(device)

    

    # Load weights with error handling for architecture mismatches

    try:

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if missing_keys:

            print(f"  Note: {len(missing_keys)} layers missing (will use random initialization)")

        if unexpected_keys:

            print(f"  Note: {len(unexpected_keys)} unexpected keys in checkpoint")

        model.eval()

        print("✓ Model loaded successfully\n")

    except RuntimeError as e:

        if "size mismatch" in str(e):

            print(f"  ⚠️  Architecture mismatch detected. Attempting to load compatible layers only...")

            # Try to load compatible layers

            state_dict = checkpoint['model_state_dict']

            model_state_dict = model.state_dict()

            compatible_state_dict = {}

            for key, value in state_dict.items():

                if key in model_state_dict:

                    if value.shape == model_state_dict[key].shape:

                        compatible_state_dict[key] = value

            model.load_state_dict(compatible_state_dict, strict=False)

            print(f"  ✓ Loaded {len(compatible_state_dict)}/{len(state_dict)} compatible layers")

            print("  ⚠️  Warning: Some layers will use random initialization due to architecture mismatch")

            model.eval()

        else:

            raise

    

    # Load BOLD data

    print("Loading BOLD data...")

    if args.dataset_type.upper() == 'UKB':

        bold_dir = Path(args.data_dir)

        fc_dir = bold_dir.parent / 'FC'

        if fc_dir.exists():

            # Build kwargs conditionally to handle optional parameters

            dataset_kwargs = {

                'bold_dir': args.data_dir,

                'sequence_length': 100,

                'stride': 100,  # Non-overlapping for evaluation

                'normalize': True,

                'max_files': args.max_subjects,

            }

            # Try to add fc_mode and hybrid_weight if they exist in args

            # If dataset class doesn't support them, retry without them

            if hasattr(args, 'fc_mode') and args.fc_mode is not None:

                dataset_kwargs['fc_mode'] = args.fc_mode

            if hasattr(args, 'hybrid_weight') and args.hybrid_weight is not None:

                dataset_kwargs['hybrid_weight'] = args.hybrid_weight

            

            try:

                dataset = UKBiobankBOLDWithFCDataset(**dataset_kwargs)

            except TypeError as e:

                # If dataset class doesn't support fc_mode/hybrid_weight, retry without them

                if 'fc_mode' in str(e) or 'hybrid_weight' in str(e) or 'unexpected keyword' in str(e):

                    print(f"⚠️  Dataset class doesn't support fc_mode/hybrid_weight, using defaults")

                    dataset_kwargs.pop('fc_mode', None)

                    dataset_kwargs.pop('hybrid_weight', None)

                    dataset = UKBiobankBOLDWithFCDataset(**dataset_kwargs)

                else:

                    raise

        else:

            dataset = UKBiobankBOLDDataset(

                data_dir=args.data_dir,

                sequence_length=100,

                stride=100,

                normalize=True,

                max_files=args.max_subjects

            )

    elif args.dataset_type.upper() == 'HCPA':

        dataset = HCPABOLDDataset(

            data_dir=args.data_dir,

            sequence_length=100,

            stride=100,  # Non-overlapping for evaluation

            normalize=True,

            max_files=args.max_subjects

        )

    else:

        # Try to use dataset registry from disease_diagnosis for other datasets

        use_custom_collate = False

        try:

            from eval.disease_diagnosis import DATASET_REGISTRY, custom_collate_fn

            # Special handling for ABCD - it uses bold_dir instead of data_dir

            if args.dataset_type.upper() == 'ABCD':

                from utils.data_loader import ABCDBOLDWithFCDataset

                dataset = ABCDBOLDWithFCDataset(

                    bold_dir=args.data_dir,  # Use bold_dir, not data_dir

                    fc_dir=None,  # Will auto-detect

                    sequence_length=100,

                    stride=100,

                    normalize=True,

                    task_filter=['REST'],

                    max_files=args.max_subjects

                )

                print(f"  Using ABCD dataset from data_loader")

                use_custom_collate = False  # ABCDBOLDWithFCDataset uses standard collate

            elif args.dataset_type.upper() in DATASET_REGISTRY:

                dataset_class = DATASET_REGISTRY[args.dataset_type.upper()]

                dataset = dataset_class(

                    data_dir=args.data_dir,

                    labels_file=None,  # Age/sex evaluation doesn't need labels

                    sequence_length=100,

                    stride=100,

                    normalize=True,

                    max_files=args.max_subjects

                )

                print(f"  Using {args.dataset_type.upper()} dataset from disease_diagnosis registry")

                use_custom_collate = True  # Use custom collate for non-standard datasets

            else:

                raise ValueError(f"Dataset type '{args.dataset_type}' not found in registry")

        except ImportError as e:

            print(f"  ⚠️  Could not import disease_diagnosis registry: {e}")

            # Fall back to HCPA-style loader as generic

            print(f"  Falling back to HCPA-style loader for {args.dataset_type}")

            dataset = HCPABOLDDataset(

                data_dir=args.data_dir,

                sequence_length=100,

                stride=100,

                normalize=True,

                max_files=args.max_subjects

            )

    

    # Create DataLoader with appropriate settings

    # For non-standard datasets (PPMI etc.), use num_workers=0 and custom collate

    # to avoid tensor size mismatch issues

    if args.dataset_type.upper() not in ['HCPA', 'UKB']:

        try:

            from eval.disease_diagnosis import custom_collate_fn

            from functools import partial

            # Get expected ROI from model config (default 116 for AAL_116)

            expected_roi = 116  # Default for AAL_116

            if hasattr(model, 'n_roi'):

                expected_roi = model.n_roi

            elif hasattr(model, 'config') and isinstance(model.config, dict) and 'n_roi' in model.config:

                expected_roi = model.config['n_roi']

            elif hasattr(model, 'spatial_encoder') and hasattr(model.spatial_encoder, 'n_roi'):

                expected_roi = model.spatial_encoder.n_roi

            collate_fn = partial(custom_collate_fn, expected_roi=expected_roi)

            dataloader = DataLoader(

                dataset,

                batch_size=args.batch_size,

                shuffle=False,

                num_workers=0,  # Avoid multiprocessing issues with variable-size tensors

                collate_fn=collate_fn

            )

            print(f"  Using custom collate function for {args.dataset_type}")

        except ImportError:

            dataloader = DataLoader(

                dataset,

                batch_size=args.batch_size,

                shuffle=False,

                num_workers=0

            )

    else:

        dataloader = DataLoader(

            dataset,

            batch_size=args.batch_size,

            shuffle=False,

            num_workers=args.num_workers

        )

    

    # Check if dataset is empty

    if len(dataset) == 0:

        raise ValueError(

            f"Dataset is empty! No data loaded from {args.data_dir}. "

            f"Possible issues:\n"

            f"  1. Data directory structure doesn't match expected format\n"

            f"  2. No files found matching expected patterns\n"

            f"  3. All files were filtered out or failed to load\n"

            f"  For {args.dataset_type}, check:\n"

            f"    - Data directory: {args.data_dir}\n"

            f"    - Expected file format (CSV, TXT, or MAT files)\n"

            f"    - File naming conventions"

        )

    

    print(f"✓ Dataset loaded: {len(dataset)} samples")

    

    # Check cache for features

    model_params = get_model_params(model)

    cache_key = get_cache_key(

        checkpoint_path=args.checkpoint,

        dataset_path=args.data_dir,

        dataset_type=args.dataset_type,

        feature_type=args.feature_type,

        sequence_length=100,  # Fixed for age/sex evaluation

        stride=100,  # Fixed for age/sex evaluation

        **model_params

    )

    

    cached_data = load_cached_features(cache_key)

    if cached_data is not None:

        if len(cached_data) == 4:

            features, subjects, _, eventnames = cached_data

        else:

            features, subjects = cached_data[:2]

            eventnames = None

        print(f"✓ Loaded features from cache: {features.shape}")

    else:

        # Extract features

        print("  (Cache miss - extracting features from model...)")

        result = extract_features(model, dataloader, device, feature_type=args.feature_type)

        if len(result) == 3:

            features, subjects, eventnames = result

        else:

            features, subjects = result[:2]

            eventnames = None

        # Save to cache

        save_cached_features(cache_key, features, subjects, eventnames=eventnames)

    

    # Create feature DataFrame

    print("\nMatching features with demographics...")

    feature_df = pd.DataFrame(features)

    

    # Ensure all feature columns are numeric (convert any non-numeric to NaN, then drop)

    for col in feature_df.columns:

        try:

            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

        except:

            # If conversion fails, drop the column

            print(f"  ⚠️  Warning: Dropping non-numeric column '{col}' from features")

            feature_df = feature_df.drop(columns=[col])

    

    # Drop any columns that are all NaN

    feature_df = feature_df.dropna(axis=1, how='all')

    

    # Clean subject IDs - extract ID from filenames

    # Handle formats like:

    # - "sub-HCA6002236_ses-2_task-hariri_..." -> "HCA6002236"

    # - "sub-1000026_ses-2_task-hariri_..." -> "1000026"

    # - "1000026.csv" -> "1000026"

    # - "HCA6002236" -> "HCA6002236"

    # - UKB format: "1234567.csv" -> "1234567"

    # - PPMI format: "sub-control032057" or "sub-patient032057" -> "032057"

    import re

    cleaned_subjects = []

    for s in subjects:

        s_str = str(s)

        # Remove file extension first

        s_str = s_str.replace('.csv', '').replace('.npy', '').replace('.txt', '')

        

        # ADNI format: "002_S_0295" - keep as-is (matches metadata format exactly)

        adni_match = re.match(r'^\d{3}_S_\d{4}$', s_str)

        if adni_match:

            cleaned_subjects.append(s_str)

            continue

        

        # Special handling for PPMI: sub-control032057 or sub-patient032057

        # Extract just the numeric ID at the end (may have leading zeros like "032057")

        # Try multiple patterns to handle various folder naming conventions

        subject_id_patterns = [

            r'sub-(?:control|patient|prodromal|swedd)[\(\)]?(?:control|patient)?[\(\)]?(\d+)',  # sub-control032057

            r'sub-(?:control|patient)[\(\)]?(\d+)',  # sub-control(032057) or sub-control032057

            r'(?:control|patient)(\d+)',  # control032057

            r'(\d{6})',  # Just 6-digit number (032057)

        ]

        for pattern in subject_id_patterns:

            id_match = re.search(pattern, s_str, re.IGNORECASE)

            if id_match:

                cleaned_subjects.append(id_match.group(1))

                break

        else:

            # If no pattern matched, try to extract any numeric ID

            numeric_match = re.search(r'(\d+)', s_str)

            if numeric_match:

                cleaned_subjects.append(numeric_match.group(1))

            else:

                cleaned_subjects.append(s_str)

        continue

        

        # Try to extract full ID after "sub-" (including any prefix like HCA, case-insensitive)

        match = re.search(r'sub-([A-Za-z0-9]+)', s_str, re.IGNORECASE)

        if match:

            # Check if it's just a number (standard format)

            extracted = match.group(1)

            # If it starts with control/patient prefix, extract number

            prefix_match = re.match(r'(?:control|patient|prodromal|swedd)(\d+)', extracted, re.IGNORECASE)

            if prefix_match:

                cleaned_subjects.append(prefix_match.group(1))

            else:

                cleaned_subjects.append(extracted)

        else:

            # For UKB or other formats: extract first numeric or alphanumeric segment

            # Try pure numeric first (UKB style: "1234567")

            match = re.search(r'^(\d+)', s_str)

            if match:

                cleaned_subjects.append(match.group(1))

            else:

                # Try alphanumeric (HCA style without sub- prefix)

                match = re.search(r'([A-Za-z0-9]+)', s_str)

                if match:

                    cleaned_subjects.append(match.group(1))

                else:

                    # Keep as is

                    cleaned_subjects.append(s_str)

    

    feature_df['subject_id'] = cleaned_subjects

    

    # For ABCD: add eventname column if available

    if eventnames and any(e is not None for e in eventnames):

        feature_df['eventname'] = [str(e).lower().strip() if e is not None else None for e in eventnames]

        print(f"  Found eventname info for {sum(1 for e in eventnames if e is not None)} samples")

    

    # IMPORTANT: Aggregate multiple time windows per subject

    # For ABCD: aggregate by (subject_id, eventname) to match metadata

    # For other datasets: aggregate by subject_id only

    if args.dataset_type.upper() == 'ABCD' and 'eventname' in feature_df.columns:

        print(f"Found {len(feature_df)} time windows from {feature_df['subject_id'].nunique()} unique subjects")

        print("Aggregating features by (subject_id, eventname) for ABCD matching...")

        feature_cols = [col for col in feature_df.columns if col not in ['subject_id', 'eventname']]

        feature_df_agg = feature_df.groupby(['subject_id', 'eventname'])[feature_cols].mean().reset_index()

        print(f"✓ Aggregated to {len(feature_df_agg)} subject-eventname pairs\n")

    else:

        # Each subject may have multiple time windows - average their features

        print(f"Found {len(feature_df)} time windows from {feature_df['subject_id'].nunique()} unique subjects")

        print("Aggregating features by subject (averaging multiple time windows)...")

        feature_cols = [col for col in feature_df.columns if col != 'subject_id']

        feature_df_agg = feature_df.groupby('subject_id')[feature_cols].mean().reset_index()

        print(f"✓ Aggregated to {len(feature_df_agg)} subjects\n")

    

    # Try multiple matching strategies for better subject ID matching

    print("Attempting subject ID matching...")

    

    # Strategy 1: Direct merge (with eventname for ABCD)

    if args.dataset_type.upper() == 'ABCD' and 'eventname' in feature_df_agg.columns and 'eventname' in demographics.columns:

        # ABCD: match on both subject_id and eventname

        merged_df = feature_df_agg.merge(

            demographics[['subject_id', 'eventname', 'age', 'sex']], 

            on=['subject_id', 'eventname'], 

            how='inner'

        )

        print(f"  Direct merge (subject_id + eventname): {len(merged_df)} matches")

    else:

        # Other datasets: match on subject_id only

        merged_df = feature_df_agg.merge(demographics, on='subject_id', how='inner')

        print(f"  Direct merge: {len(merged_df)} matches")

    

    # Strategy 2: If direct merge fails, try normalized matching (remove leading zeros, case-insensitive)

    if len(merged_df) < 10:

        print("  Direct merge insufficient, trying normalized matching...")

        feature_df_agg_norm = feature_df_agg.copy()

        demographics_norm = demographics.copy()

        

        # Normalize IDs: remove leading zeros, lowercase, strip

        feature_df_agg_norm['subject_id_norm'] = feature_df_agg_norm['subject_id'].astype(str).str.lstrip('0').str.lower().str.strip()

        if 'subject_id_norm' in demographics_norm.columns:

            # Use existing normalized column if available

            pass

        else:

            demographics_norm['subject_id_norm'] = demographics_norm['subject_id'].astype(str).str.lstrip('0').str.lower().str.strip()

        

        # Try merge on normalized IDs - include all columns from both dataframes

        merged_df_norm = feature_df_agg_norm.merge(

            demographics_norm[['subject_id_norm', 'subject_id', 'age', 'sex']], 

            on='subject_id_norm', 

            how='inner',

            suffixes=('', '_demo')

        )

        if len(merged_df_norm) > len(merged_df):

            # Use the merged dataframe directly (it already has subject_id, age, sex, and features)

            # Drop the normalized column and any duplicate subject_id columns

            if 'subject_id_norm' in merged_df_norm.columns:

                merged_df_norm = merged_df_norm.drop(columns=['subject_id_norm'])

            if 'subject_id_demo' in merged_df_norm.columns:

                merged_df_norm = merged_df_norm.drop(columns=['subject_id_demo'])

            merged_df = merged_df_norm.copy()

            print(f"  ✓ Normalized matching found {len(merged_df)} matches")

    

    # Strategy 3: Partial matching (for datasets where folder names might have prefixes)

    if len(merged_df) < 10:

        print("  Trying partial matching...")

        feature_ids = set(feature_df_agg['subject_id'].astype(str))

        demo_ids = set(demographics['subject_id'].astype(str))

        

        # Try to find matches where one ID contains the other (normalized)

        matches = []

        for fid in feature_ids:

            fid_clean = str(fid).lstrip('0').lower().strip()

            for did in demo_ids:

                did_clean = str(did).lstrip('0').lower().strip()

                if fid_clean == did_clean:

                    matches.append((fid, did))

                    break

                # Also try if one contains the other (for cases like "032057" vs "32057")

                elif fid_clean and did_clean and (fid_clean in did_clean or did_clean in fid_clean):

                    matches.append((fid, did))

                    break

        

        if matches:

            # Create mapping and merge

            id_mapping = {fid: did for fid, did in matches}

            feature_df_agg_mapped = feature_df_agg.copy()

            feature_df_agg_mapped['subject_id_mapped'] = feature_df_agg_mapped['subject_id'].astype(str).map(

                lambda x: id_mapping.get(x, x)

            )

            merged_df_mapped = feature_df_agg_mapped.merge(

                demographics, 

                left_on='subject_id_mapped', 

                right_on='subject_id', 

                how='inner'

            )

            if len(merged_df_mapped) > len(merged_df):

                merged_df = merged_df_mapped.drop(columns=['subject_id_mapped'])

                print(f"  ✓ Partial matching found {len(merged_df)} matches")

    

    # Compute match statistics

    n_unique_features = feature_df_agg['subject_id'].nunique()

    n_unique_demo = demographics['subject_id'].nunique()

    match_rate_features = len(merged_df) / n_unique_features * 100 if n_unique_features > 0 else 0

    match_rate_demo = len(merged_df) / n_unique_demo * 100 if n_unique_demo > 0 else 0

    

    print(f"✓ Matched {len(merged_df)} subjects with demographics")

    print(f"  Features: {n_unique_features} unique subjects ({match_rate_features:.1f}% matched)")

    print(f"  Demographics: {n_unique_demo} unique subjects ({match_rate_demo:.1f}% matched)\n")

    

    if match_rate_features < 50 or match_rate_demo < 50:

        print("⚠️  WARNING: Low match rate detected! Possible subject ID format mismatch.")

        print(f"   Example feature subject IDs (original): {subjects[:5]}")

        print(f"   Example feature subject IDs (cleaned): {cleaned_subjects[:5]}")

        print(f"   Example demo subject IDs: {demographics['subject_id'].head().tolist()}")

        

        # Try to diagnose the mismatch

        feature_ids_set = set(feature_df_agg['subject_id'].astype(str))

        demo_ids_set = set(demographics['subject_id'].astype(str))

        

        # Check for common patterns

        feature_sample = list(feature_ids_set)[:3]

        demo_sample = list(demo_ids_set)[:3]

        print(f"\n   Feature ID samples: {feature_sample}")

        print(f"   Demo ID samples: {demo_sample}")

        

        # Check if one set has prefixes the other doesn't

        if any(fid.startswith('sub-') for fid in feature_sample) and not any(did.startswith('sub-') for did in demo_sample):

            print("   → Feature IDs have 'sub-' prefix, demographics don't")

        elif not any(fid.startswith('sub-') for fid in feature_sample) and any(did.startswith('sub-') for did in demo_sample):

            print("   → Demographics have 'sub-' prefix, feature IDs don't")

    

    # Get minimum subjects from args (defaults to 50)

    min_subjects = getattr(args, 'min_subjects', 50)

    

    if len(merged_df) < min_subjects:

        raise ValueError(f"Subject ID mismatch - only {len(merged_df)} subjects matched (need at least {min_subjects})")

    

    # Prepare data

    # Explicitly exclude subject_id, age, and sex columns, and ensure all are numeric

    feature_cols = [col for col in merged_df.columns if col not in ['subject_id', 'age', 'sex']]

    

    # Ensure all feature columns are numeric (convert to float, drop any that can't be converted)

    X_df = merged_df[feature_cols].copy()

    for col in feature_cols:

        try:

            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

        except:

            # If conversion fails, drop the column

            print(f"  ⚠️  Warning: Dropping non-numeric column '{col}' from features")

            X_df = X_df.drop(columns=[col])

    

    # Drop any rows with all NaN features

    X_df = X_df.dropna(axis=1, how='all')

    

    # Convert to numpy array

    X = X_df.values.astype(np.float32)

    y_age = merged_df['age'].values.astype(np.float32)

    y_sex = merged_df['sex'].values.astype(np.int64)

    

    # Multi-run evaluation with aggregation

    def aggregate_metrics(run_results_list, metric_keys):

        """Aggregate numeric metrics across runs"""

        aggregated = {}

        for key in metric_keys:

            values = [r[key] for r in run_results_list if r.get(key) is not None and not np.isnan(r[key])]

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

    

    # Store results from all runs

    all_runs_age_results = []

    all_runs_sex_results = []

    

    # Run evaluation N times with different random seeds

    for run_idx in range(args.n_runs):

        random_seed = 42 + run_idx  # Different seed for each run

        print(f"\n{'='*70}")

        print(f"RUN {run_idx + 1}/{args.n_runs} (random_seed={random_seed})")

        print(f"{'='*70}")

        

        # Split data with different random seed for each run

        from sklearn.model_selection import train_test_split

        indices = np.arange(len(X))

        train_idx, test_idx = train_test_split(indices, test_size=args.test_split, random_state=random_seed, stratify=y_sex)

        

        X_train, X_test = X[train_idx], X[test_idx]

        y_age_train, y_age_test = y_age[train_idx], y_age[test_idx]

        y_sex_train, y_sex_test = y_sex[train_idx], y_sex[test_idx]

        

        if run_idx == 0:

            print(f"Train set: {len(X_train)} subjects")

            print(f"Test set: {len(X_test)} subjects")

            print(f"Age distribution - Train: [{y_age_train.min():.1f}, {y_age_train.max():.1f}] years, mean={y_age_train.mean():.1f}±{y_age_train.std():.1f}")

            print(f"Age distribution - Test:  [{y_age_test.min():.1f}, {y_age_test.max():.1f}] years, mean={y_age_test.mean():.1f}±{y_age_test.std():.1f}")

            

            # Compute baseline: predicting mean age for everyone

            baseline_mse = np.mean((y_age_test - y_age_train.mean()) ** 2)

            baseline_mae = np.mean(np.abs(y_age_test - y_age_train.mean()))

            print(f"\n📊 BASELINE (predict mean age {y_age_train.mean():.1f} for all):")

            print(f"   Baseline MSE: {baseline_mse:.2f} years²  (RMSE: {np.sqrt(baseline_mse):.2f} years)")

            print(f"   Baseline MAE: {baseline_mae:.2f} years")

            print(f"   → Model must beat this baseline to be useful!\n")

        

        # Train age predictor

        if run_idx == 0:

            print("=" * 60)

            print("AGE PREDICTION")

            print("=" * 60)

        if args.head_type == 'linear':

            age_model, age_scaler, y_age_pred, age_results = train_age_predictor(

                X_train, y_age_train, X_test, y_age_test

            )

        else:

            age_results = train_regression_head(

                X_train,

                y_age_train,

                X_test,

                y_age_test,

                hidden_dims=args.mlp_hidden_dims,

                dropout=args.mlp_dropout,

                lr=args.mlp_lr,

                weight_decay=args.mlp_weight_decay,

                batch_size=args.mlp_batch_size,

                epochs=args.mlp_epochs,

                patience=args.mlp_patience,

                device=device,

            )

        

        if run_idx == 0:

            print("\nAge Prediction Results (Run 1):")

            print(f"  Train MSE: {age_results['train_mse']:.3f} years²")

            print(f"  Train Pearson ρ: {age_results['train_pearson']:.3f}")

            print(f"  Test MSE: {age_results['test_mse']:.3f} years²")

            print(f"  Test Pearson ρ: {age_results['test_pearson']:.3f}")

        else:

            print(f"  Run {run_idx + 1}: Test MSE={age_results['test_mse']:.3f}, Pearson ρ={age_results['test_pearson']:.3f}")

        

        # Store age results

        all_runs_age_results.append(age_results)

        

        # Train sex classifier

        if run_idx == 0:

            print("\n" + "=" * 60)

            print("SEX CLASSIFICATION")

            print("=" * 60)

        if args.head_type == 'linear':

            sex_model, sex_scaler, y_sex_pred, y_sex_proba, _, sex_results = train_sex_classifier(

                X_train, y_sex_train, X_test, y_sex_test, use_feature_selection=True

            )

        else:

            sex_results = train_classification_head(

                X_train,

                y_sex_train,

                X_test,

                y_sex_test,

                hidden_dims=args.mlp_hidden_dims,

                dropout=args.mlp_dropout,

                lr=args.mlp_lr,

                weight_decay=args.mlp_weight_decay,

                batch_size=args.mlp_batch_size,

                epochs=args.mlp_epochs,

                patience=args.mlp_patience,

                device=device,

            )

        

        if run_idx == 0:

            print("\nSex Classification Results (Run 1):")

            print(f"  Train Accuracy: {sex_results['train_accuracy'] * 100:.2f}%")

            print(f"  Train F1: {sex_results['train_f1'] * 100:.2f}%")

            print(f"  Test Accuracy: {sex_results['test_accuracy'] * 100:.2f}%")

            print(f"  Test F1: {sex_results['test_f1'] * 100:.2f}%")

        else:

            print(f"  Run {run_idx + 1}: Test Acc={sex_results['test_accuracy'] * 100:.2f}%, F1={sex_results['test_f1'] * 100:.2f}%")

        

        # Store sex results

        all_runs_sex_results.append(sex_results)

    

    # Aggregate results across all runs

    print("\n" + "=" * 70)

    print("AGGREGATED RESULTS (Mean ± Std across all runs)")

    print("=" * 70)

    

    age_metric_keys = ['train_mse', 'train_pearson', 'test_mse', 'test_pearson', 'test_mae', 'test_r2']

    sex_metric_keys = ['train_accuracy', 'train_f1', 'test_accuracy', 'test_f1', 'test_balanced_accuracy', 'test_auc']

    

    aggregated_age = aggregate_metrics(all_runs_age_results, age_metric_keys)

    aggregated_sex = aggregate_metrics(all_runs_sex_results, sex_metric_keys)

    

    print("\nAge Prediction (n={} runs):".format(args.n_runs))

    print(f"  Train MSE: {aggregated_age['train_mse']:.3f} ± {aggregated_age['train_mse_std']:.3f} years²")

    print(f"  Train Pearson ρ: {aggregated_age['train_pearson']:.3f} ± {aggregated_age['train_pearson_std']:.3f}")

    print(f"  Test MSE: {aggregated_age['test_mse']:.3f} ± {aggregated_age['test_mse_std']:.3f} years²")

    print(f"  Test Pearson ρ: {aggregated_age['test_pearson']:.3f} ± {aggregated_age['test_pearson_std']:.3f}")

    if 'test_mae' in aggregated_age and aggregated_age['test_mae'] is not None:

        print(f"  Test MAE: {aggregated_age['test_mae']:.3f} ± {aggregated_age['test_mae_std']:.3f} years")

    

    print("\nSex Classification (n={} runs):".format(args.n_runs))

    print(f"  Train Accuracy: {aggregated_sex['train_accuracy'] * 100:.2f}% ± {aggregated_sex['train_accuracy_std'] * 100:.2f}%")

    print(f"  Train F1: {aggregated_sex['train_f1'] * 100:.2f}% ± {aggregated_sex['train_f1_std'] * 100:.2f}%")

    print(f"  Test Accuracy: {aggregated_sex['test_accuracy'] * 100:.2f}% ± {aggregated_sex['test_accuracy_std'] * 100:.2f}%")

    print(f"  Test F1: {aggregated_sex['test_f1'] * 100:.2f}% ± {aggregated_sex['test_f1_std'] * 100:.2f}%")

    if 'test_balanced_accuracy' in aggregated_sex and aggregated_sex['test_balanced_accuracy'] is not None:

        print(f"  Test Balanced Acc: {aggregated_sex['test_balanced_accuracy'] * 100:.2f}% ± {aggregated_sex['test_balanced_accuracy_std'] * 100:.2f}%")

    

    print("\n" + "=" * 60)

    print("Evaluation Complete!")

    print("=" * 60)

    print(f"\nSummary (Mean ± Std, n={args.n_runs} runs):")

    print(f"  Age Test MSE: {aggregated_age['test_mse']:.3f} ± {aggregated_age['test_mse_std']:.3f} years²")

    print(f"  Age Test Pearson ρ: {aggregated_age['test_pearson']:.3f} ± {aggregated_age['test_pearson_std']:.3f}")

    print(f"  Sex Test Accuracy: {aggregated_sex['test_accuracy'] * 100:.2f}% ± {aggregated_sex['test_accuracy_std'] * 100:.2f}%")

    print(f"  Sex Test F1: {aggregated_sex['test_f1'] * 100:.2f}% ± {aggregated_sex['test_f1_std'] * 100:.2f}%")





if __name__ == '__main__':

    main()




"""
Fine-tuning Script for Disease Diagnosis on PPMI, ADNI, and ABIDE

This script fine-tunes the entire BrainMech Foundation Model (backbone + head) on disease datasets.
Key differences from frozen evaluation:
- Fine-tunes the ENTIRE model (backbone + classification head)
- Uses lower learning rate for stable fine-tuning
- K-fold cross-validation
- Early stopping based on validation F1

Usage:
    # Fine-tune on ABIDE (ASD vs Control)
    python eval/finetune_disease.py --checkpoint results/best_model.pt --dataset ABIDE
    
    # Fine-tune on PPMI (4-class)
    python eval/finetune_disease.py --checkpoint results/best_model.pt --dataset PPMI
    
    # Fine-tune on ADNI (CN vs AD)
    python eval/finetune_disease.py --checkpoint results/best_model.pt --dataset ADNI
    
    # Fine-tune on all three
    python eval/finetune_disease.py --checkpoint results/best_model.pt --dataset ALL
"""

import sys
import os
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import argparse
from datetime import datetime
from tqdm import tqdm, trange
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    precision_recall_fscore_support
)
import warnings
warnings.filterwarnings("ignore")

# Suppress mamba warning during import (from cached .pyc files)
_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from eval.evaluate_downstream import load_model
finally:
    sys.stdout = _stdout
from eval.disease_diagnosis import (
    PPMIDatasetNested, ADNIDataset, ABIDEDataset,
    custom_collate_fn
)
from functools import partial

# Dataset registry for this script
FINETUNE_DATASET_REGISTRY = {
    'ABIDE': ABIDEDataset,
    'ADNI': ADNIDataset,
    'PPMI': PPMIDatasetNested,
}

# Dataset configurations
DATASET_CONFIG = {
    'ADNI': {
        'n_classes': 2,
        'class_names': ['CN-group', 'AD-group'],
        'data_dir': '../data/adni/BOLD',
        'labels_file': '../data/adni/metadata.csv',
        'n_folds': 5,
    },
    'ABIDE': {
        'n_classes': 2,
        'class_names': ['Control', 'ASD'],
        'data_dir': '../data/abide',
        'labels_file': '../data/abide/labels.csv',
        'n_folds': 10,
    },
    'PPMI': {
        'n_classes': 4,
        'class_names': ['Control', 'SWEDD', 'Prodromal', 'PD'],
        'data_dir': '../data/ppmi',
        'labels_file': '../data/ppmi/labels.csv',
        'n_folds': 10,
    }
}


class FineTuneClassifier(nn.Module):
    """
    Classification head for fine-tuning.
    Similar architecture to their BNDecoder but simpler.
    """
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.classifier = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.classifier(x)


class BrainMechFineTuner(nn.Module):
    """
    Wrapper that combines BrainMech backbone with classification head for fine-tuning.
    """
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        feature_type: str = 'z_concat',
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.feature_type = feature_type
        self.freeze_backbone = freeze_backbone
        
        # Determine input dimension based on feature type
        if hasattr(backbone, 'd_latent'):
            latent_dim = backbone.d_latent
        elif hasattr(backbone, 'base_model') and hasattr(backbone.base_model, 'd_latent'):
            latent_dim = backbone.base_model.d_latent
        else:
            latent_dim = 256  # Default
        
        if feature_type == 'z_concat':
            input_dim = latent_dim * 2  # mean + std
        elif feature_type == 'z_all':
            input_dim = latent_dim * 4  # mean + std + max + min
        else:
            input_dim = latent_dim
        
        self.classifier = FineTuneClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def extract_features(self, x, fc_matrix=None):
        """Extract features from backbone"""
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.backbone(x, task_vectors=None, fc_matrix=fc_matrix, return_uncertainty=False)
            z = outputs['z_latent']  # [B, T, d_latent]
            
            if self.feature_type == 'z_mean':
                features = z.mean(dim=1)
            elif self.feature_type == 'z_std':
                features = z.std(dim=1)
            elif self.feature_type == 'z_concat':
                z_mean = z.mean(dim=1)
                z_std = z.std(dim=1)
                features = torch.cat([z_mean, z_std], dim=-1)
            elif self.feature_type == 'z_all':
                z_mean = z.mean(dim=1)
                z_std = z.std(dim=1)
                z_max = z.max(dim=1)[0]
                z_min = z.min(dim=1)[0]
                features = torch.cat([z_mean, z_std, z_max, z_min], dim=-1)
            else:
                features = z.mean(dim=1)
        
        return features
    
    def forward(self, x, fc_matrix=None):
        features = self.extract_features(x, fc_matrix)
        logits = self.classifier(features)
        return logits


def get_subject_labels(dataset, binary_mapping=None):
    """Extract subject IDs and labels from dataset"""
    subjects = []
    labels = []
    
    for i in range(len(dataset)):
        _, meta = dataset[i]
        subjects.append(meta.get('subject', str(i)))
        label = meta.get('label', meta.get('labels', 0))
        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = 0
        label = int(label)
        
        labels.append(label)
    
    return np.array(subjects), np.array(labels)


def create_fold_dataloaders(
    dataset,
    subjects,
    labels,
    unique_subjects,
    train_idx,
    val_idx,
    batch_size: int = 32,
    expected_roi: int = 116,
):
    """
    Create train and validation dataloaders for a fold.
    
    Args:
        dataset: Full dataset
        subjects: Array of subject IDs for each sample (length = num_samples)
        labels: Array of labels for each sample (length = num_samples)
        unique_subjects: Array of unique subject IDs (length = num_unique_subjects)
        train_idx: Indices into unique_subjects for training subjects
        val_idx: Indices into unique_subjects for validation subjects
        batch_size: Batch size
        expected_roi: Expected number of ROIs
    """
    train_subjects = set(unique_subjects[train_idx])
    val_subjects = set(unique_subjects[val_idx])
    
    train_sample_idx = [i for i in range(len(dataset)) 
                        if dataset[i][1].get('subject', str(i)) in train_subjects]
    val_sample_idx = [i for i in range(len(dataset)) 
                      if dataset[i][1].get('subject', str(i)) in val_subjects]
    
    train_subset = Subset(dataset, train_sample_idx)
    val_subset = Subset(dataset, val_sample_idx)
    
    collate_fn = partial(custom_collate_fn, expected_roi=expected_roi)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True if len(train_sample_idx) > batch_size else False,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, debug=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (batch_x, batch_meta) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        
        # Get labels
        labels = torch.tensor([
            int(m.get('label', m.get('labels', 0))) 
            if not (isinstance(m.get('label', m.get('labels', 0)), float) and 
                    np.isnan(m.get('label', m.get('labels', 0))))
            else 0
            for m in batch_meta
        ], dtype=torch.long, device=device)
        
        # Get FC matrices
        fc_matrices = []
        for meta in batch_meta:
            if 'fc_matrix' in meta and meta['fc_matrix'] is not None:
                fc_matrices.append(meta['fc_matrix'])
            else:
                fc_matrices.append(torch.eye(batch_x.shape[2]))
        fc_matrix = torch.stack(fc_matrices).to(device)
        
        optimizer.zero_grad()
        
        logits = model(batch_x, fc_matrix)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if debug and batch_idx == 0 and epoch == 1:
            print(f"Train batch 0: labels={labels.cpu().numpy()}, preds={preds}")
            print(f"Logits: {logits[:3].detach().cpu().numpy()}")
    
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


def evaluate(model, val_loader, criterion, device, n_classes, debug=False, return_predictions=False):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_subjects = []
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_meta) in enumerate(val_loader):
            batch_x = batch_x.to(device)
            
            # Get labels
            labels = torch.tensor([
                int(m.get('label', m.get('labels', 0))) 
                if not (isinstance(m.get('label', m.get('labels', 0)), float) and 
                        np.isnan(m.get('label', m.get('labels', 0))))
                else 0
                for m in batch_meta
            ], dtype=torch.long, device=device)
            
            # Get subjects
            subjects = [m.get('subject', str(i)) for i, m in enumerate(batch_meta)]
            all_subjects.extend(subjects)
            
            # Get FC matrices
            fc_matrices = []
            for meta in batch_meta:
                if 'fc_matrix' in meta and meta['fc_matrix'] is not None:
                    fc_matrices.append(meta['fc_matrix'])
                else:
                    fc_matrices.append(torch.eye(batch_x.shape[2]))
            fc_matrix = torch.stack(fc_matrices).to(device)
            
            logits = model(batch_x, fc_matrix)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if debug and batch_idx == 0:
                print(f"Val batch 0: labels={labels.cpu().numpy()}, preds={preds}")
                print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    if n_classes == 2:
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = 0.5
    else:
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.5
    
    prec, rec, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    result = {
        'loss': avg_loss,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc,
    }
    
    if return_predictions:
        result['predictions'] = all_preds.tolist()
        result['labels'] = all_labels.tolist()
        result['subjects'] = all_subjects
    
    return result


def finetune_on_dataset(
    checkpoint_path: str,
    dataset_name: str,
    data_dir: str = None,
    labels_file: str = None,
    n_folds: int = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    hidden_dims: list = [512, 256],
    dropout: float = 0.3,
    feature_type: str = 'z_concat',
    freeze_backbone: bool = False,
    device: torch.device = None,
    single_window_per_subject: bool = True,
    save_model: bool = False,
    output_dir: str = 'outputs/finetune',
    debug: bool = False,
    sequence_length: int = None,
    stride: int = None,
    seed: int = 42,
):
    """
    Fine-tune the BrainMech model on a specific dataset using K-fold cross-validation.
    """
    # Set random seeds for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = DATASET_CONFIG[dataset_name]
    n_classes = config['n_classes']
    class_names = config['class_names']
    
    if data_dir is None:
        data_dir = config['data_dir']
    if labels_file is None:
        labels_file = config['labels_file']
    if n_folds is None:
        n_folds = config['n_folds']
    
    print(f"\n{'='*70}")
    print(f"FINE-TUNING ON {dataset_name}")
    print(f"{'='*70}")
    print(f"  Data directory: {data_dir}")
    print(f"  Labels file: {labels_file}")
    print(f"  N classes: {n_classes} ({class_names})")
    print(f"  N folds: {n_folds}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Freeze backbone: {freeze_backbone}")
    print(f"  Single window per subject: {single_window_per_subject}")
    print(f"  Random seed: {seed}")
    
    base_model, model_config = load_model(checkpoint_path, device)
    
    seq_length = sequence_length if sequence_length is not None else model_config.get('sequence_length', 100)
    stride_val = stride if stride is not None else model_config.get('stride', 50)
    n_roi = model_config.get('n_roi', 116)
    
    print(f"  Sequence length: {seq_length}" + (" (from checkpoint)" if sequence_length is None else " (override)"))
    print(f"  Stride: {stride_val}" + (" (from checkpoint)" if stride is None else " (override)"))
    print(f"  N ROI: {n_roi}")
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    dataset_class = FINETUNE_DATASET_REGISTRY.get(dataset_name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(FINETUNE_DATASET_REGISTRY.keys())}")
    
    dataset = dataset_class(
        data_dir=data_dir,
        labels_file=labels_file,
        sequence_length=seq_length,
        stride=stride_val,
        normalize=True,
        single_window_per_subject=single_window_per_subject,
    )
    
    print(f"✓ Loaded {len(dataset)} samples")
    
    # If no or few samples, try with shorter sequence length
    if len(dataset) == 0:
        for try_seq, try_stride in [(100, 50), (50, 25), (30, 15)]:
            print(f"No samples with seq_length={seq_length}. Trying seq_length={try_seq}, stride={try_stride}...")
            seq_length = try_seq
            stride_val = try_stride
            dataset = dataset_class(
                data_dir=data_dir,
                labels_file=labels_file,
                sequence_length=seq_length,
                stride=stride_val,
                normalize=True,
                single_window_per_subject=single_window_per_subject,
            )
            print(f"✓ Loaded {len(dataset)} samples with seq_length={seq_length}, stride={stride_val}")
            if len(dataset) > 0:
                break
        
        if len(dataset) == 0:
            print(f"ERROR: Still no samples! Check data format.")
            return None
    
    # Get subject-level labels for stratified K-fold
    binary_mapping = config.get('binary_mapping', None)
    if binary_mapping:
        print(f"  Using binary mapping: {binary_mapping}")
    subjects, labels = get_subject_labels(dataset, binary_mapping=binary_mapping)
    unique_subjects = np.unique(subjects)
    
    # Print dataset summary BEFORE fine-tuning
    print(f"\n{'─'*50}")
    print(f"DATASET SUMMARY (before fine-tuning)")
    print(f"{'─'*50}")
    print(f"  Total samples: {len(dataset)} (per-sample evaluation)")
    print(f"  Unique subjects: {len(unique_subjects)}")
    print(f"  Samples per subject: ~{len(dataset) / max(len(unique_subjects), 1):.1f}")
    
    # Class distribution by subject
    subject_label_counts = {}
    for subj in unique_subjects:
        subj_mask = subjects == subj
        subj_label = labels[subj_mask][0]  # Get label for this subject
        subject_label_counts[subj_label] = subject_label_counts.get(subj_label, 0) + 1
    
    print(f"  Subject class distribution:")
    for label_idx in sorted(subject_label_counts.keys()):
        count = subject_label_counts[label_idx]
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class_{label_idx}"
        print(f"    {class_name} (label={label_idx}): {count} subjects ({count/len(unique_subjects)*100:.1f}%)")
    print(f"{'─'*50}\n")
    
    # Get one label per subject (mode)
    from scipy import stats
    subject_labels = []
    for subj in unique_subjects:
        subj_mask = subjects == subj
        subj_labels_arr = labels[subj_mask]
        mode_result = stats.mode(subj_labels_arr, keepdims=True)
        subject_labels.append(mode_result.mode[0])
    subject_labels = np.array(subject_labels)
    
    # Filter out invalid labels
    valid_mask = ~np.isnan(subject_labels.astype(float))
    if valid_mask.sum() < len(subject_labels):
        print(f"  ⚠️  Filtering {(~valid_mask).sum()} subjects with invalid labels")
        unique_subjects = unique_subjects[valid_mask]
        subject_labels = subject_labels[valid_mask]
    
    print(f"✓ {len(unique_subjects)} unique subjects")
    print(f"  Class distribution: {dict(zip(*np.unique(subject_labels, return_counts=True)))}")
    
    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(unique_subjects, subject_labels)):
        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'-'*70}")
        
        # Check class balance in validation set
        val_labels = subject_labels[val_idx]
        unique_val_labels = np.unique(val_labels)
        if len(unique_val_labels) < 2:
            print(f"  ⚠️  Only one class in validation set, skipping fold")
            continue
        
        print(f"  Train subjects: {len(train_idx)}, Val subjects: {len(val_idx)}")
        train_class_dist = dict(zip(*np.unique(subject_labels[train_idx], return_counts=True)))
        val_class_dist = dict(zip(*np.unique(subject_labels[val_idx], return_counts=True)))
        print(f"  Train class dist: {train_class_dist}")
        print(f"  Val class dist: {val_class_dist}")
        
        # Warn about small minority classes
        min_val_count = min(val_class_dist.values()) if val_class_dist else 0
        if min_val_count < 5:
            print(f"  ⚠️  WARNING: Smallest val class has only {min_val_count} subjects - results will be unstable!")
        
        if debug:
            # Print sample of actual labels
            print(f"  [DEBUG] First 5 train subject labels: {subject_labels[train_idx][:5]}")
            print(f"  [DEBUG] First 5 val subject labels: {subject_labels[val_idx][:5]}")
            # Show which subjects are in validation
            print(f"  [DEBUG] Val subjects: {unique_subjects[val_idx][:10]}..." if len(val_idx) > 10 else f"  [DEBUG] Val subjects: {unique_subjects[val_idx]}")
        
        # Create dataloaders
        train_loader, val_loader = create_fold_dataloaders(
            dataset, subjects, labels, unique_subjects, train_idx, val_idx,
            batch_size=batch_size, expected_roi=n_roi,
        )
        
        if len(train_loader) == 0 or len(val_loader) == 0:
            print(f"  ⚠️  Empty dataloader, skipping fold")
            continue
        
        # Create fresh model for this fold
        base_model, _ = load_model(checkpoint_path, device)
        model = BrainMechFineTuner(
            backbone=base_model,
            n_classes=n_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            feature_type=feature_type,
            freeze_backbone=freeze_backbone,
        ).to(device)
        
        # Optimizer and scheduler
        if freeze_backbone:
            optimizer = optim.AdamW(model.classifier.parameters(), lr=lr * 10, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': lr * 0.3},
                {'params': model.classifier.parameters(), 'lr': lr * 10},
            ], weight_decay=weight_decay)
            print(f"  Optimizer LRs: backbone={lr * 0.3:.6f}, head={lr * 10:.6f}")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        class_counts = np.bincount(subject_labels[train_idx].astype(int), minlength=n_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * n_classes
        class_weights = np.clip(class_weights, 0.33, 3.0)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        
        if debug:
            print(f"Class counts: {class_counts}")
            print(f"Class weights: {class_weights.cpu().numpy()}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        best_f1 = -float('inf')
        best_metrics = None
        best_epoch = 0
        patience_counter = 0
        
        pbar = trange(1, epochs + 1, desc=f'Fold {fold_idx + 1}')
        for epoch in pbar:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, debug=(debug and epoch == 1))
            val_metrics = evaluate(model, val_loader, criterion, device, n_classes, debug=(debug and epoch == 1))
            
            scheduler.step(val_metrics['f1'])
            
            pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'train_acc': f"{train_acc:.4f}",
                'val_acc': f"{val_metrics['accuracy']:.4f}",
                'val_f1': f"{val_metrics['f1']:.4f}",
            })
            
            # Early stopping based on F1
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model for this fold
                if save_model:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), output_path / f'{dataset_name}_fold{fold_idx}_best.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        # Final evaluation with predictions
        final_metrics = evaluate(model, val_loader, criterion, device, n_classes, debug=False, return_predictions=True)
        
        print(f"\n  Best epoch: {best_epoch}")
        print(f"  Best F1: {best_metrics['f1']:.4f}")
        print(f"  Best Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  Best Balanced Acc: {best_metrics['balanced_accuracy']:.4f}")
        print(f"  Best AUC: {best_metrics['auc']:.4f}")
        
        # Print prediction summary
        print(f"\n  Predictions (fold {fold_idx + 1}):")
        preds = np.array(final_metrics['predictions'])
        labels = np.array(final_metrics['labels'])
        print(f"    True labels:  {labels}")
        print(f"    Predictions:  {preds}")
        print(f"    Correct:      {(preds == labels).sum()}/{len(labels)} ({(preds == labels).mean()*100:.1f}%)")
        
        # Per-class breakdown
        for cls_idx in range(n_classes):
            cls_mask = labels == cls_idx
            if cls_mask.sum() > 0:
                cls_correct = (preds[cls_mask] == cls_idx).sum()
                cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"
                print(f"    {cls_name}: {cls_correct}/{cls_mask.sum()} correct")
        
        # Store fold info including predictions
        best_metrics['fold_predictions'] = preds.tolist()
        best_metrics['fold_labels'] = labels.tolist()
        best_metrics['fold_subjects'] = final_metrics['subjects']
        
        fold_results.append(best_metrics)
    
    # Aggregate results across folds
    if len(fold_results) == 0:
        print("\n✗ No valid folds completed!")
        return None
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS ({dataset_name}) - {len(fold_results)}-Fold CV")
    print(f"{'='*70}")
    
    metrics_to_aggregate = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'auc']
    final_results = {}
    
    for metric in metrics_to_aggregate:
        values = [r[metric] for r in fold_results if r[metric] is not None]
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            final_results[metric] = {'mean': mean_val, 'std': std_val}
            print(f"  {metric:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Print summary line in requested format
    print(f"\n  ════════════════════════════════════════════")
    print(f"  SUMMARY: Acc = {final_results['accuracy']['mean']*100:.2f}% ± {final_results['accuracy']['std']*100:.2f}%")
    print(f"           Weighted F1 = {final_results['f1']['mean']:.4f} ± {final_results['f1']['std']:.4f}")
    print(f"           Balanced Acc = {final_results['balanced_accuracy']['mean']*100:.2f}% ± {final_results['balanced_accuracy']['std']*100:.2f}%")
    print(f"  ════════════════════════════════════════════")
    
    final_results['n_folds'] = len(fold_results)
    final_results['fold_results'] = fold_results
    
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Koopman model on disease diagnosis datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--mode', type=str, default='finetune',
                       choices=['finetune', 'frozen_eval'],
                       help='Mode: finetune (train whole model) or frozen_eval (freeze backbone, train classifiers)')
    parser.add_argument('--dataset', type=str, default='ALL',
                       choices=['ABIDE', 'ADNI', 'PPMI', 'PPMI_binary', 'ALL'],
                       help='Dataset to fine-tune on (default: ALL). PPMI_binary = PD vs non-PD')
    
    # Dataset paths
    parser.add_argument('--abide_data_dir', type=str, default='../data/abide',
                       help='Path to ABIDE data directory')
    parser.add_argument('--abide_labels_file', type=str, default='../data/abide/labels.csv',
                       help='Path to ABIDE labels file')
    parser.add_argument('--adni_data_dir', type=str, default='../data/adni/BOLD',
                       help='Path to ADNI data directory')
    parser.add_argument('--adni_labels_file', type=str, default='../data/adni/metadata.csv',
                       help='Path to ADNI labels file')
    parser.add_argument('--ppmi_data_dir', type=str, default='../data/ppmi',
                       help='Path to PPMI data directory')
    parser.add_argument('--ppmi_labels_file', type=str, default='../data/ppmi/labels.csv',
                       help='Path to PPMI labels file')
    
    # Training parameters
    parser.add_argument('--n_folds', type=int, default=None,
                       help='Number of folds for cross-validation (default: dataset-specific)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4 for fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='Hidden dimensions for classifier head')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--feature_type', type=str, default='z_concat',
                       choices=['z_mean', 'z_std', 'z_concat', 'z_all'],
                       help='Feature type to extract')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone (only train classifier head)')
    
    # Sequence settings (override checkpoint config if needed)
    parser.add_argument('--sequence_length', type=int, default=None,
                       help='Sequence length (default: from checkpoint config)')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride for sliding window (default: from checkpoint config)')
    
    # Other options
    parser.add_argument('--single_window_per_subject', action='store_true', default=False,
                       help='Use single window per subject (default: False - uses sliding window)')
    parser.add_argument('--sliding_window', action='store_true', default=True,
                       help='Use sliding window (multiple windows per subject) - default: True')
    parser.add_argument('--save_model', action='store_true',
                       help='Save best model for each fold')
    parser.add_argument('--output_dir', type=str, default='outputs/finetune',
                       help='Output directory for saved models and results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output to diagnose issues')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Frozen evaluation mode parameters (used when --mode frozen_eval)
    parser.add_argument('--classifiers', type=str, nargs='+',
                       default=['FrozenMLP', 'LogisticRegression', 'RandomForest'],
                       help='Classifiers for frozen_eval mode (default: FrozenMLP, LogisticRegression, RandomForest)')
    parser.add_argument('--n_runs', type=int, default=30,
                       help='Number of runs with different seeds for frozen_eval mode (default: 30)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size for frozen_eval mode (default: 0.2)')
    
    args = parser.parse_args()
    
    # Handle sliding window flag
    if args.sliding_window:
        args.single_window_per_subject = False
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    
    # =========================================================================
    # FROZEN EVAL MODE: Use disease_diagnosis.py with frozen backbone
    # =========================================================================
    if args.mode == 'frozen_eval':
        print("\n" + "="*70)
        print("FROZEN EVALUATION MODE")
        print("="*70)
        print("Using frozen backbone + training classifiers (like disease_only.py)")
        print(f"Classifiers: {args.classifiers}")
        print(f"N runs: {args.n_runs}")
        print(f"Test size: {args.test_size}")
        
        # Import disease_diagnosis
        from eval.disease_diagnosis import main as disease_diagnosis_main
        import sys as sys_module
        
        # Build datasets list
        if args.dataset == 'ALL':
            datasets = ['ABIDE', 'ADNI', 'PPMI']
        else:
            datasets = [args.dataset.replace('_binary', '')]  # PPMI_binary -> PPMI
        
        # Filter to existing datasets
        data_dirs = []
        labels_files = []
        valid_datasets = []
        
        dataset_paths = {
            'ABIDE': (args.abide_data_dir, args.abide_labels_file),
            'ADNI': (args.adni_data_dir, args.adni_labels_file),
            'PPMI': (args.ppmi_data_dir, args.ppmi_labels_file),
        }
        
        for ds in datasets:
            data_dir, labels_file = dataset_paths.get(ds, (None, None))
            if data_dir and Path(data_dir).exists():
                valid_datasets.append(ds)
                data_dirs.append(data_dir)
                labels_files.append(labels_file if labels_file else 'None')
                print(f"✓ Added {ds}: {data_dir}")
            else:
                print(f"⚠️  Skipping {ds}: data not found at {data_dir}")
        
        if not valid_datasets:
            print("\n✗ No valid datasets found!")
            return
        
        # Build sys.argv for disease_diagnosis
        sys_argv = [
            'finetune_disease.py',
            '--checkpoint', args.checkpoint,
            '--datasets'
        ] + valid_datasets + [
            '--data_dirs'
        ] + data_dirs + [
            '--labels_files'
        ] + labels_files + [
            '--classifiers'
        ] + args.classifiers + [
            '--n_runs', str(args.n_runs),
            '--base_seed', str(args.seed),
            '--test_size', str(args.test_size),
            '--feature_type', args.feature_type,
            '--batch_size', str(args.batch_size),
        ]
        
        if args.single_window_per_subject:
            sys_argv.append('--single_window_per_subject')
        
        # Run disease_diagnosis
        original_argv = sys_module.argv
        sys_module.argv = sys_argv
        try:
            disease_diagnosis_main()
        finally:
            sys_module.argv = original_argv
        
        return
    
    # =========================================================================
    # FINETUNE MODE: Fine-tune entire model (default)
    # =========================================================================
    print("\n" + "="*70)
    print("FINE-TUNING MODE")
    print("="*70)
    
    # Determine which datasets to run
    if args.dataset == 'ALL':
        datasets = ['ABIDE', 'ADNI', 'PPMI']
    else:
        datasets = [args.dataset]
    
    # Dataset-specific paths
    dataset_paths = {
        'ABIDE': {'data_dir': args.abide_data_dir, 'labels_file': args.abide_labels_file},
        'ADNI': {'data_dir': args.adni_data_dir, 'labels_file': args.adni_labels_file},
        'PPMI': {'data_dir': args.ppmi_data_dir, 'labels_file': args.ppmi_labels_file},
        'PPMI_binary': {'data_dir': args.ppmi_data_dir, 'labels_file': args.ppmi_labels_file},  # Same as PPMI
    }
    
    all_results = {}
    
    for dataset_name in datasets:
        paths = dataset_paths[dataset_name]
        
        # Check if data exists
        if not Path(paths['data_dir']).exists():
            print(f"Skipping {dataset_name}: data directory not found at {paths['data_dir']}")
            continue
        
        try:
            results = finetune_on_dataset(
                checkpoint_path=args.checkpoint,
                dataset_name=dataset_name,
                data_dir=paths['data_dir'],
                labels_file=paths['labels_file'],
                n_folds=args.n_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
                feature_type=args.feature_type,
                freeze_backbone=args.freeze_backbone,
                device=device,
                single_window_per_subject=args.single_window_per_subject,
                save_model=args.save_model,
                output_dir=args.output_dir,
                debug=args.debug,
                sequence_length=args.sequence_length,
                stride=args.stride,
                seed=args.seed,
            )
            
            if results is not None:
                all_results[dataset_name] = results
        
        except Exception as e:
            print(f"Error fine-tuning on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    if all_results:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_names = '_'.join(all_results.keys())
        results_file = output_path / f'finetune_{dataset_names}_{timestamp}.json'
        
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            return obj
        
        all_results_json = convert_to_json_serializable(all_results)
        
        with open(results_file, 'w') as f:
            json.dump(all_results_json, f, indent=2)
        
        print(f"\n{'='*70}")
        print("ALL RESULTS SUMMARY")
        print(f"{'='*70}")
        
        for dataset_name, results in all_results.items():
            n_folds = results.get('n_folds', '?')
            print(f"\n{dataset_name} ({n_folds}-fold CV):")
            print(f"  Acc = {results['accuracy']['mean']*100:.2f}% ± {results['accuracy']['std']*100:.2f}%")
            print(f"  Weighted F1 = {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
            print(f"  Balanced Acc = {results['balanced_accuracy']['mean']*100:.2f}% ± {results['balanced_accuracy']['std']*100:.2f}%")
            print(f"  AUC = {results['auc']['mean']:.4f} ± {results['auc']['std']:.4f}")
        
        print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()

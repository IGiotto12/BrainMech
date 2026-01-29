import numpy as np

from typing import Dict, Sequence, Tuple, Optional



import torch

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import (

    mean_squared_error,

    mean_absolute_error,

    r2_score,

    accuracy_score,

    balanced_accuracy_score,

    f1_score,

    roc_auc_score,

    precision_score,

    recall_score,

    confusion_matrix,

    classification_report,

)

from sklearn.model_selection import train_test_split





class FrozenMLPHead(nn.Module):

    """

    MLP head for frozen-backbone fine-tuning.

    

    Architecture matches paper methodology:

    - Pooled backbone features (averaged over time and ROIs) → 

    - Linear → LayerNorm → GELU → Dropout (p=0.1) → 

    - ... (repeated for each hidden layer) → 

    - Linear (logits)

    

    The backbone is frozen; only this head is trainable.

    """



    def __init__(

        self,

        input_dim: int,

        hidden_dims: Sequence[int] = (512, 256),

        dropout: float = 0.4,

        output_dim: int = 1,

    ):

        super().__init__()

        layers = []

        prev_dim = input_dim

        for dim in hidden_dims:

            layers.append(nn.Linear(prev_dim, dim))

            layers.append(nn.LayerNorm(dim))

            layers.append(nn.GELU())

            layers.append(nn.Dropout(dropout))

            prev_dim = dim

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        self.head = nn.Linear(prev_dim, output_dim)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """

        Forward pass through MLP head.

        

        Args:

            x: Pooled backbone features [B, D] (already averaged over time and ROIs)

        

        Returns:

            logits: [B, output_dim]

        """

        x = self.mlp(x)

        return self.head(x)





def _build_loaders(

    X_train: np.ndarray,

    y_train: np.ndarray,

    batch_size: int,

) -> DataLoader:

    dataset = TensorDataset(

        torch.from_numpy(X_train).float(),

        torch.from_numpy(y_train),

    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)





def _split_train_val(

    X: np.ndarray,

    y: np.ndarray,

    val_ratio: float = 0.25,  # 25% of train for val -> 60:20:20 overall split

    seed: int = 42,

    is_regression: bool = False,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if len(X) < 10:

        return X, y, X, y

    

    # For regression, don't stratify. For classification, stratify if possible.

    unique_y = len(np.unique(y))

    use_stratify = False

    if not is_regression and unique_y > 1:

        # Check if we have enough samples per class for stratification

        min_class_size = min([np.sum(y == val) for val in np.unique(y)])

        if min_class_size >= 2:  # Need at least 2 samples per class

            use_stratify = True

    

    idx_train, idx_val = train_test_split(

        np.arange(len(X)),

        test_size=val_ratio,

        random_state=seed,

        stratify=y if use_stratify else None,

    )

    return X[idx_train], y[idx_train], X[idx_val], y[idx_val]





def train_regression_head(

    X_train: np.ndarray,

    y_train: np.ndarray,

    X_test: np.ndarray,

    y_test: np.ndarray,

    hidden_dims: Sequence[int] = (512, 256),

    dropout: float = 0.4,

    lr: float = 1e-3,

    weight_decay: float = 1e-4,

    batch_size: int = 64,

    epochs: int = 200,

    patience: int = 20,

    device: Optional[torch.device] = None,

    use_robust_norm: bool = True,

    grad_clip: float = None,

) -> Dict:

    """

    Train regression head (e.g., for age prediction) following paper methodology.

    

    Methodology:

    - Freeze pretrained backbone, train only MLP head

    - MLP: Linear → LayerNorm → GELU → Dropout (p=0.1) → ... → Linear

    - Loss: MSE (mean squared error)

    - Optimizer: Adam with ReduceLROnPlateau scheduler

    - Normalization: Robust normalization (median/IQR) retained from pretraining

    

    Args:

        use_robust_norm: If True, use RobustScaler (median/IQR). If False, use StandardScaler (mean/std).

    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # Use robust normalization (median/IQR) as mentioned in paper, or standard normalization

    if use_robust_norm:

        scaler = RobustScaler()  # Uses median and IQR (more robust to outliers)

    else:

        scaler = StandardScaler()  # Uses mean and std

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)



    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(-1)

    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(-1)



    X_train_sub, y_train_sub, X_val, y_val = _split_train_val(

        X_train_scaled, y_train, val_ratio=0.25, seed=42, is_regression=True

    )

    train_loader = _build_loaders(X_train_sub, y_train_sub[:, None], batch_size)



    model = FrozenMLPHead(

        input_dim=X_train.shape[1],

        hidden_dims=hidden_dims,

        dropout=dropout,

        output_dim=1,

    ).to(device)

    criterion = nn.MSELoss()  # MSE for regression tasks (as per paper)

    # Use Adam (not AdamW) to match paper methodology exactly

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5

    )



    best_state = None

    best_val = float("inf")

    patience_counter = 0



    X_val_tensor = torch.from_numpy(X_val).float().to(device)

    y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(-1).to(device)



    for epoch in range(epochs):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)

            yb = yb.to(device).float()

            preds = model(xb)

            loss = criterion(preds, yb)

            optimizer.zero_grad()

            loss.backward()

            # Gradient clipping to prevent exploding gradients and improve stability

            if grad_clip is not None and grad_clip > 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()



        model.eval()

        with torch.no_grad():

            val_preds = model(X_val_tensor)

            val_loss = criterion(val_preds, y_val_tensor).item()

        scheduler.step(val_loss)



        if val_loss < best_val - 1e-4:

            best_val = val_loss

            best_state = model.state_dict()

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= patience:

                break



    if best_state is not None:

        model.load_state_dict(best_state)



    def _predict(X_np: np.ndarray) -> np.ndarray:

        tensor = torch.from_numpy(X_np).float().to(device)

        with torch.no_grad():

            preds = model(tensor).cpu().numpy().squeeze(-1)

        return preds



    train_preds = _predict(X_train_scaled)

    test_preds = _predict(X_test_scaled)



    # Pearson correlation

    train_pearson, _ = pearsonr(y_train, train_preds)

    test_pearson, _ = pearsonr(y_test, test_preds)



    results = {

        "train_mse": float(mean_squared_error(y_train, train_preds)),

        "train_mae": float(mean_absolute_error(y_train, train_preds)),

        "train_r2": float(r2_score(y_train, train_preds)),

        "train_pearson": float(train_pearson),

        "test_mse": float(mean_squared_error(y_test, test_preds)),

        "test_mae": float(mean_absolute_error(y_test, test_preds)),

        "test_r2": float(r2_score(y_test, test_preds)),

        "test_pearson": float(test_pearson),

        "scaler": scaler,

        "model_state_dict": model.state_dict(),

        "train_predictions": train_preds.tolist(),

        "test_predictions": test_preds.tolist(),

    }

    return results





def train_classification_head(

    X_train: np.ndarray,

    y_train: np.ndarray,

    X_test: np.ndarray,

    y_test: np.ndarray,

    hidden_dims: Sequence[int] = (512, 256),

    dropout: float = 0.4,

    lr: float = 1e-3,

    weight_decay: float = 1e-4,

    batch_size: int = 64,

    epochs: int = 200,

    patience: int = 20,

    device: Optional[torch.device] = None,

    use_robust_norm: bool = True,

    label_smoothing: float = 0.0,

    grad_clip: float = None,

) -> Dict:

    """

    Train classification head (e.g., for sex, diagnosis) following paper methodology.

    

    Methodology:

    - Freeze pretrained backbone, train only MLP head

    - MLP: Linear → LayerNorm → GELU → Dropout (p=0.1) → ... → Linear

    - Loss: Cross-entropy

    - Optimizer: Adam with ReduceLROnPlateau scheduler

    - Normalization: Robust normalization (median/IQR) retained from pretraining

    

    Args:

        use_robust_norm: If True, use RobustScaler (median/IQR). If False, use StandardScaler (mean/std).

    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = int(np.max(y_train)) + 1



    # Use robust normalization (median/IQR) as mentioned in paper, or standard normalization

    if use_robust_norm:

        scaler = RobustScaler()  # Uses median and IQR (more robust to outliers)

    else:

        scaler = StandardScaler()  # Uses mean and std

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)



    X_train_sub, y_train_sub, X_val, y_val = _split_train_val(

        X_train_scaled, y_train, val_ratio=0.25, seed=42, is_regression=False

    )

    train_loader = _build_loaders(X_train_sub, y_train_sub[:, None], batch_size)



    model = FrozenMLPHead(

        input_dim=X_train.shape[1],

        hidden_dims=hidden_dims,

        dropout=dropout,

        output_dim=num_classes,

    ).to(device)



    class_counts = np.bincount(y_train.astype(int), minlength=num_classes)

    class_weights = (len(y_train) / (num_classes * class_counts + 1e-6)).astype(np.float32)

    # Use label smoothing if specified (helps with overfitting)

    if label_smoothing > 0:

        criterion = nn.CrossEntropyLoss(

            weight=torch.from_numpy(class_weights).to(device),

            label_smoothing=label_smoothing

        )

    else:

        criterion = nn.CrossEntropyLoss(  # Cross-entropy for classification tasks (as per paper)

            weight=torch.from_numpy(class_weights).to(device)

        )

    # Use Adam (not AdamW) to match paper methodology exactly

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5

    )



    best_state = None

    best_val = -float("inf")

    patience_counter = 0



    X_val_tensor = torch.from_numpy(X_val).float().to(device)

    y_val_tensor = torch.from_numpy(y_val).long().to(device)



    for epoch in range(epochs):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)

            yb = yb.to(device).long().squeeze()

            logits = model(xb)

            loss = criterion(logits, yb)

            optimizer.zero_grad()

            loss.backward()

            # Gradient clipping to prevent exploding gradients and improve stability

            if grad_clip is not None and grad_clip > 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()



        model.eval()

        with torch.no_grad():

            val_logits = model(X_val_tensor)

            val_preds = val_logits.argmax(dim=-1)

            val_acc = (val_preds == y_val_tensor).float().mean().item()

            # Also track validation loss for better early stopping

            val_loss = criterion(val_logits, y_val_tensor).item()

        scheduler.step(val_acc)



        # Early stopping based on validation accuracy (primary) and loss (secondary)

        # Use a small tolerance to prevent stopping too early due to noise

        improvement_threshold = 1e-4

        if val_acc > best_val + improvement_threshold:

            best_val = val_acc

            best_state = model.state_dict()

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= patience:

                break



    if best_state is not None:

        model.load_state_dict(best_state)



    def _predict_logits(X_np: np.ndarray) -> np.ndarray:

        tensor = torch.from_numpy(X_np).float().to(device)

        with torch.no_grad():

            logits = model(tensor).cpu().numpy()

        return logits



    train_logits = _predict_logits(X_train_scaled)

    test_logits = _predict_logits(X_test_scaled)

    train_probs = torch.softmax(torch.from_numpy(train_logits), dim=-1).numpy()

    test_probs = torch.softmax(torch.from_numpy(test_logits), dim=-1).numpy()

    train_preds = np.argmax(train_probs, axis=-1)

    test_preds = np.argmax(test_probs, axis=-1)



    average = "binary" if num_classes == 2 else "macro"

    test_auc = None

    if num_classes == 2:

        test_auc = roc_auc_score(y_test, test_probs[:, 1])

    elif num_classes > 2:

        try:

            test_auc = roc_auc_score(

                y_test, test_probs, average="macro", multi_class="ovr"

            )

        except ValueError:

            test_auc = None



    cm = confusion_matrix(y_test, test_preds)

    per_class_f1 = f1_score(y_test, test_preds, average=None, zero_division=0)

    clf_report = classification_report(

        y_test, test_preds, output_dict=True, zero_division=0

    )



    if num_classes == 2 and cm.size == 4:

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    else:

        sensitivity = None

        specificity = None



    results = {

        "train_accuracy": float(accuracy_score(y_train, train_preds)),

        "train_balanced_accuracy": float(

            balanced_accuracy_score(y_train, train_preds)

        ),

        "train_f1": float(f1_score(y_train, train_preds, average=average)),

        "test_accuracy": float(accuracy_score(y_test, test_preds)),

        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_preds)),

        "test_f1_macro": float(f1_score(y_test, test_preds, average="macro")),

        "test_f1_weighted": float(f1_score(y_test, test_preds, average="weighted")),

        "test_f1": float(f1_score(y_test, test_preds, average=average)),

        "test_precision": float(

            precision_score(y_test, test_preds, average=average, zero_division=0)

        ),

        "test_recall": float(

            recall_score(y_test, test_preds, average=average, zero_division=0)

        ),

        "test_auc": float(test_auc) if test_auc is not None else None,

        "confusion_matrix": cm.tolist(),

        "per_class_f1": [float(x) for x in per_class_f1],

        "classification_report": clf_report,

        "sensitivity": float(sensitivity) if sensitivity is not None else None,

        "specificity": float(specificity) if specificity is not None else None,

        "train_predictions": train_preds.tolist(),

        "test_predictions": test_preds.tolist(),

        "test_probabilities": test_probs.tolist(),

        "model_state_dict": model.state_dict(),

        "scaler": scaler,

        "predictions": {

            "test_true": y_test.tolist(),

            "test_pred": test_preds.tolist(),

            "test_probs": test_probs.tolist(),

        },

        "cv_balanced_accuracy": None,

        "cv_balanced_accuracy_std": None,

    }

    return results




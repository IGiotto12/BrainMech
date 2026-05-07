"""
Pretraining configuration.

Loss objective:

    L = L_pre + lambda_1 * L_dyn + lambda_2 * L_stab

  * L_pre  = one-step forecasting Gaussian NLL    (--forecast_weight)
  * L_dyn  = Koopman-step latent consistency MSE  (--consistency_weight)
  * L_stab = spectral-radius hinge with rho=0.95  (--spectral_weight)

Rollout (--rollout_weight) is zeroed because it is not part of the objective;
its CLI default is non-zero, so it must be set explicitly.

Subject age is normalized to [0, 1] (range [age_min, age_max]) and embedded as
the first dimension of the context vector v that drives the control signal.

Datasets: HCPA + UKB + ABCD pooled, resting-state only, AAL-116 atlas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class PretrainConfig:
    """Pretraining experiment configuration. Serialises to ``pretrain.py`` CLI args."""

    # --- Identification ---
    name: str
    description: str = ""

    # --- Datasets (pooled HCPA + UKB + ABCD, REST only) ---
    dataset_type: str = "COMBINED"
    data_dir: str = "../../data/hcpa/AAL_116/BOLD"
    fc_matrix_path: Optional[str] = "../../data/hcpa/AAL_116/FC"
    demographics_csv: Optional[str] = "../../data/hcpa/HCPA_metadata.csv"
    task_filter: Optional[str] = "REST"
    additional_data_dir: Optional[str] = "../../data/ukb/AAL_116/BOLD"
    additional_fc_matrix_path: Optional[str] = "../../data/ukb/AAL_116/FC"
    third_data_dir: Optional[str] = "../../data/abcd/BOLD"
    third_fc_matrix_path: Optional[str] = "../../data/abcd/FC"

    # --- Data loading ---
    n_roi: int = 116
    sequence_length: int = 300
    stride: int = 300
    train_split: float = 0.8
    num_workers: int = 4

    # --- Model architecture ---
    latent_dim: int = 768
    control_dim: int = 192
    episodic_dim: int = 192
    context_dim: int = 192
    task_dim: int = 32

    # --- Graph encoder (TASEN spatial branch) ---
    graph_type: str = "cheb"
    n_graph_layers: int = 4
    cheb_k: int = 3
    drop_edge_p: float = 0.1

    # --- Temporal encoder (TASEN temporal branch) ---
    n_temporal_layers: int = 4
    n_heads: int = 8

    # --- DPLR-Koopman core (rank scales with latent dim) ---
    koopman_rank: int = 48
    target_spectral_radius: float = 0.95  # rho in L_stab
    use_sde: bool = True
    diffusion_mode: str = "learned"
    uncertainty_mode: str = "heteroscedastic"

    # --- Training ---
    batch_size: int = 32
    epochs: int = 100
    lr: float = 5e-5
    weight_decay: float = 2e-4
    dropout: float = 0.2
    early_stopping_patience: Optional[int] = 30

    # --- ReduceLROnPlateau scheduler ---
    use_plateau_scheduler: bool = True
    plateau_factor: float = 0.8
    plateau_patience: int = 5

    # --- Loss weights ---
    forecast_weight: float = 1.0       # L_pre
    consistency_weight: float = 0.5    # lambda_1 (L_dyn)
    spectral_weight: float = 0.1       # lambda_2 (L_stab)
    rollout_weight: float = 0.0        # override CLI default of 0.5

    # --- Age conditioning (age normalized to [0, 1] embedded in context vectors) ---
    use_age_context: bool = True
    age_min: float = 8.0
    age_max: float = 100.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_args(self) -> List[str]:
        """Convert config to a list of CLI arguments for ``pretrain.py``."""
        args: List[str] = []

        # Datasets
        args += ["--data_dir", self.data_dir]
        if self.fc_matrix_path:
            args += ["--fc_matrix_path", self.fc_matrix_path]
        args += ["--dataset_type", self.dataset_type]
        if self.task_filter:
            args += ["--task_filter", self.task_filter]
        if self.additional_data_dir:
            args += ["--additional_data_dir", self.additional_data_dir]
        if self.additional_fc_matrix_path:
            args += ["--additional_fc_matrix_path", self.additional_fc_matrix_path]
        if self.third_data_dir:
            args += ["--third_data_dir", self.third_data_dir]
        if self.third_fc_matrix_path:
            args += ["--third_fc_matrix_path", self.third_fc_matrix_path]

        # Data loading
        args += ["--n_roi", str(self.n_roi)]
        args += ["--sequence_length", str(self.sequence_length)]
        args += ["--stride", str(self.stride)]
        args += ["--train_split", str(self.train_split)]
        args += ["--num_workers", str(self.num_workers)]

        # Architecture
        args += ["--latent_dim", str(self.latent_dim)]
        args += ["--control_dim", str(self.control_dim)]
        args += ["--episodic_dim", str(self.episodic_dim)]
        args += ["--context_dim", str(self.context_dim)]
        args += ["--task_dim", str(self.task_dim)]

        # Graph encoder
        args += ["--graph_type", self.graph_type]
        args += ["--n_graph_layers", str(self.n_graph_layers)]
        args += ["--cheb_k", str(self.cheb_k)]
        args += ["--drop_edge_p", str(self.drop_edge_p)]

        # Temporal encoder
        args += ["--n_temporal_layers", str(self.n_temporal_layers)]
        args += ["--n_heads", str(self.n_heads)]

        # DPLR-Koopman core
        args += ["--koopman_rank", str(self.koopman_rank)]
        args += ["--target_spectral_radius", str(self.target_spectral_radius)]
        if self.use_sde:
            args.append("--use_sde")
        args += ["--diffusion_mode", self.diffusion_mode]
        args += ["--uncertainty_mode", self.uncertainty_mode]

        # Training
        args += ["--batch_size", str(self.batch_size)]
        args += ["--epochs", str(self.epochs)]
        args += ["--lr", str(self.lr)]
        args += ["--weight_decay", str(self.weight_decay)]
        args += ["--dropout", str(self.dropout)]
        if self.early_stopping_patience is not None:
            args += ["--early_stopping_patience", str(self.early_stopping_patience)]

        # Scheduler
        if self.use_plateau_scheduler:
            args.append("--use_plateau_scheduler")
        args += ["--plateau_factor", str(self.plateau_factor)]
        args += ["--plateau_patience", str(self.plateau_patience)]

        # Loss weights
        args += ["--forecast_weight", str(self.forecast_weight)]
        args += ["--consistency_weight", str(self.consistency_weight)]
        args += ["--spectral_weight", str(self.spectral_weight)]
        args += ["--rollout_weight", str(self.rollout_weight)]

        # Age conditioning
        if self.use_age_context:
            args.append("--use_age_context")
            if self.demographics_csv:
                args += ["--demographics_csv", self.demographics_csv]
            args += ["--age_min", str(self.age_min)]
            args += ["--age_max", str(self.age_max)]

        return args

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PretrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# Default pretraining experiment
# ============================================================================
DEFAULT = PretrainConfig(
    name="default",
    description=(
        "Pooled HCPA + UKB + ABCD, resting-state only, AAL-116, T=300, stride=300. "
        "Loss = L_pre + lambda_1 * L_dyn + lambda_2 * L_stab "
        "(forecasting NLL + Koopman-step consistency MSE + spectral hinge, rho=0.95). "
        "AdamW, lr=5e-5, wd=2e-4, dropout=0.2, ReduceLROnPlateau (patience=5, factor=0.8)."
    ),
)


CONFIG_REGISTRY: Dict[str, PretrainConfig] = {DEFAULT.name: DEFAULT}


def get_config(name: str) -> PretrainConfig:
    """Look up a registered config by name, or load a JSON file."""
    if name in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[name]
    if name.endswith(".json"):
        return PretrainConfig.load(name)
    available = ", ".join(sorted(CONFIG_REGISTRY))
    raise KeyError(f"Unknown config '{name}'. Available: {available}")


def list_configs() -> List[str]:
    """Return all registered config names."""
    return sorted(CONFIG_REGISTRY)

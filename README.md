# BrainMech: A Mechanistic Foundation Model for Human Brain Dynamics Across Lifespan

This repository contains the official PyTorch implementation of **BrainMech**, a mechanistic foundation model for fMRI dynamics grounded in **Neural Koopman Observable State-Space Linearization (NKOSSL)**, **Topology-Aware Spatiotemporal Embedding Network (TASEN)**, and a **DPLR-Koopman** core with stochastic diffusion.

## Model Architecture & Methodology

BrainMech models functional brain dynamics in a stochastic Koopman latent space, learning a population-level governing law with uncertainty-aware decoding.

**Core Components:**

1. **Topology-Aware Spatiotemporal Embedding Network (TASEN):**
   - Spatial encoder operates on FC topology (ChebNet/GAT).
   - Temporal encoder captures long-range dependencies across BOLD sequences.

2. **Control Signal Fusion (Episodic + Context):**
   - Episodic feedback and context encoder generate control input $u_\tau$.
   - Control modulates latent dynamics through low-rank control matrix $C$.

3. **DPLR-Koopman Core (Structured SDE):**
   - Linear drift with diagonal-plus-low-rank structure:
     $$K = \mathrm{diag}(k) + U V^\top$$
   - Latent SDE:
     $$dZ_t = (K Z_t + C u_t + \beta)\,dt + \Sigma_\theta(Z_t,u_t)\,dW_t$$

4. **Stochastic Decoder:**
   - Gaussian observation model:
     $$x_\tau \mid z_\tau \sim \mathcal{N}(\mu_{\theta}(z_\tau), \sigma^2_{\theta}(z_\tau)I)$$

## Repository Structure

```text
BrainMech/
├── README.md
├── requirements.txt
├── data_loader.py
├── configs/
│   ├── __init__.py
│   └── pretrain_configs.py
├── eval/
│   ├── age_sex.py
│   ├── disease_diagnosis.py
│   ├── evaluate_downstream.py
│   ├── feature_cache.py
│   └── frozen_mlp_head.py
├── models/
│   ├── __init__.py
│   ├── brainmech.py
│   ├── control_fusion.py
│   ├── decoder_stochastic.py
│   ├── episodic_feedback.py
│   ├── graph_spatial_encoder.py
│   ├── koopman_core_structured.py
│   ├── masked_pretraining.py
│   ├── module_probes.py
│   ├── regularizers.py
│   ├── task_context_encoder.py
│   └── temporal_transformer.py
└── training/
    ├── pretrain.py
    └── training_utils.py
```

## Usage Instructions

## 1. Pretraining (Self-Supervised)
Train BrainMech with stability-regularized Koopman dynamics. The objective is

$$\mathcal{L} = \mathcal{L}_{\mathrm{pre}} + \lambda_1\,\mathcal{L}_{\mathrm{dyn}} + \lambda_2\,\mathcal{L}_{\mathrm{stab}}$$

(one-step forecasting NLL + Koopman-step latent consistency MSE + spectral-radius hinge with $\rho=0.95$).

The recommended entry point is the bundled config, which fixes the dataset pool
(HCPA + UKB + ABCD, resting-state, AAL-116), sequence length, hyperparameters,
and loss weights:

```bash
python training/pretrain.py --config default
```

Any field can be overridden by passing the corresponding flag on the command line:

```bash
python training/pretrain.py --config default --batch_size 8 --epochs 50
```

Listing the registered configs:

```bash
python -c "from configs.pretrain_configs import list_configs; print(list_configs())"
```

## 2. Sex Classification
Evaluate frozen-backbone representations with an MLP head:

```bash
python eval/age_sex.py \
  --checkpoint best_model.pt \
  --dataset HCPA \
  --data_dir \dataset_dir\
```

## 3. Disease Diagnosis
Run disease diagnosis on ABIDE/ADNI/PPMI:

```bash
python eval/disease_diagnosis.py \
  --checkpoint best_model.pt \
  --dataset ABIDE \
  --labels_file /dataset_dir/labels.csv
```

## 4. Unified Downstream Evaluation
Single entry for downstream tasks:

```bash
python eval/evaluate_downstream.py \
  --checkpoint best_model.pt \
  --dataset ABIDE \
  --data_dir \dataset_dir\ \
  --task_type classification
```

## Configuration
The recommended workflow is to start from a config in `configs/pretrain_configs.py`
and override only what differs (`--config default --<flag> <value>`). Key options
exposed on the CLI:

- `--koopman_rank`: DPLR rank $r$.
- `--latent_dim`: Latent dimension $d$.
- `--use_sde`: Enable stochastic diffusion.
- `--graph_type`: `cheb` or `gat` for TASEN spatial encoder.
- `--sequence_length`: BOLD segment length $T$.
- `--use_age_context` / `--age_min` / `--age_max`: enable age-conditioned context.
- `--forecast_weight` / `--consistency_weight` / `--spectral_weight`: tune the three loss terms.

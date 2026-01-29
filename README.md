# BrainMech: Mechanistic Foundation Model (Anonymous)

**[Supplementary Code for ICML 2026 Submission]**

This repository contains the official PyTorch implementation of **BrainMech**, a mechanistic foundation model for fMRI dynamics grounded in **Neural Koopman Observable State-Space Linearization (NKOSSL)**, **Topology-Aware Spatiotemporal Embedding Network (TASEN)**, and a **DPLR-Koopman** core with stochastic diffusion.

## 🧠 Model Architecture & Methodology

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

## 📂 Repository Structure

```text
BrainMech_min/
├── README.md
├── requirements.txt
├── eval/
│   ├── age_sex.py
│   ├── disease_diagnosis.py
│   ├── evaluate_downstream.py
│   ├── feature_cache.py
│   └── frozen_mlp_head.py
├── models/
│   ├── __init__.py
│   ├── brainmech.py
│   ├── cognitive_readout.py
│   ├── control_fusion.py
│   ├── decoder_stochastic.py
│   ├── episodic_feedback.py
│   ├── graph_spatial_encoder_real.py
│   ├── koopman_brain_model_v2.py
│   ├── koopman_core_structured.py
│   ├── masked_pretraining.py
│   ├── module_probes.py
│   ├── regularizers.py
│   ├── task_context_encoder.py
│   └── temporal_transformer.py
├── training/
│   ├── train_with_mae.py
│   └── training_utils.py
└── utils/
    └── data_loader.py
```

## 🚀 Usage Instructions

## 1. Pretraining (Self-Supervised)
Train BrainMech with masked reconstruction and stochastic Koopman dynamics:

```bash
python training/train_with_mae.py \
  --npz_path data/hcpa_bold_with_fc.npz \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.0002 \
  --koopman_rank 16 \
  --use_sde
```

## 2. Sex Classification
Evaluate frozen-backbone representations with an MLP head:

```bash
python eval/age_sex.py \
  --checkpoint results/brainmech_hcpa/best_model.pt \
  --dataset HCPA \
  --data_dir data/HCPA/BOLD
```

## 3. Disease Diagnosis
Run disease diagnosis on ABIDE/ADNI/PPMI:

```bash
python eval/disease_diagnosis.py \
  --checkpoint results/brainmech_hcpa/best_model.pt \
  --dataset ABIDE \
  --labels_file data/ABIDE/labels.csv
```

## 4. Unified Downstream Evaluation
Single entry for downstream tasks:

```bash
python eval/evaluate_downstream.py \
  --checkpoint results/brainmech_hcpa/best_model.pt \
  --dataset ABIDE \
  --data_dir data/ABIDE/BOLD \
  --task_type classification
```

## ⚙️ Configuration
Hyperparameters are set via command line arguments in `training/train_with_mae.py`. Key options:

- `--koopman_rank`: DPLR rank $r$.
- `--latent_dim`: Latent dimension $d$.
- `--use_sde`: Enable stochastic diffusion.
- `--graph_type`: `cheb` or `gat` for TASEN spatial encoder.
- `--sequence_length`: BOLD segment length $T$.

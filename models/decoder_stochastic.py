"""

Stochastic Decoder with Uncertainty Quantification

Outputs (μ_x, σ_x) for probabilistic reconstruction/forecasting

"""



import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Tuple





class StochasticDecoder(nn.Module):

    """

    Stochastic Decoder that outputs mean and uncertainty

    

    Contribution 2: Uncertainty-aware predictions

    - Outputs (μ_x, log σ_x) instead of point estimate

    - Trained with Gaussian negative log-likelihood

    - Calibrated uncertainty for risk-aware control

    """

    

    def __init__(

        self,

        d_latent: int,

        n_roi: int,

        hidden_dim: int = 256,

        dropout: float = 0.1,

        min_std: float = 1e-3,

        uncertainty_mode: str = 'heteroscedastic',  # 'heteroscedastic', 'homoscedastic', 'none'

        detach_sigma_head: bool = False,

        sigma_calibration: str = 'none',  # 'none', 'roi', 'roi+time'

        max_time_steps: int = 512,

    ):

        """

        Args:

            d_latent: Latent dimension

            n_roi: Number of ROIs (output dimension)

            hidden_dim: Hidden layer dimension

            dropout: Dropout rate

            min_std: Minimum standard deviation (for numerical stability)

            uncertainty_mode: How to model uncertainty

                - 'heteroscedastic': σ depends on input (learned per sample)

                - 'homoscedastic': σ is global (learned scalar per ROI)

                - 'none': deterministic (no uncertainty)

        """

        super().__init__()

        self.d_latent = d_latent

        self.n_roi = n_roi

        self.min_std = min_std

        self.uncertainty_mode = uncertainty_mode

        self.detach_sigma_head = detach_sigma_head

        self.sigma_calibration = sigma_calibration

        self.max_time_steps = max_time_steps

        

        # Shared encoder

        self.shared = nn.Sequential(

            nn.Linear(d_latent, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.LayerNorm(hidden_dim)

        )

        

        # Mean head

        self.mu_head = nn.Linear(hidden_dim, n_roi)

        

        # Uncertainty head

        if uncertainty_mode == 'heteroscedastic':

            # Input-dependent uncertainty

            # Use a separate path (not shared) to encourage heteroscedastic learning

            self.sigma_encoder = nn.Sequential(

                nn.Linear(d_latent, hidden_dim),

                nn.GELU(),

                nn.Dropout(dropout),

                nn.LayerNorm(hidden_dim)

            )

            self.log_sigma_head = nn.Sequential(

                nn.Linear(hidden_dim, hidden_dim // 2),

                nn.GELU(),

                nn.Dropout(dropout),

                nn.Linear(hidden_dim // 2, n_roi)

            )

            # Initialize to produce larger, more variable uncertainty

            # Target: σ ≈ 0.3-0.4 initially (softplus(-1.0) ≈ 0.31)

            with torch.no_grad():

                # Use normal initialization (not scaled down) to allow learning

                nn.init.normal_(self.log_sigma_head[-1].weight, mean=0.0, std=0.1)

                # Initialize bias to -1.0 (softplus(-1.0) ≈ 0.31) for better scale

                self.log_sigma_head[-1].bias.data.fill_(-1.0)

        

        elif uncertainty_mode == 'homoscedastic':

            # Global uncertainty (one per ROI)

            # Initialize to -1.0 for better scale (softplus(-1.0) ≈ 0.31)

            self.log_sigma = nn.Parameter(torch.zeros(n_roi) - 1.0)

        

        # If 'none', no uncertainty parameters

        

        # Calibration parameters (optional, controlled by sigma_calibration)

        if self.uncertainty_mode == 'heteroscedastic' and self.sigma_calibration in ('roi', 'roi+time'):

            # ROI-wise bias on log variance

            self.sigma_bias_roi = nn.Parameter(torch.zeros(n_roi))

        else:

            self.sigma_bias_roi = None

        

        if self.uncertainty_mode == 'heteroscedastic' and self.sigma_calibration == 'roi+time':

            # Time-wise bias on log variance (up to max_time_steps)

            self.sigma_bias_time = nn.Parameter(torch.zeros(max_time_steps))

        else:

            self.sigma_bias_time = None

    

    def forward(

        self, 

        z: torch.Tensor,

        return_uncertainty: bool = True

    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """

        Forward pass with uncertainty estimation

        

        Args:

            z: [B, T, d_latent] Latent states

            return_uncertainty: Whether to return uncertainty

        

        Returns:

            mu: [B, T, n_roi] Mean prediction

            sigma: [B, T, n_roi] Standard deviation (if return_uncertainty=True)

        """

        # Shared processing for mean

        h = self.shared(z)  # [B, T, hidden_dim]

        

        # Mean prediction

        mu = self.mu_head(h)  # [B, T, n_roi]

        

        # Uncertainty prediction

        if return_uncertainty and self.uncertainty_mode != 'none':

            if self.uncertainty_mode == 'heteroscedastic':

                # Use separate encoder path for sigma to encourage heteroscedastic learning

                # This allows sigma to learn different patterns than mu

                h_sigma_encoded = self.sigma_encoder(z)  # [B, T, hidden_dim]

                # Optionally detach if requested (but default is False for better learning)

                h_sigma = h_sigma_encoded.detach() if self.detach_sigma_head else h_sigma_encoded

                

                if self.sigma_calibration == 'none':

                    # Backwards-compatible behavior: original softplus head

                    log_sigma = self.log_sigma_head(h_sigma)  # [B, T, n_roi]

                    sigma = F.softplus(log_sigma) + self.min_std

                else:

                    # New path: model log variance with calibration terms

                    # log_sigma2_raw: arbitrary real values

                    log_sigma2 = self.log_sigma_head(h_sigma)  # [B, T, n_roi]

                    

                    B, T, N = log_sigma2.shape

                    

                    # ROI-wise bias

                    if self.sigma_bias_roi is not None:

                        log_sigma2 = log_sigma2 + self.sigma_bias_roi.view(1, 1, N)

                    

                    # Time-wise bias

                    if self.sigma_bias_time is not None:

                        # Use first T positions of bias (assume T <= max_time_steps in practice)

                        time_bias = self.sigma_bias_time[:T].view(1, T, 1)

                        log_sigma2 = log_sigma2 + time_bias

                    

                    # Clamp log variance for numerical stability

                    log_sigma2 = torch.clamp(log_sigma2, min=-10.0, max=10.0)

                    

                    # Convert log variance to standard deviation

                    sigma = torch.exp(0.5 * log_sigma2) + self.min_std

            elif self.uncertainty_mode == 'homoscedastic':

                # Global (broadcast)

                sigma = F.softplus(self.log_sigma) + self.min_std

                sigma = sigma.unsqueeze(0).unsqueeze(0).expand_as(mu)

            

            return mu, sigma

        else:

            return mu, None

    

    def sample(self, z: torch.Tensor, n_samples: int = 1) -> torch.Tensor:

        """

        Sample from predictive distribution p(x|z)

        

        Args:

            z: [B, T, d_latent] Latent states

            n_samples: Number of samples to draw

        

        Returns:

            samples: [n_samples, B, T, n_roi] Sampled predictions

        """

        mu, sigma = self.forward(z, return_uncertainty=True)

        

        if sigma is None:

            # Deterministic: just repeat

            return mu.unsqueeze(0).repeat(n_samples, 1, 1, 1)

        

        # Sample from N(μ, σ²)

        samples = []

        for _ in range(n_samples):

            eps = torch.randn_like(mu)

            sample = mu + sigma * eps

            samples.append(sample)

        

        return torch.stack(samples, dim=0)  # [n_samples, B, T, n_roi]

    

    def gaussian_nll_loss(

        self,

        z: torch.Tensor,

        x_target: torch.Tensor,

        reduction: str = 'mean'

    ) -> torch.Tensor:

        """

        Gaussian negative log-likelihood loss

        

        Args:

            z: [B, T, d_latent] Latent states

            x_target: [B, T, n_roi] Target ROI signals

            reduction: 'mean', 'sum', or 'none'

        

        Returns:

            nll: Scalar loss or [B, T] tensor if reduction='none'

        """

        mu, sigma = self.forward(z, return_uncertainty=True)

        

        if sigma is None:

            # Fallback to MSE if no uncertainty

            return F.mse_loss(mu, x_target, reduction=reduction)

        

        # NLL = 0.5 * [log(2πσ²) + (x - μ)²/σ²]

        # Simplified: log(σ) + 0.5 * ((x - μ)/σ)²

        squared_error = (x_target - mu) ** 2

        nll = torch.log(sigma) + 0.5 * (squared_error / (sigma ** 2 + 1e-8))

        

        if reduction == 'mean':

            return nll.mean()

        elif reduction == 'sum':

            return nll.sum()

        else:

            return nll  # [B, T, n_roi]





class CognitiveStateReadoutStochastic(nn.Module):

    """

    Cognitive readout with uncertainty for task predictions

    """

    

    def __init__(

        self,

        d_latent: int,

        n_classes: int,

        task_type: str = 'classification',

        hidden_dim: int = 256,

        dropout: float = 0.1,

        mc_samples: int = 10

    ):

        """

        Args:

            d_latent: Latent dimension

            n_classes: Number of output classes (or output dim for regression)

            task_type: 'classification' or 'regression'

            hidden_dim: Hidden dimension

            dropout: Dropout rate (also used for MC Dropout)

            mc_samples: Number of MC samples for uncertainty

        """

        super().__init__()

        self.d_latent = d_latent

        self.n_classes = n_classes

        self.task_type = task_type

        self.mc_samples = mc_samples

        

        # Prediction network with MC Dropout

        self.predictor = nn.Sequential(

            nn.Linear(d_latent, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, n_classes)

        )

    

    def forward(

        self, 

        z: torch.Tensor, 

        pool: bool = True,

        mc_uncertainty: bool = False

    ):

        """

        Args:

            z: [B, T, d_latent] Latent states

            pool: Whether to pool over time

            mc_uncertainty: Whether to use MC Dropout for uncertainty

        

        Returns:

            predictions: [B, n_classes] or [B, T, n_classes]

            uncertainty: [B, n_classes] epistemic uncertainty (if mc_uncertainty=True)

        """

        # Temporal pooling

        if pool:

            z_pooled = torch.mean(z, dim=1)  # [B, d_latent]

        else:

            z_pooled = z  # [B, T, d_latent]

        

        if mc_uncertainty:

            # MC Dropout for epistemic uncertainty

            self.train()  # Enable dropout

            predictions_list = []

            for _ in range(self.mc_samples):

                pred = self.predictor(z_pooled)

                predictions_list.append(pred)

            

            predictions = torch.stack(predictions_list, dim=0)  # [mc_samples, B, ...]

            

            # Mean prediction

            mean_pred = predictions.mean(dim=0)

            

            # Epistemic uncertainty (variance across samples)

            epistemic_unc = predictions.var(dim=0)

            

            return mean_pred, epistemic_unc

        else:

            # Standard forward pass

            predictions = self.predictor(z_pooled)

            return predictions, None





if __name__ == "__main__":

    # Test stochastic decoder

    print("Stochastic Decoder with Uncertainty")

    print("=" * 60)

    

    B, T = 4, 100

    d_latent, n_roi = 128, 116

    

    # Test heteroscedastic mode

    print("\n1. Heteroscedastic Mode (input-dependent uncertainty)")

    decoder_hetero = StochasticDecoder(

        d_latent=d_latent,

        n_roi=n_roi,

        uncertainty_mode='heteroscedastic'

    )

    

    z = torch.randn(B, T, d_latent)

    mu, sigma = decoder_hetero(z, return_uncertainty=True)

    

    print(f"  Input: {z.shape}")

    print(f"  Mean output: {mu.shape}")

    print(f"  Std output: {sigma.shape}")

    print(f"  Std range: [{sigma.min():.4f}, {sigma.max():.4f}]")

    

    # Test NLL loss

    x_target = torch.randn(B, T, n_roi)

    nll = decoder_hetero.gaussian_nll_loss(z, x_target)

    print(f"  NLL loss: {nll.item():.4f}")

    

    # Test sampling

    samples = decoder_hetero.sample(z, n_samples=5)

    print(f"  Samples shape: {samples.shape}")

    

    # Test homoscedastic mode

    print("\n2. Homoscedastic Mode (global uncertainty)")

    decoder_homo = StochasticDecoder(

        d_latent=d_latent,

        n_roi=n_roi,

        uncertainty_mode='homoscedastic'

    )

    

    mu, sigma = decoder_homo(z, return_uncertainty=True)

    print(f"  Mean output: {mu.shape}")

    print(f"  Std output: {sigma.shape}")

    print(f"  Std (unique values): {torch.unique(sigma).shape[0]}")

    

    # Test deterministic mode

    print("\n3. Deterministic Mode (no uncertainty)")

    decoder_det = StochasticDecoder(

        d_latent=d_latent,

        n_roi=n_roi,

        uncertainty_mode='none'

    )

    

    mu, sigma = decoder_det(z, return_uncertainty=True)

    print(f"  Mean output: {mu.shape}")

    print(f"  Std output: {sigma}")

    

    # Test cognitive readout with MC Dropout

    print("\n4. Cognitive Readout with MC Uncertainty")

    cognitive = CognitiveStateReadoutStochastic(

        d_latent=d_latent,

        n_classes=4,

        task_type='classification',

        mc_samples=10

    )

    

    pred, unc = cognitive(z, pool=True, mc_uncertainty=True)

    print(f"  Predictions: {pred.shape}")

    print(f"  Epistemic uncertainty: {unc.shape}")

    print(f"  Uncertainty range: [{unc.min():.4f}, {unc.max():.4f}]")

    

    # Parameter counts

    total_params_hetero = sum(p.numel() for p in decoder_hetero.parameters())

    total_params_homo = sum(p.numel() for p in decoder_homo.parameters())

    total_params_det = sum(p.numel() for p in decoder_det.parameters())

    

    print(f"\nParameter Counts:")

    print(f"  Heteroscedastic: {total_params_hetero:,}")

    print(f"  Homoscedastic: {total_params_homo:,}")

    print(f"  Deterministic: {total_params_det:,}")




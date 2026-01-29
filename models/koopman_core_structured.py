"""

Structured Koopman Core with Low-Rank Parameterization

Implements parameter-efficient DPLR (Diagonal Plus Low-Rank) decomposition

and Neural SDE dynamics for modeling individual differences.



Contribution 1: Parameter Reduction via Approximation Theory

- K = diag(d) + UV^T (rank r << d_latent)

- C = AB^T (low-rank control)

- Reduces params from O(d^2) to O(d*r)



Contribution 2: Stochastic Dynamics for Individual Differences

- dz = (Kz + Cu + b)dt + Σ(z,u)dW_t

- Deterministic drift (Kz + Cu): learns true underlying mode (robust to noise)

- Stochastic diffusion σ(z,u): models individual-specific variations

- u contains subject-specific embedding, allowing diffusion to learn person-to-person differences

"""



import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Tuple, Optional, Dict

import math





class DPLRKoopmanCore(nn.Module):

    """

    Structured Koopman Core with DPLR + SDE dynamics

    

    Dynamics: dz = (Kz + Cu + b)dt + Σ(z,u)dW_t

    

    where:

    - K = diag(d) + UV^T  (DPLR: diagonal + rank-r) - learns true underlying mode

    - C = AB^T            (low-rank control)

    - Σ = diag(σ(z,u))    (learned diffusion for individual differences)

    - u contains subject-specific embedding information

    

    Philosophy:

    - Deterministic drift (Kz + Cu): Captures the true underlying dynamics,

      robust to measurement noise. This is the "genuine mode" shared across individuals.

      The control signal u contains subject info, but the drift learns the shared mode.

    - Stochastic diffusion σ(z,u): Models individual-specific variations and

      person-to-person differences, NOT measurement noise. The network learns to

      extract individual differences from u.

    """

    

    def __init__(

        self,

        d_latent: int,

        d_control: int,

        rank: int = 16,

        spectral_radius: float = 0.95,

        use_sde: bool = True,

        diffusion_mode: str = 'learned',  # 'learned', 'fixed', 'none'

        init_noise_std: float = 0.01,

        use_bias: bool = True

    ):

        """

        Args:

            d_latent: Latent state dimension

            d_control: Control signal dimension

            rank: Rank for low-rank decomposition (r << d_latent)

            spectral_radius: Target spectral radius for stability

            use_sde: Whether to use stochastic dynamics

            diffusion_mode: How to model diffusion ('learned', 'fixed', 'none')

            init_noise_std: Initial noise level

            use_bias: Whether to use bias term

        """

        super().__init__()

        self.d_latent = d_latent

        self.d_control = d_control

        self.rank = rank

        self.spectral_radius = spectral_radius

        self.use_sde = use_sde

        self.diffusion_mode = diffusion_mode

        

        # === DPLR Koopman Matrix: K = diag(d) + UV^T ===

        # Diagonal component (always stable)

        self.diag = nn.Parameter(torch.zeros(d_latent))

        

        # Low-rank component UV^T

        self.U = nn.Parameter(torch.randn(d_latent, rank) * (1.0 / math.sqrt(rank)))

        self.V = nn.Parameter(torch.randn(d_latent, rank) * (1.0 / math.sqrt(rank)))

        

        # === Low-Rank Control Matrix: C = AB^T ===

        control_rank = min(rank, d_control)

        self.A = nn.Parameter(torch.randn(d_latent, control_rank) * (1.0 / math.sqrt(control_rank)))

        self.B = nn.Parameter(torch.randn(d_control, control_rank) * (1.0 / math.sqrt(control_rank)))

        

        # Bias term

        if use_bias:

            self.bias = nn.Parameter(torch.zeros(d_latent))

        else:

            self.register_parameter('bias', None)

        

        # === Stochastic Dynamics (SDE) for Individual Differences ===

        if use_sde and diffusion_mode == 'learned':

            # Learn subject-dependent diffusion: σ = softplus(g(z, u, s))

            # This models individual variations, NOT measurement noise

            # Input: [z, u] or [z, u, subject_embedding] if subject-aware

            self.diffusion_net = nn.Sequential(

                nn.Linear(d_latent + d_control, d_latent),

                nn.Tanh(),

                nn.Linear(d_latent, d_latent)

            )

            # Initialize to moderate values (individual differences can be substantial)

            with torch.no_grad():

                self.diffusion_net[-1].weight.data *= 0.1  # Allow more variation

                self.diffusion_net[-1].bias.data.fill_(math.log(init_noise_std))

        elif use_sde and diffusion_mode == 'fixed':

            self.register_buffer('fixed_noise_std', torch.tensor(init_noise_std))

        

        # Initialize for stability

        self._initialize_stable()

    

    def _initialize_stable(self):

        """Initialize parameters for stable dynamics"""

        with torch.no_grad():

            # Diagonal should start stable (negative real parts for continuous-time)

            # For discrete-time stability: |eigenvalues| < spectral_radius

            self.diag.data.uniform_(-self.spectral_radius * 0.5, self.spectral_radius * 0.5)

            

            # Low-rank components start small

            nn.init.orthogonal_(self.U)

            nn.init.orthogonal_(self.V)

            self.U.data *= 0.1

            self.V.data *= 0.1

            

            # Control matrices start small

            nn.init.xavier_uniform_(self.A, gain=0.1)

            nn.init.xavier_uniform_(self.B, gain=0.1)

    

    def get_K_matrix(self) -> torch.Tensor:

        """

        Compute full K matrix: K = diag(d) + UV^T

        

        Returns:

            K: [d_latent, d_latent] Koopman operator

        """

        # Diagonal component

        K_diag = torch.diag(self.diag)

        

        # Low-rank component

        K_lr = torch.mm(self.U, self.V.T)

        

        return K_diag + K_lr

    

    def get_C_matrix(self) -> torch.Tensor:

        """

        Compute full C matrix: C = AB^T

        

        Returns:

            C: [d_latent, d_control] Control matrix

        """

        return torch.mm(self.A, self.B.T)

    

    def _apply_spectral_constraint(self):

        """

        Apply spectral radius constraint to K matrix

        More efficient for DPLR structure

        """

        with torch.no_grad():

            try:

                # Get full K matrix

                K = self.get_K_matrix()

                

                # Compute eigenvalues

                eigenvals = torch.linalg.eigvals(K)

                max_eigenval = torch.max(torch.abs(eigenvals))

                

                # Scale if exceeds target

                if max_eigenval > self.spectral_radius:

                    scale_factor = self.spectral_radius / (max_eigenval.real + 1e-8)

                    # Scale both components

                    self.diag.data *= scale_factor

                    self.U.data *= torch.sqrt(scale_factor)

                    self.V.data *= torch.sqrt(scale_factor)

            except:

                # Fallback: constrain diagonal component only

                max_diag = torch.max(torch.abs(self.diag))

                if max_diag > self.spectral_radius:

                    self.diag.data *= (self.spectral_radius / (max_diag + 1e-8))

    

    def compute_diffusion(self, z_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:

        """

        Compute diffusion σ(z, u) for individual differences

        

        This models person-specific variations from the true underlying mode,

        NOT measurement noise. The deterministic drift (Kz + Cu) learns the

        true mode that is robust to noise.

        

        The control signal u already contains subject-specific embedding information,

        so the diffusion network learns to extract individual differences from u.

        

        Args:

            z_t: [B, T, d_latent] Current state

            u_t: [B, T, d_control] Control input (contains subject-specific info)

        

        Returns:

            sigma: [B, T, d_latent] Diffusion coefficients (positive)

                   Models individual differences, not noise

        """

        if not self.use_sde or self.diffusion_mode == 'none':

            return None

        

        if self.diffusion_mode == 'fixed':

            return self.fixed_noise_std.expand_as(z_t)

        

        # Learned diffusion for individual differences: σ = softplus(g(z, u))

        # u already contains subject-specific embedding, so the network learns

        # to extract individual variation patterns from it

        combined = torch.cat([z_t, u_t], dim=-1)  # [B, T, d_latent + d_control]

        log_sigma = self.diffusion_net(combined)  # [B, T, d_latent]

        sigma = F.softplus(log_sigma) + 1e-6  # Ensure positivity

        

        return sigma

    

    def step(

        self,

        z_t: torch.Tensor,

        u_t: torch.Tensor,

        dt: float = 1.0,

        sample: bool = True,

        return_components: bool = False

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """

        Single-step Koopman-SDE with explicit drift and diffusion

        

        The deterministic drift (Kz + Cu) learns the true underlying mode,

        robust to measurement noise. The stochastic term models individual differences.

        

        The control signal u already contains subject-specific embedding information,

        allowing the diffusion to learn person-to-person variations.

        

        Args:

            z_t: [B, *, d_latent] Current latent state

            u_t: [B, *, d_control] Control signal (contains subject-specific info)

            dt: Time step (default: 1.0)

            sample: Whether to sample from SDE (True) or return mean (False)

            return_components: Return full dict with components

        

        Returns:

            z_next: [B, *, d_latent] Next state (sampled or mean)

            mu_K: [B, *, d_latent] Koopman drift mean (true underlying mode)

            sigma_K: [B, *, d_latent] SDE diffusion std (individual differences)

        """

        # Apply spectral constraint during training

        if self.training:

            self._apply_spectral_constraint()

        

        # === Koopman Drift: μ_K = z_t + (Kz + Cu + b)·dt ===

        # This learns the TRUE UNDERLYING MODE, robust to noise

        

        # Efficient DPLR computation

        # Kz = diag(d)·z + U(V^T·z)

        drift_K = self.diag.unsqueeze(0).unsqueeze(0) * z_t  # Diagonal part

        drift_K += torch.matmul(z_t, self.U) @ self.V.T      # Low-rank part

        

        # Cu = A(B^T·u)

        # u contains subject-specific embedding, but the drift learns the shared mode

        drift_C = torch.matmul(u_t, self.B) @ self.A.T

        

        # Total drift (true mode, not affected by individual differences)

        drift = drift_K + drift_C

        if self.bias is not None:

            drift = drift + self.bias

        

        # Mean prediction (the genuine underlying dynamics)

        mu_K = z_t + drift * dt

        

        # === SDE Diffusion: σ_K(z,u) for Individual Differences ===

        # This models person-specific variations, NOT measurement noise

        # u already contains subject-specific embedding, so diffusion learns from it

        sigma_K = self.compute_diffusion(z_t, u_t)

        if sigma_K is None:

            sigma_K = torch.zeros_like(z_t)

        

        # === Sample or return mean ===

        if sample and self.training:

            # Sample: z_next = μ_K + σ_K·√dt·ε

            # μ_K: true underlying mode (robust to noise)

            # σ_K·ε: individual-specific variation (learned from u)

            noise = torch.randn_like(z_t)

            sqrt_dt = math.sqrt(dt)

            z_next = mu_K + sigma_K * noise * sqrt_dt

        else:

            z_next = mu_K

        

        if return_components:

            return {

                'z_next': z_next,

                'mu_K': mu_K,

                'sigma_K': sigma_K,

                'drift': drift,

                'drift_K': drift_K,

                'drift_C': drift_C

            }

        

        return z_next, mu_K, sigma_K

    

    def forward(

        self, 

        z_t: torch.Tensor, 

        u_t: torch.Tensor,

        return_uncertainty: bool = False

    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """

        Forward pass for individual differences modeling

        

        The control signal u already contains subject-specific embedding information,

        allowing the diffusion to learn person-to-person variations.

        

        Args:

            z_t: [B, T, d_latent] Current latent state

            u_t: [B, T, d_control] Control signal (contains subject-specific info)

            return_uncertainty: Whether to return uncertainty estimate

        

        Returns:

            z_next: [B, T, d_latent] Next latent state (mean + individual variation)

            sigma: [B, T, d_latent] Diffusion coefficients for individual differences

                   (if return_uncertainty=True)

        """

        z_next, mu_K, sigma_K = self.step(z_t, u_t, dt=1.0, sample=self.training)

        

        if return_uncertainty:

            return z_next, sigma_K

        else:

            return z_next, None

    

    def get_spectral_radius(self) -> float:

        """

        Compute current spectral radius of K matrix

        

        Returns:

            spectral_radius: float - Largest eigenvalue magnitude

        """

        with torch.no_grad():

            try:

                K = self.get_K_matrix()

                eigenvals = torch.linalg.eigvals(K)

                spectral_radius = torch.max(torch.abs(eigenvals))

                return spectral_radius.real.item()

            except:

                # Fallback: use norm of diagonal + Frobenius of low-rank

                diag_norm = torch.max(torch.abs(self.diag)).item()

                lr_norm = torch.norm(self.U @ self.V.T, p=2).item()

                return diag_norm + lr_norm

    

    def get_num_params(self) -> dict:

        """

        Count parameters vs. dense baseline

        

        Returns:

            dict with param counts

        """

        # DPLR K params: d + 2*d*r

        k_params = self.d_latent + 2 * self.d_latent * self.rank

        

        # Low-rank C params: d*r_c + c*r_c

        control_rank = min(self.rank, self.d_control)

        c_params = self.d_latent * control_rank + self.d_control * control_rank

        

        # Dense baseline

        k_params_dense = self.d_latent * self.d_latent

        c_params_dense = self.d_latent * self.d_control

        

        # Diffusion net params

        if hasattr(self, 'diffusion_net'):

            diffusion_params = sum(p.numel() for p in self.diffusion_net.parameters())

        else:

            diffusion_params = 0

        

        return {

            'k_params_structured': k_params,

            'k_params_dense': k_params_dense,

            'k_reduction': k_params_dense / max(k_params, 1),

            'c_params_structured': c_params,

            'c_params_dense': c_params_dense,

            'c_reduction': c_params_dense / max(c_params, 1),

            'diffusion_params': diffusion_params,

            'total_structured': k_params + c_params + diffusion_params,

            'total_dense': k_params_dense + c_params_dense

        }

    

    def rollout(

        self,

        z_0: torch.Tensor,

        controls: torch.Tensor,

        return_trajectory: bool = True,

        return_uncertainty: bool = False

    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """

        Multi-step rollout for long-term prediction

        

        Args:

            z_0: [B, d_latent] Initial state

            controls: [B, T, d_control] Control sequence

            return_trajectory: Whether to return full trajectory

            return_uncertainty: Whether to return uncertainty estimates

        

        Returns:

            trajectory: [B, T+1, d_latent] if return_trajectory else [B, d_latent]

            uncertainties: [B, T+1, d_latent] if return_uncertainty else None

        """

        B, T, _ = controls.shape

        

        if return_trajectory:

            trajectory = [z_0]

            uncertainties = [] if return_uncertainty else None

            z_t = z_0

            

            for t in range(T):

                u_t = controls[:, t:t+1, :]  # [B, 1, d_control]

                z_t, sigma = self.forward(z_t.unsqueeze(1), u_t, return_uncertainty=return_uncertainty)

                z_t = z_t.squeeze(1)

                trajectory.append(z_t)

                

                if return_uncertainty and sigma is not None:

                    uncertainties.append(sigma.squeeze(1))

            

            traj = torch.stack(trajectory, dim=1)  # [B, T+1, d_latent]

            

            if return_uncertainty:

                unc = torch.stack(uncertainties, dim=1) if uncertainties else None

                return traj, unc

            else:

                return traj, None

        else:

            z_t = z_0

            for t in range(T):

                u_t = controls[:, t:t+1, :]

                z_t, _ = self.forward(z_t.unsqueeze(1), u_t, return_uncertainty=False)

                z_t = z_t.squeeze(1)

            

            return z_t, None





if __name__ == "__main__":

    # Test structured Koopman core

    print("Structured DPLR Koopman Core")

    print("=" * 60)

    

    B, T = 4, 100

    d_latent, d_control = 128, 64

    rank = 16

    

    # Create structured core

    koopman = DPLRKoopmanCore(

        d_latent=d_latent,

        d_control=d_control,

        rank=rank,

        spectral_radius=0.95,

        use_sde=True,

        diffusion_mode='learned'

    )

    

    # Test forward pass

    z_t = torch.randn(B, T, d_latent)

    u_t = torch.randn(B, T, d_control)

    

    z_next, sigma = koopman(z_t, u_t, return_uncertainty=True)

    

    print(f"\nInput shapes:")

    print(f"  z_t: {z_t.shape}")

    print(f"  u_t: {u_t.shape}")

    print(f"\nOutput shapes:")

    print(f"  z_next: {z_next.shape}")

    print(f"  sigma: {sigma.shape if sigma is not None else None}")

    

    # Parameter analysis

    params = koopman.get_num_params()

    print(f"\nParameter Analysis:")

    print(f"  K matrix (DPLR):")

    print(f"    Structured: {params['k_params_structured']:,}")

    print(f"    Dense baseline: {params['k_params_dense']:,}")

    print(f"    Reduction: {params['k_reduction']:.2f}x")

    print(f"  C matrix (low-rank):")

    print(f"    Structured: {params['c_params_structured']:,}")

    print(f"    Dense baseline: {params['c_params_dense']:,}")

    print(f"    Reduction: {params['c_reduction']:.2f}x")

    print(f"  Diffusion network: {params['diffusion_params']:,}")

    print(f"  Total:")

    print(f"    Structured: {params['total_structured']:,}")

    print(f"    Dense: {params['total_dense']:,}")

    print(f"    Overall reduction: {params['total_dense'] / params['total_structured']:.2f}x")

    

    # Test spectral radius

    print(f"\nSpectral radius: {koopman.get_spectral_radius():.4f}")

    

    # Test rollout with uncertainty

    z_0 = torch.randn(B, d_latent)

    controls = torch.randn(B, T, d_control)

    trajectory, uncertainties = koopman.rollout(z_0, controls, return_uncertainty=True)

    print(f"\nRollout:")

    print(f"  Trajectory: {trajectory.shape}")

    print(f"  Uncertainties: {uncertainties.shape if uncertainties is not None else None}")




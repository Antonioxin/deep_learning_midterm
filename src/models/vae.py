"""
Fully-connected VAE.

Architecture follows Kingma & Welling (2013) "Auto-Encoding Variational Bayes",
Appendix C / Section 4.2: two hidden layers of size 400, latent dim 20.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    q(z | x): maps input x to Gaussian parameters (mu, log_var).

    Output parameterises q_phi(z|x) = N(mu, diag(exp(log_var))).
    Kingma & Welling (2013) Eq. 9 (recognition model).
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """
    p(x | z): maps latent z back to pixel probabilities.

    Output is in (0, 1) via Sigmoid, consistent with Bernoulli likelihood
    used in the BCE reconstruction loss.
    Kingma & Welling (2013) Eq. 9 (generative model).
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """
    Standard Variational Autoencoder.

    Combines Encoder and Decoder with the reparameterization trick
    so gradients flow through the stochastic sampling step.
    Kingma & Welling (2013) Section 3 / Algorithm 1.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma,  eps ~ N(0, I).

        sigma = exp(0.5 * log_var)

        Kingma & Welling (2013) Eq. 4 — makes sampling differentiable
        by pushing randomness into eps, which is independent of phi.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 784) flattened images in [0, 1]

        Returns:
            recon_x:  (batch, 784) reconstructed pixel probabilities
            mu:       (batch, latent_dim)
            log_var:  (batch, latent_dim)
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n images by drawing z ~ N(0, I) and decoding."""
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)

"""
Convolutional VAE for 32x32 RGB images (CIFAR-10).

Architecture uses strided convolutions for downsampling (encoder) and
transposed convolutions for upsampling (decoder), with Batch Normalisation
for training stability on colour images.

Designed to replace the FC VAE on CIFAR-10 where spatial structure matters.
Interface is identical to VAE in vae.py: forward() returns (recon_x, mu, log_var).
"""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Maps (B, 3, 32, 32) → (mu, log_var) each of shape (B, latent_dim).

    Three strided-conv layers halve spatial dims at each step:
    32 → 16 → 8 → 4, then flatten to 2048-d before linear projections.
    BatchNorm after each conv (except the projection layers) improves
    gradient flow and reduces sensitivity to lr on colour data.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   64,  kernel_size=4, stride=2, padding=1),  # →64×16×16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1),  # →128×8×8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # →256×4×4
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(start_dim=1)   # (B, 4096)
        return self.fc_mu(h), self.fc_log_var(h)


class ConvDecoder(nn.Module):
    """
    Maps z (B, latent_dim) → (B, 3, 32, 32) pixel probabilities in (0,1).

    Projects z to a 128×4×4 feature map, then three transposed-conv layers
    upsample back to 3×32×32. Sigmoid on the final layer to match the
    Bernoulli BCE reconstruction loss.

    Note: no BN on the last layer — BN before Sigmoid disturbs the output range.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # →128×8×8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # →64×16×16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,   3, kernel_size=4, stride=2, padding=1),   # →3×32×32
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z)).view(-1, 256, 4, 4)
        return self.deconv(h)


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for 32x32 RGB images.

    Drop-in replacement for VAE (vae.py) on spatial image data.
    Same reparameterization trick: z = mu + eps*sigma, eps~N(0,I).
    Kingma & Welling (2013) Eq. 4.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder   = ConvEncoder(latent_dim)
        self.decoder   = ConvDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * exp(0.5 * log_var),  eps ~ N(0, I)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 32, 32) images in [0, 1]
        Returns:
            recon_x:  (B, 3, 32, 32) reconstructed pixel probabilities
            mu:       (B, latent_dim)
            log_var:  (B, latent_dim)
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n images by drawing z ~ N(0, I) and decoding. Returns (n, 3, 32, 32)."""
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)

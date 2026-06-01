"""
Two-level hierarchical convolutional VAE for CIFAR-10.

Probability structure:
  p(z_top) = N(0, I)
  p(z_bottom | z_top) = N(mu_p(z_top), diag(sigma_p^2(z_top)))
  q(z_top | x)
  q(z_bottom | x, z_top)

The top latent is a vector for global structure. The bottom latent is spatial
64x4x4 by default, so it can carry local texture and position-dependent detail.
"""

import torch
import torch.nn as nn


def _group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Pick a valid GroupNorm group count that divides channels."""
    groups = min(num_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    """Resolution-preserving residual block with GroupNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class DownBlock(nn.Module):
    """Stride-2 downsampling block followed by a residual refinement block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            _group_norm(out_ch),
            nn.ReLU(),
        )
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """Transposed-convolution upsampling block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            _group_norm(out_ch),
            nn.ReLU(),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalConvVAE(nn.Module):
    """
    Two-level top-down VAE for 32x32 RGB images.

    Args:
        top_latent_dim: vector latent dimension for global structure.
        bottom_latent_channels: channels of spatial latent at 4x4.
        base_channels: width multiplier. Default 64 matches ConvVAE v4.
    """

    def __init__(
        self,
        top_latent_dim: int = 64,
        bottom_latent_channels: int = 64,
        base_channels: int = 64,
        version: str = "hier_v1",
    ) -> None:
        super().__init__()
        c = base_channels
        self.version = version
        self.top_latent_dim = top_latent_dim
        self.bottom_latent_channels = bottom_latent_channels
        self.bottom_spatial = 4
        self.latent_dim = top_latent_dim
        if version == "hier_v2":
            # A less deterministic posterior makes the spatial bottom latent
            # harder to use as a pure autoencoder code.
            self.log_var_min = -4.0
            self.log_var_max = 4.0
        elif version == "hier_v3":
            # v2 escaped by setting bottom prior/posterior sigma to exp(2).
            # Keep variance at or below unit scale so KL remains meaningful.
            self.log_var_min = -4.0
            self.log_var_max = 0.0
        else:
            self.log_var_min = -8.0
            self.log_var_max = 8.0

        # Bottom-up encoder: 32 -> 16 -> 8 -> 4 -> 2.
        self.enc1 = DownBlock(3, c)
        self.enc2 = DownBlock(c, c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)
        self.enc4 = DownBlock(c * 4, c * 8)

        top_flat = c * 8 * 2 * 2
        self.top_mu = nn.Linear(top_flat, top_latent_dim)
        self.top_log_var = nn.Linear(top_flat, top_latent_dim)

        # Top-down path creates a 4x4 feature map from z_top.
        self.top_fc = nn.Linear(top_latent_dim, c * 8 * 2 * 2)
        self.top_up = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            _group_norm(c * 4),
            nn.ReLU(),
            ResBlock(c * 4),
        )
        if version in ("hier_v2", "hier_v3"):
            self.top_up8 = UpBlock(c * 4, c * 2)
            self.top_up16 = UpBlock(c * 2, c)

        # Conditional prior p(z_bottom | z_top) and posterior q(z_bottom | x,z_top).
        self.bottom_prior = nn.Conv2d(
            c * 4, bottom_latent_channels * 2, kernel_size=3, padding=1
        )
        self.bottom_posterior = nn.Sequential(
            nn.Conv2d(c * 8, c * 4, kernel_size=3, padding=1),
            _group_norm(c * 4),
            nn.ReLU(),
            ResBlock(c * 4),
            nn.Conv2d(c * 4, bottom_latent_channels * 2, kernel_size=3, padding=1),
        )

        # Decoder combines the sampled 4x4 bottom latent with top-down features.
        self.dec_in = nn.Sequential(
            nn.Conv2d(c * 4 + bottom_latent_channels, c * 4, kernel_size=3, padding=1),
            _group_norm(c * 4),
            nn.ReLU(),
            ResBlock(c * 4),
        )
        self.dec_up8 = UpBlock(c * 4, c * 2)
        self.dec_up16 = UpBlock(c * 2, c)
        self.dec_out = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1),
            _group_norm(c),
            nn.ReLU(),
            ResBlock(c),
            nn.Conv2d(c, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f16 = self.enc1(x)
        f8 = self.enc2(f16)
        f4 = self.enc3(f8)
        f2 = self.enc4(f4)
        return f16, f8, f4, f2

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def top_to_feature(self, z_top: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.top_fc(z_top)).view(z_top.shape[0], -1, 2, 2)
        return self.top_up(h)

    def split_params(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = params.chunk(2, dim=1)
        return mu, log_var.clamp(min=self.log_var_min, max=self.log_var_max)

    def decode(self, z_top: torch.Tensor, z_bottom: torch.Tensor) -> torch.Tensor:
        top4 = self.top_to_feature(z_top)
        h = self.dec_in(torch.cat([top4, z_bottom], dim=1))
        h = self.dec_up8(h)
        if self.version in ("hier_v2", "hier_v3"):
            top8 = self.top_up8(top4)
            h = h + top8
        h = self.dec_up16(h)
        if self.version in ("hier_v2", "hier_v3"):
            top16 = self.top_up16(top8)
            h = h + top16
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, _, f4, f2 = self.encode_features(x)
        flat = f2.flatten(start_dim=1)

        top_mu = self.top_mu(flat)
        top_log_var = self.top_log_var(flat).clamp(
            min=self.log_var_min, max=self.log_var_max
        )
        z_top = self.reparameterize(top_mu, top_log_var)

        top4 = self.top_to_feature(z_top)
        bottom_prior_mu, bottom_prior_log_var = self.split_params(
            self.bottom_prior(top4)
        )
        bottom_mu, bottom_log_var = self.split_params(
            self.bottom_posterior(torch.cat([f4, top4], dim=1))
        )
        z_bottom = self.reparameterize(bottom_mu, bottom_log_var)

        recon = self.decode_from_top4(top4, z_bottom)
        stats = {
            "top_mu": top_mu,
            "top_log_var": top_log_var,
            "bottom_mu": bottom_mu,
            "bottom_log_var": bottom_log_var,
            "bottom_prior_mu": bottom_prior_mu,
            "bottom_prior_log_var": bottom_prior_log_var,
        }
        return recon, stats

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z_top = torch.randn(n, self.top_latent_dim, device=device)
        top4 = self.top_to_feature(z_top)
        bottom_mu, bottom_log_var = self.split_params(self.bottom_prior(top4))
        z_bottom = self.reparameterize(bottom_mu, bottom_log_var)
        return self.decode_from_top4(top4, z_bottom)

    def decode_from_top4(self, top4: torch.Tensor, z_bottom: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(torch.cat([top4, z_bottom], dim=1))
        h = self.dec_up8(h)
        if self.version in ("hier_v2", "hier_v3"):
            top8 = self.top_up8(top4)
            h = h + top8
        h = self.dec_up16(h)
        if self.version in ("hier_v2", "hier_v3"):
            top16 = self.top_up16(top8)
            h = h + top16
        return self.dec_out(h)

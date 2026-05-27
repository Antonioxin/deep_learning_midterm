"""
VAE loss: negative ELBO = reconstruction loss + beta * KL divergence.

Convention (critical for reproducibility):
  - Sum losses over pixels and over the batch, then divide by dataset_size.
  - NOT dividing by batch_size avoids loss magnitude changing with batch size.
  - This matches Kingma & Welling (2013) Eq. 10 where the bound is per datapoint.

References:
  - Kingma & Welling (2013) "Auto-Encoding Variational Bayes", Eq. 10
  - Higgins et al. (2017) "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework" (beta weighting)
"""

import torch
import torch.nn.functional as F


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    dataset_size: int,
    beta: float = 1.0,
    recon_type: str = "bce",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE loss (negative ELBO).

    Loss = recon_loss + beta * kl_loss

    recon_loss (recon_type="bce"):
      Binary cross-entropy — assumes Bernoulli likelihood p(x|z).
      Appropriate for binary images (MNIST). Kingma & Welling (2013) Appendix C.1.

    recon_loss (recon_type="mse"):
      Mean squared error — assumes Gaussian likelihood p(x|z) = N(recon_x, I).
      Appropriate for continuous-valued RGB images (CIFAR-10).
      -log p(x|z) ∝ ||x - recon_x||^2,  summed then divided by dataset_size.

    kl_loss:
      KL( q(z|x) || p(z) ) where p(z)=N(0,I), q(z|x)=N(mu, diag(exp(log_var))).
      Closed-form: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
      Kingma & Welling (2013) Appendix B, Eq. B.3.

    Args:
        recon_x:      reconstructed output, any shape matching x
        x:            original input in [0, 1]
        mu:           (batch, latent_dim)
        log_var:      (batch, latent_dim)
        dataset_size: total training samples N (normalisation constant)
        beta:         KL weight; beta=1 is standard VAE (Higgins et al. 2017)
        recon_type:   "bce" for Bernoulli, "mse" for Gaussian decoder

    Returns:
        total_loss, recon_loss, kl_loss  — all scalars, total is backpropagatable
    """
    if recon_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / dataset_size
    else:
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / dataset_size

    # KL divergence: -0.5 * sum_j(1 + log_var_j - mu_j^2 - exp(log_var_j))
    # Kingma & Welling (2013) Appendix B Eq. B.3
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / dataset_size

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

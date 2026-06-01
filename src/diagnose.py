"""
Post-training diagnostic: real vs reconstruction vs prior samples.

Usage:
    python src/diagnose.py --checkpoint experiments/.../checkpoints/final.pt \\
                           --config experiments/.../config.yaml
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from datasets import get_dataloader, get_device
from models.conv_vae import ConvVAE


def to_disp(t: torch.Tensor, pixel_range: str):
    imgs = ((t.cpu() + 1) * 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    if pixel_range != "11":
        imgs = t.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)
    return imgs


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE diagnostic grid")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    pixel_range = config["data"].get("pixel_range", "01")
    version = config["model"].get("version", "v2")
    latent_dim = config["model"]["latent_dim"]

    model = ConvVAE(latent_dim=latent_dim, version=version).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader, _ = get_dataloader(
        config["data"]["dataset"],
        batch_size=args.n,
        train=True,
        pixel_range=pixel_range,
    )
    x, _ = next(iter(loader))
    x = x.to(device)

    with torch.no_grad():
        mu, log_var = model.encoder(x)
        z_post = model.reparameterize(mu, log_var)
        recon = model.decoder(z_post)
        z_prior = torch.randn(args.n, latent_dim, device=device)
        fake = model.decoder(z_prior)

    kl_per_dim = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean(0)
    print("=== Encoder posterior q(z|x) ===")
    print(f"  mu: mean={mu.mean():+.4f}  std={mu.std():.4f}")
    print(f"  sigma: mean={(0.5 * log_var).exp().mean():.4f}")
    print(f"  KL/dim: mean={kl_per_dim.mean():.3f}  total/sample={kl_per_dim.sum():.1f}")
    print(f"  recon MSE: {((recon - x) ** 2).mean():.4f}")
    print(f"  prior cross-sample std: {fake.std(dim=0).mean():.4f}")
    print(f"  recon cross-sample std: {recon.std(dim=0).mean():.4f}")
    print(f"  real cross-sample std:  {x.std(dim=0).mean():.4f}")

    fig, axes = plt.subplots(3, args.n, figsize=(args.n * 1.5, 4.5))
    for i in range(args.n):
        axes[0, i].imshow(to_disp(x, pixel_range)[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(to_disp(recon, pixel_range)[i])
        axes[1, i].axis("off")
        axes[2, i].imshow(to_disp(fake, pixel_range)[i])
        axes[2, i].axis("off")
    axes[0, 0].set_ylabel("real", fontsize=10)
    axes[1, 0].set_ylabel("recon", fontsize=10)
    axes[2, 0].set_ylabel("prior", fontsize=10)
    plt.tight_layout()

    out = Path(args.out) if args.out else Path(args.checkpoint).parent.parent / "diag_real_recon_fake.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

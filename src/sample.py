"""
Generate samples from a trained VAE checkpoint.

Usage:
    python src/sample.py --checkpoint experiments/exp_.../checkpoints/final.pt \\
                         --config     experiments/exp_.../config.yaml \\
                         --n 64       \\
                         --out        out.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from datasets import get_device
from models.vae import VAE


def save_grid(images: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    """Save a (n, 28, 28) tensor as a grid PNG."""
    n = images.shape[0]
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved grid to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained VAE")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to experiment config.yaml")
    parser.add_argument("--n", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--out", default="samples.png", help="Output PNG path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()

    model = VAE(
        input_dim=784,
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = model.sample(args.n, device).cpu().view(args.n, 28, 28)
    save_grid(samples, Path(args.out))


if __name__ == "__main__":
    main()

"""
Training entry point for VAE.

Usage:
    python src/train.py --config configs/baseline.yaml --tag baseline
"""

import argparse
import csv
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets import get_dataloader, get_device
from models.vae import VAE
from losses import vae_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_exp_dir(base: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base) / f"exp_{ts}_{tag}"
    for sub in ("checkpoints", "samples", "logs"):
        (exp_dir / sub).mkdir(parents=True)
    return exp_dir


def save_sample_grid(
    model: VAE,
    device: torch.device,
    out_path: Path,
    input_dim: int = 784,
    n: int = 64,
) -> None:
    """Sample 64 images from N(0,I), save as 8x8 PNG grid.

    Layout is inferred from the tensor shape returned by model.sample():
      (n, 784)       → 28x28 grayscale  (FC VAE)
      (n, 3, H, W)   → H×W RGB          (ConvVAE)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    raw = model.sample(n, device).cpu()
    model.train()

    if raw.ndim == 4:             # ConvVAE: (n, C, H, W)
        imgs = raw.permute(0, 2, 3, 1).numpy().clip(0, 1)
        cmap = None
    else:                         # FC VAE: (n, flat_dim)
        flat = raw.shape[1]
        if flat == 784:
            imgs = raw.view(n, 28, 28).numpy()
        else:
            side = int(flat ** 0.5)
            imgs = raw.view(n, side, side).numpy()
        cmap = "gray"

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("vae_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict, tag: str) -> None:
    device = get_device()
    set_seed(config["training"]["seed"], device)

    # Data
    train_loader, dataset_size = get_dataloader(
        dataset_name=config["data"]["dataset"],
        batch_size=config["data"]["batch_size"],
        train=True,
    )

    # Model
    arch = config["model"].get("arch", "fc")
    input_dim = config["model"].get("input_dim", 784)
    if arch == "conv":
        from models.conv_vae import ConvVAE
        model = ConvVAE(latent_dim=config["model"]["latent_dim"]).to(device)
    else:
        model = VAE(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            latent_dim=config["model"]["latent_dim"],
        ).to(device)

    # Optimizer
    lr = config["training"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    beta_max       = config["training"]["beta"]
    epochs         = config["training"]["epochs"]
    recon_type     = config["training"].get("recon_loss", "bce")
    warmup_epochs  = config["training"].get("beta_warmup_epochs", 0)

    # Experiment directory
    exp_dir = make_exp_dir("experiments", tag)
    log = setup_logging(exp_dir / "logs" / "train.log")
    log.info(f"Experiment: {exp_dir.name}")
    log.info(f"Device: {device}  |  Dataset size: {dataset_size}")
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Save config snapshot
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Metrics CSV
    metrics_path = exp_dir / "logs" / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "recon_loss", "kl_loss", "beta"])

    # Training
    for epoch in range(1, epochs + 1):
        # Linear KL warmup: beta 0→beta_max over warmup_epochs
        # Bowman et al. (2016) "Generating Sentences from a Continuous Space" Sec 3.1
        current_beta = (
            min(beta_max, beta_max * epoch / warmup_epochs) if warmup_epochs > 0 else beta_max
        )

        model.train()
        total_loss_sum = recon_sum = kl_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}/{epochs}", leave=False)
        for x, _ in pbar:
            if arch == "conv":
                x = x.to(device)                    # keep (B, C, H, W)
            else:
                x = x.view(x.size(0), -1).to(device)  # flatten for FC

            optimizer.zero_grad()
            recon_x, mu, log_var = model(x)
            loss, recon_loss, kl_loss = vae_loss(
                recon_x, x, mu, log_var, dataset_size,
                beta=current_beta, recon_type=recon_type,
            )
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()
            recon_sum += recon_loss.item()
            kl_sum += kl_loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss  = total_loss_sum / n_batches
        avg_recon = recon_sum / n_batches
        avg_kl    = kl_sum / n_batches

        log.info(
            f"Epoch {epoch:>3}/{epochs}  "
            f"loss={avg_loss:.4f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  beta={current_beta:.3f}"
        )

        # Append metrics
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss, avg_recon, avg_kl, current_beta])

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = exp_dir / "checkpoints" / f"epoch_{epoch:02d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, ckpt_path)

        # Sample grid
        grid_path = exp_dir / "samples" / f"sample_epoch_{epoch:02d}.png"
        save_sample_grid(model, device, grid_path, input_dim=input_dim)

    # Final checkpoint
    torch.save({"epoch": epochs, "model_state": model.state_dict()},
               exp_dir / "checkpoints" / "final.pt")
    log.info(f"Training complete. Results saved to: {exp_dir}")

    # Placeholder README for the experiment
    with open(exp_dir / "README.md", "w") as f:
        f.write(f"# {exp_dir.name}\n\nTODO: 记录本次实验的关键发现和观察。\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VAE")
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    parser.add_argument("--tag", default="baseline", help="Experiment tag")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.tag)


if __name__ == "__main__":
    main()

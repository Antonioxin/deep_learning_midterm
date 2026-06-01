"""
第二阶段 VAE 训练入口（2-Stage VAE，Dai & Wipf 2019）。

用法：
    python src/train_stage2.py \
        --stage1 experiments/exp_20260527_092330_cifar10_conv_v4 \
        --config configs/cifar10_2stage_v5.yaml \
        --tag cifar10_2stage_v5

流程：
  1. 加载冻结的第一阶段 ConvVAE（用其 config + final.pt）
  2. 把 CIFAR-10 训练集编码一次，缓存每张图的 (μ, logvar)
  3. 训练 Stage2VAE 拟合第一阶段 latent 分布 q(z)
     （每个 batch 用 z=μ+σ⊙ε 重采样，保持与一阶段后验一致）
  4. 周期性保存两阶段采样网格，并在结束时输出
     N(0,I) 基线 vs 两阶段采样 的多样性对比图与指标
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import get_dataloader, get_device
from models.stage2_vae import Stage2VAE
from train import (
    build_model, make_exp_dir, set_seed, setup_logging, to_display_rgb,
)


# ---------------------------------------------------------------------------
# 第一阶段：加载并冻结，编码整个训练集
# ---------------------------------------------------------------------------

def load_stage1(stage1_dir: Path, device: torch.device):
    """从实验目录加载冻结的一阶段 ConvVAE 与其 config。"""
    cfg = yaml.safe_load((stage1_dir / "config.yaml").open())
    model, arch, _ = build_model(cfg, device)
    assert arch == "conv", "Stage-2 仅针对 ConvVAE 实验"
    ckpt = torch.load(stage1_dir / "checkpoints" / "final.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def encode_dataset(stage1, device, pixel_range: str):
    """把 CIFAR-10 训练集编码为 (μ, logvar) 张量缓存在内存。"""
    loader, _ = get_dataloader("cifar10", batch_size=512, train=True,
                               data_dir="data/", pixel_range=pixel_range)
    mus, logvars = [], []
    for x, _ in tqdm(loader, desc="Encoding stage-1 latents", leave=False):
        mu, log_var = stage1.encoder(x.to(device))
        mus.append(mu.cpu())
        logvars.append(log_var.cpu())
    return torch.cat(mus), torch.cat(logvars)


# ---------------------------------------------------------------------------
# 采样与可视化
# ---------------------------------------------------------------------------

@torch.no_grad()
def two_stage_sample(stage1, stage2, n, device, z_mean, z_std):
    """u~N(0,I) → Stage2 → 标准化 z → 反标准化 → Stage1 解码出图像。"""
    z_std_space = stage2.sample_latent(n, device)          # 标准化空间的 z
    z = z_std_space * z_std.to(device) + z_mean.to(device)  # 反标准化回一阶段 latent
    return stage1.decoder(z)


@torch.no_grad()
def save_grid(imgs_raw, out_path, pixel_range, title=None):
    """把 (n,3,32,32) 输出存成 8×8 网格 PNG。"""
    imgs = to_display_rgb(imgs_raw.cpu(), pixel_range)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img, vmin=0, vmax=1)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def save_comparison(stage1, stage2, device, out_path, pixel_range, z_mean, z_std):
    """左：N(0,I) 直接解码（一阶段基线）；右：两阶段采样。同一行同一随机种子。"""
    n = 32
    torch.manual_seed(0)
    base = stage1.decoder(torch.randn(n, stage1.latent_dim, device=device))
    twostage = two_stage_sample(stage1, stage2, n, device, z_mean, z_std)

    base_d = to_display_rgb(base.cpu(), pixel_range)
    two_d = to_display_rgb(twostage.cpu(), pixel_range)
    fig, axes = plt.subplots(8, 8, figsize=(9, 9))
    for col_block, imgs, name in [(0, base_d, "z~N(0,I)"), (4, two_d, "2-stage")]:
        for i in range(n):
            r, c = i // 4, i % 4
            ax = axes[r, c + col_block]
            ax.imshow(imgs[i], vmin=0, vmax=1)
            ax.axis("off")
    fig.text(0.27, 0.99, "Stage-1  z~N(0,I)", ha="center", va="top", fontsize=12)
    fig.text(0.75, 0.99, "2-Stage VAE", ha="center", va="top", fontsize=12)
    plt.tight_layout(pad=0.2, rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def diversity_std(imgs_raw):
    """解码图像逐像素跨样本 std 的均值（越高=样本越多样，越不糊）。"""
    return imgs_raw.std(dim=0).mean().item()


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-2 VAE on frozen Stage-1 latents")
    parser.add_argument("--stage1", required=True, help="一阶段实验目录（含 config.yaml 与 checkpoints/final.pt）")
    parser.add_argument("--config", required=True, help="二阶段超参 yaml")
    parser.add_argument("--tag", default="2stage", help="实验标签")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = get_device()
    set_seed(cfg["training"]["seed"], device)

    # --- 第一阶段：冻结 + 编码数据集 ---
    stage1, s1_cfg = load_stage1(Path(args.stage1), device)
    pixel_range = s1_cfg["data"]["pixel_range"]
    d1 = stage1.latent_dim

    mu_all, logvar_all = encode_dataset(stage1, device, pixel_range)
    N = mu_all.shape[0]
    # 标准化统计量（用 μ 点云；实测一阶段 std≈1，mean≈0）
    z_mean = mu_all.mean(0)
    z_std = mu_all.std(0).clamp_min(1e-6)

    # --- 第二阶段模型 ---
    stage2 = Stage2VAE(
        dim=d1,
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"].get("latent_dim", d1),
        depth=cfg["model"].get("depth", 3),
    ).to(device)

    lr = cfg["training"]["lr"]
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    warmup_epochs = cfg["training"].get("beta_warmup_epochs", 0)  # KL 从 0 线性升到 1
    optimizer = torch.optim.Adam(stage2.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg["training"].get("lr_eta_min", 1e-5)
    )

    exp_dir = make_exp_dir("experiments", args.tag)
    log = setup_logging(exp_dir / "logs" / "train.log")
    log.info(f"Experiment: {exp_dir.name}")
    log.info(f"Stage-1: {Path(args.stage1).name}  |  d1={d1}  |  encoded N={N}")
    log.info(f"Stage-2 params: {sum(p.numel() for p in stage2.parameters()):,}")
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump({**cfg, "stage1_dir": str(args.stage1)}, f, default_flow_style=False)

    metrics_path = exp_dir / "logs" / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "recon_nll", "kl", "beta", "gamma", "lr",
                                "std_base", "std_2stage"])

    # 一阶段基线多样性（固定参考）
    std_base = diversity_std(
        stage1.decoder(torch.randn(256, d1, device=device)).cpu()
    )

    mu_all = mu_all.to(device)
    std_all = torch.exp(0.5 * logvar_all).to(device)
    z_mean_d, z_std_d = z_mean.to(device), z_std.to(device)

    for epoch in range(1, epochs + 1):
        # beta warmup：前 warmup_epochs 个 epoch 内 KL 权重从 0 线性升到 1
        beta = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0

        stage2.train()
        perm = torch.randperm(N, device=device)
        loss_sum = recon_sum = kl_sum = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            # 用一阶段后验重采样 z = μ + σ⊙ε，再标准化喂给 Stage2
            z = mu_all[idx] + std_all[idx] * torch.randn_like(std_all[idx])
            z = (z - z_mean_d) / z_std_d

            optimizer.zero_grad()
            loss, recon, kl = stage2.loss(z, beta=beta)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item(); recon_sum += recon.item(); kl_sum += kl.item()
            n_batches += 1

        scheduler.step()
        gamma = stage2.log_gamma2.exp().sqrt().item()
        lr_now = optimizer.param_groups[0]["lr"]

        # 评估两阶段采样多样性
        stage2.eval()
        std_2 = diversity_std(
            two_stage_sample(stage1, stage2, 256, device, z_mean, z_std).cpu()
        )

        log.info(
            f"Epoch {epoch:>3}/{epochs}  loss={loss_sum/n_batches:.4f}  "
            f"recon_nll={recon_sum/n_batches:.4f}  kl={kl_sum/n_batches:.4f}  "
            f"beta={beta:.3f}  gamma={gamma:.4f}  lr={lr_now:.2e}  "
            f"std_base={std_base:.4f}  std_2stage={std_2:.4f}"
        )
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, loss_sum/n_batches, recon_sum/n_batches,
                                    kl_sum/n_batches, beta, gamma, lr_now, std_base, std_2])

        if epoch % 10 == 0 or epoch == epochs:
            save_grid(
                two_stage_sample(stage1, stage2, 64, device, z_mean, z_std),
                exp_dir / "samples" / f"sample_epoch_{epoch:03d}.png", pixel_range,
            )

    # --- 收尾：保存权重、对比图 ---
    torch.save({
        "epoch": epochs,
        "model_state": stage2.state_dict(),
        "z_mean": z_mean, "z_std": z_std,
        "stage1_dir": str(args.stage1),
    }, exp_dir / "checkpoints" / "final.pt")

    save_comparison(stage1, stage2, device,
                    exp_dir / "compare_baseline_vs_2stage.png", pixel_range,
                    z_mean, z_std)
    log.info(f"Final diversity: std_base={std_base:.4f}  std_2stage={std_2:.4f}  "
             f"(gap recovered vs 0.439 upper-bound)")
    log.info(f"Done. Results in {exp_dir}")
    print(f"\nEXP_DIR={exp_dir}")


if __name__ == "__main__":
    main()

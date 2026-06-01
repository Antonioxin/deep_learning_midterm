"""
潜空间分析可视化（用于报告插图）。

针对单层向量-latent 的 ConvVAE（如 v4，latent_dim=64），产出三张图：
  1. latent 插值        ：两张真实图编码后在 latent 空间线性插值再解码，看过渡是否平滑语义连续
  2. latent 维度遍历    ：固定基码，逐个改变最活跃的若干维，看每一维控制了什么
  3. 编码 t-SNE         ：测试集编码 μ 降到 2D，按 CIFAR-10 类别上色，看 latent 是否按语义聚类

用法：
    python src/visualize.py \
        --exp experiments/exp_20260527_092330_cifar10_conv_v4 \
        --out-dir experiments/exp_20260527_092330_cifar10_conv_v4/latent_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import get_dataloader, get_device
from train import build_model, to_display_rgb

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def load_model(exp_dir: Path, device):
    """加载单层向量-latent VAE（conv 或 fc）。返回 (model, pixel_range, arch)。"""
    cfg = yaml.safe_load((exp_dir / "config.yaml").open())
    model, arch, _ = build_model(cfg, device)
    assert arch in ("conv", "fc"), "visualize.py 针对单层向量-latent VAE（conv / fc）"
    ckpt = torch.load(exp_dir / "checkpoints" / "final.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # 默认 "01" 与 train.py 一致（MNIST baseline 未显式设 pixel_range）
    return model, cfg["data"].get("pixel_range", "01"), arch


def _enc_mu(model, x, arch):
    """编码取 (mu, log_var)；fc 需先展平。"""
    if arch == "fc":
        x = x.view(x.size(0), -1)
    return model.encoder(x)


@torch.no_grad()
def collect_mu(model, device, pixel_range, arch, dataset, n_max):
    """编码测试集，返回 (mu, labels)。"""
    loader, _ = get_dataloader(dataset, 256, train=False, data_dir="data/", pixel_range=pixel_range)
    mus, ys = [], []
    seen = 0
    for x, y in loader:
        mu, _ = _enc_mu(model, x.to(device), arch)
        mus.append(mu.cpu()); ys.append(y)
        seen += x.shape[0]
        if seen >= n_max:
            break
    return torch.cat(mus)[:n_max], torch.cat(ys)[:n_max]


# ---------------------------------------------------------------------------
# 1. latent 插值
# ---------------------------------------------------------------------------

@torch.no_grad()
def fig_interpolation(model, device, pixel_range, arch, dataset, cmap, out_path,
                     n_pairs=6, steps=10):
    loader, _ = get_dataloader(dataset, 256, train=False, data_dir="data/", pixel_range=pixel_range)
    x, y = next(iter(loader))
    x = x.to(device)
    # 取不同类别的图配对，过渡更直观
    idx_a = list(range(n_pairs))
    idx_b = list(range(n_pairs, 2 * n_pairs))
    mu, _ = _enc_mu(model, x, arch)
    mu_a, mu_b = mu[idx_a], mu[idx_b]

    alphas = torch.linspace(0, 1, steps, device=device)
    fig, axes = plt.subplots(n_pairs, steps, figsize=(steps, n_pairs))
    for r in range(n_pairs):
        z = (1 - alphas)[:, None] * mu_a[r][None] + alphas[:, None] * mu_b[r][None]
        imgs = to_display_rgb(model.decoder(z).cpu(), pixel_range)
        for c in range(steps):
            axes[r, c].imshow(imgs[c], cmap=cmap, vmin=0, vmax=1); axes[r, c].axis("off")
    axes[0, 0].set_title("img A", fontsize=8)
    axes[0, -1].set_title("img B", fontsize=8)
    fig.suptitle("Latent-space interpolation (linear, between encoded means)", fontsize=11)
    plt.tight_layout(pad=0.15, rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=120); plt.close(fig)
    print("saved", out_path)


# ---------------------------------------------------------------------------
# 2. latent 维度遍历
# ---------------------------------------------------------------------------

@torch.no_grad()
def fig_traversal(model, device, pixel_range, cmap, mu_all, base_mu, out_path,
                  n_dims=12, steps=9, span=3.0):
    """围绕一张真实图的编码 base_mu，改变最活跃的 n_dims 维（按聚合 std 排序）。

    基码用真实图编码（数据流形上的点）而非数据集均值——后者 z≈0 会解码成均值 mush，
    遍历不可解释。最左列标 ★ 为基码重建。
    """
    agg_std = mu_all.std(0)
    base = base_mu.to(device)
    top_dims = torch.argsort(agg_std, descending=True)[:n_dims]
    offsets = torch.linspace(-span, span, steps)
    mid = steps // 2  # 中间列 offset≈0，最接近基码

    fig, axes = plt.subplots(n_dims, steps, figsize=(steps, n_dims))
    for r, d in enumerate(top_dims.tolist()):
        z = base.repeat(steps, 1).clone()
        z[:, d] = base[d] + offsets.to(device) * agg_std[d].to(device)
        imgs = to_display_rgb(model.decoder(z).cpu(), pixel_range)
        for c in range(steps):
            axes[r, c].imshow(imgs[c], cmap=cmap, vmin=0, vmax=1); axes[r, c].axis("off")
            if r == 0 and c == mid:
                axes[r, c].set_title("base", fontsize=7)
        axes[r, 0].set_ylabel(f"d{d}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"Latent traversal around a real image: top-{n_dims} active dims, each ±{span}σ",
                 fontsize=11)
    plt.tight_layout(pad=0.15, rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=120); plt.close(fig)
    print("saved", out_path)


# ---------------------------------------------------------------------------
# 3. 编码 t-SNE
# ---------------------------------------------------------------------------

@torch.no_grad()
def fig_recon(model, device, pixel_range, arch, dataset, cmap, out_path, n=8):
    """真实图 vs 重建（上行真实、下行重建）。"""
    loader, _ = get_dataloader(dataset, 64, train=False, data_dir="data/", pixel_range=pixel_range)
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    mu, log_var = _enc_mu(model, x, arch)
    recon = model.decoder(model.reparameterize(mu, log_var))
    real_d = to_display_rgb(x.cpu(), pixel_range)
    recon_d = to_display_rgb(recon.cpu(), pixel_range)
    fig, axes = plt.subplots(2, n, figsize=(n, 2.4))
    for c in range(n):
        axes[0, c].imshow(real_d[c], cmap=cmap, vmin=0, vmax=1); axes[0, c].axis("off")
        axes[1, c].imshow(recon_d[c], cmap=cmap, vmin=0, vmax=1); axes[1, c].axis("off")
    axes[0, 0].set_ylabel("real", fontsize=9, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("recon", fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle("Reconstruction (top: real, bottom: recon)", fontsize=11)
    plt.tight_layout(pad=0.2, rect=(0, 0, 1, 0.92))
    plt.savefig(out_path, dpi=120); plt.close(fig)
    print("saved", out_path)


def fig_tsne(mu_all, labels, class_names, out_path, perplexity=30):
    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca",
               random_state=42).fit_transform(mu_all.numpy())
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in range(len(class_names)):
        m = labels.numpy() == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, label=class_names[cls])
    ax.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t-SNE of encoded means μ (n={len(labels)}), colored by class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close(fig)
    print("saved", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tsne-n", type=int, default=2500)
    args = parser.parse_args()

    device = get_device()
    exp_dir = Path(args.exp)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "latent_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, pixel_range, arch = load_model(exp_dir, device)
    cfg = yaml.safe_load((exp_dir / "config.yaml").open())
    dataset = cfg["data"]["dataset"]
    cmap = "gray" if dataset in ("mnist", "fashion_mnist") else None
    class_names = ([str(i) for i in range(10)] if dataset in ("mnist", "fashion_mnist")
                   else CIFAR_CLASSES)

    mu_all, labels = collect_mu(model, device, pixel_range, arch, dataset, args.tsne_n)
    print(f"[{dataset}/{arch}] encoded {mu_all.shape[0]} test images, latent_dim={mu_all.shape[1]}")

    fig_recon(model, device, pixel_range, arch, dataset, cmap, out_dir / "reconstruction.png")
    fig_interpolation(model, device, pixel_range, arch, dataset, cmap,
                      out_dir / "latent_interpolation.png")
    base_mu = mu_all[1]  # 用一张真实测试图的编码作为遍历基码
    fig_traversal(model, device, pixel_range, cmap, mu_all, base_mu,
                  out_dir / "latent_traversal.png")
    fig_tsne(mu_all, labels, class_names, out_dir / "latent_tsne.png")
    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()

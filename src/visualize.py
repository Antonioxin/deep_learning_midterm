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
    cfg = yaml.safe_load((exp_dir / "config.yaml").open())
    model, arch, _ = build_model(cfg, device)
    assert arch == "conv", "visualize.py 针对单层向量-latent ConvVAE"
    ckpt = torch.load(exp_dir / "checkpoints" / "final.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg["data"].get("pixel_range", "11")


@torch.no_grad()
def collect_mu(model, device, pixel_range, n_max):
    """编码测试集，返回 (mu, labels)。"""
    loader, _ = get_dataloader("cifar10", 256, train=False, data_dir="data/", pixel_range=pixel_range)
    mus, ys = [], []
    seen = 0
    for x, y in loader:
        mu, _ = model.encoder(x.to(device))
        mus.append(mu.cpu()); ys.append(y)
        seen += x.shape[0]
        if seen >= n_max:
            break
    return torch.cat(mus)[:n_max], torch.cat(ys)[:n_max]


# ---------------------------------------------------------------------------
# 1. latent 插值
# ---------------------------------------------------------------------------

@torch.no_grad()
def fig_interpolation(model, device, pixel_range, out_path, n_pairs=6, steps=10):
    loader, _ = get_dataloader("cifar10", 256, train=False, data_dir="data/", pixel_range=pixel_range)
    x, y = next(iter(loader))
    x = x.to(device)
    # 取不同类别的图配对，过渡更直观
    idx_a = list(range(n_pairs))
    idx_b = list(range(n_pairs, 2 * n_pairs))
    mu, _ = model.encoder(x)
    mu_a, mu_b = mu[idx_a], mu[idx_b]

    alphas = torch.linspace(0, 1, steps, device=device)
    fig, axes = plt.subplots(n_pairs, steps, figsize=(steps, n_pairs))
    for r in range(n_pairs):
        z = (1 - alphas)[:, None] * mu_a[r][None] + alphas[:, None] * mu_b[r][None]
        imgs = to_display_rgb(model.decoder(z).cpu(), pixel_range)
        for c in range(steps):
            axes[r, c].imshow(imgs[c], vmin=0, vmax=1); axes[r, c].axis("off")
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
def fig_traversal(model, device, pixel_range, mu_all, base_mu, out_path,
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
            axes[r, c].imshow(imgs[c], vmin=0, vmax=1); axes[r, c].axis("off")
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

def fig_tsne(mu_all, labels, out_path, perplexity=30):
    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca",
               random_state=42).fit_transform(mu_all.numpy())
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in range(10):
        m = labels.numpy() == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, label=CIFAR_CLASSES[cls])
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

    model, pixel_range = load_model(exp_dir, device)
    mu_all, labels = collect_mu(model, device, pixel_range, args.tsne_n)
    print(f"encoded {mu_all.shape[0]} test images, latent_dim={mu_all.shape[1]}")

    fig_interpolation(model, device, pixel_range, out_dir / "latent_interpolation.png")
    # 用一张真实测试图的编码作为遍历基码（取一张可辨认的，索引 1）
    base_mu = mu_all[1]
    fig_traversal(model, device, pixel_range, mu_all, base_mu, out_dir / "latent_traversal.png")
    fig_tsne(mu_all, labels, out_dir / "latent_tsne.png")
    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()

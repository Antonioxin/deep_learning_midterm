"""
从已训练 VAE checkpoint 生成样本并保存为 PNG 网格。

用法示例：
    python src/sample.py --checkpoint experiments/exp_.../checkpoints/final.pt \\
                         --config     experiments/exp_.../config.yaml \\
                         --n 64 \\
                         --out        out.png

注意：当前脚本仅构建全连接 VAE（MNIST 784 维），
ConvVAE 实验需自行扩展加载逻辑或复用 train.py 中的 build_model。
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
    """
    将 (n, H, W) 灰度张量排列成网格图并保存。

    Args:
        images:   (n, 28, 28) 等，值域假定已在 [0, 1]
        out_path: 输出 PNG 路径
        nrow:     每行图像数，不足处留空
    """
    n = images.shape[0]
    ncol = (n + nrow - 1) // nrow  # 向上取整行数
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
    """加载配置与权重，从先验采样并可视化。"""
    parser = argparse.ArgumentParser(description="Sample from a trained VAE")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to experiment config.yaml")
    parser.add_argument("--n", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--out", default="samples.png", help="Output PNG path")
    args = parser.parse_args()

    # 从实验目录复制的 config.yaml 读取模型结构超参
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()

    # 按配置实例化 VAE（默认 MNIST：784 输入）
    model = VAE(
        input_dim=784,
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    ).to(device)

    # weights_only=True 减少反序列化安全风险（PyTorch 2.0+）
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 采样并 reshape 为 28×28 便于 imshow
    samples = model.sample(args.n, device).cpu().view(args.n, 28, 28)
    save_grid(samples, Path(args.out))


if __name__ == "__main__":
    main()

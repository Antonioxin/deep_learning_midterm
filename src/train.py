"""
VAE 训练入口脚本。

用法示例：
    python src/train.py --config configs/baseline.yaml --tag baseline

流程概览：
  1. 读取 YAML 配置，创建带时间戳的实验目录
  2. 加载数据、构建模型（全连接 VAE 或 ConvVAE）
  3. 按 epoch 训练，记录 CSV 指标与日志，定期保存 checkpoint 与采样网格图
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
from losses import hierarchical_vae_loss, vae_loss


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def set_seed(seed: int, device: torch.device) -> None:
    """
    固定 Python / NumPy / PyTorch 随机种子，保证实验可复现。

    根据 device 类型额外设置 MPS 或 CUDA 的种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_exp_dir(base: str, tag: str) -> Path:
    """
    在 base 下创建 experiments/exp_YYYYMMDD_HHMMSS_{tag}/ 目录结构。

    子目录：
      checkpoints/  模型权重
      samples/      每 epoch 的采样可视化 PNG
      logs/         train.log 与 metrics.csv
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base) / f"exp_{ts}_{tag}"
    for sub in ("checkpoints", "samples", "logs"):
        (exp_dir / sub).mkdir(parents=True)
    return exp_dir


def to_display_rgb(raw: torch.Tensor, pixel_range: str) -> np.ndarray:
    """
    将模型输出张量转为 matplotlib 可显示的 [0, 1] numpy 图像。

    支持：
      - 4D 张量 (B,C,H,W)：RGB，按 pixel_range 从 [-1,1] 或保持 [0,1]
      - 2D 展平向量：MNIST 784→28×28，或任意正方形 side×side
    """
    if raw.ndim == 4:
        imgs = raw.permute(0, 2, 3, 1).numpy()  # NCHW -> NHWC
        if pixel_range == "11":
            imgs = (imgs + 1.0) * 0.5  # [-1,1] -> [0,1]
        return imgs.clip(0, 1)
    flat = raw.shape[1]
    if flat == 784:
        return raw.view(-1, 28, 28).numpy()
    side = int(flat ** 0.5)
    return raw.view(-1, side, side).numpy()


def save_sample_grid(
    model,
    device: torch.device,
    out_path: Path,
    input_dim: int = 784,
    n: int = 64,
    pixel_range: str = "01",
) -> None:
    """
    从先验 N(0,I) 采样 n 张图，排列成 8×8 网格保存为 PNG。

    用于训练过程中直观观察生成质量随 epoch 的变化。
    input_dim 参数保留以兼容旧接口，ConvVAE 实际按 4D 输出处理。
    """
    import matplotlib
    matplotlib.use("Agg")  # 无头环境，不弹窗
    import matplotlib.pyplot as plt

    model.eval()
    raw = model.sample(n, device).cpu()
    model.train()  # 恢复训练模式

    if raw.ndim == 4:
        imgs = to_display_rgb(raw, pixel_range)
        cmap = None  # RGB 彩色
    else:
        imgs = to_display_rgb(raw, pixel_range)
        cmap = "gray"  # 灰度 MNIST 等

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def setup_logging(log_path: Path) -> logging.Logger:
    """配置同时写入文件与终端的 logger。"""
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


def build_model(config: dict, device: torch.device):
    """
    根据 config["model"] 构建 VAE 并移到 device。

    arch="conv"/"hierarchical" 时延迟导入对应卷积模型，避免影响 MNIST 实验。
    """
    arch = config["model"].get("arch", "fc")
    input_dim = config["model"].get("input_dim", 784)
    if arch == "conv":
        from models.conv_vae import ConvVAE
        version = config["model"].get("version", "v2")
        model = ConvVAE(
            latent_dim=config["model"]["latent_dim"],
            version=version,
        ).to(device)
    elif arch == "hierarchical":
        from models.hierarchical_vae import HierarchicalConvVAE
        model = HierarchicalConvVAE(
            top_latent_dim=config["model"].get("top_latent_dim", 64),
            bottom_latent_channels=config["model"].get("bottom_latent_channels", 64),
            base_channels=config["model"].get("base_channels", 64),
            version=config["model"].get("version", "hier_v1"),
        ).to(device)
    else:
        model = VAE(
            input_dim=input_dim,
            hidden_dim=config["model"]["hidden_dim"],
            latent_dim=config["model"]["latent_dim"],
        ).to(device)
    return model, arch, input_dim


def compute_prior_sample_std(model, device: torch.device, n: int = 64) -> float:
    """
    从 N(0,I) 采样 n 张图，计算逐像素跨样本标准差的均值。

    prior hole 时该值会长期 < 0.3（所有先验样本几乎相同）。
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        fake = model.sample(n, device)
    if was_training:
        model.train()
    return fake.std(dim=0).mean().item()


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------

def train(config: dict, tag: str) -> None:
    """
    完整训练流程：读配置 → 训练多 epoch → 保存结果。

    Args:
        config: 从 YAML 解析的字典
        tag:    实验标签，出现在目录名中
    """
    device = get_device()
    set_seed(config["training"]["seed"], device)

    # 像素值域须与数据增强、解码器激活、感知损失预处理一致
    pixel_range = config["data"].get("pixel_range", "01")

    train_loader, dataset_size = get_dataloader(
        dataset_name=config["data"]["dataset"],
        batch_size=config["data"]["batch_size"],
        train=True,
        pixel_range=pixel_range,
    )

    model, arch, input_dim = build_model(config, device)

    lr = config["training"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 损失与调度相关超参
    is_hierarchical = arch == "hierarchical"
    beta_max = config["training"].get("beta", 1.0)  # KL 权重上限（beta-VAE）
    beta_top_max = config["training"].get("beta_top", beta_max)
    beta_bottom_max = config["training"].get("beta_bottom", beta_max)
    epochs = config["training"]["epochs"]
    recon_type = config["training"].get("recon_loss", "bce")
    warmup_epochs = config["training"].get("beta_warmup_epochs", 0)  # KL 线性 warmup
    free_bits = config["training"].get("free_bits", 0.0)
    free_bits_top = config["training"].get("free_bits_top", free_bits)
    free_bits_bottom = config["training"].get("free_bits_bottom", free_bits)
    lambda_pixel = config["training"].get("lambda_pixel", 1.0)
    lambda_perc = config["training"].get("lambda_perc", 0.0)

    # 仅当 lambda_perc > 0 时加载 VGG（占用显存且较慢）
    perceptual_fn = None
    if lambda_perc > 0:
        from perceptual import VGGPerceptualLoss
        perceptual_fn = VGGPerceptualLoss(pixel_range=pixel_range).to(device)

    scheduler = None
    sched_cfg = config["training"].get("lr_scheduler")
    if sched_cfg == "cosine":
        eta_min = config["training"].get("lr_eta_min", 1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )

    exp_dir = make_exp_dir("experiments", tag)
    log = setup_logging(exp_dir / "logs" / "train.log")
    log.info(f"Experiment: {exp_dir.name}")
    log.info(f"Device: {device}  |  Dataset size: {dataset_size}")
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    if perceptual_fn is not None:
        log.info(f"Perceptual loss: lambda_perc={lambda_perc}, lambda_pixel={lambda_pixel}")
    if free_bits > 0:
        log.info(f"Free bits KL floor: {free_bits} nats/dim")
    if is_hierarchical:
        log.info(
            "Hierarchical KL: "
            f"beta_top={beta_top_max}, beta_bottom={beta_bottom_max}, "
            f"free_bits_top={free_bits_top}, free_bits_bottom={free_bits_bottom}"
        )

    # 备份本次实验使用的完整配置
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    metrics_path = exp_dir / "logs" / "metrics.csv"
    header = [
        "epoch", "train_loss", "recon_loss", "kl_loss", "perc_loss",
        "beta", "lr", "prior_std", "kl_top", "kl_bottom",
    ]
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    for epoch in range(1, epochs + 1):
        # beta warmup：前 warmup_epochs 个 epoch 内从 0 线性增至 beta_max
        current_beta = (
            min(beta_max, beta_max * epoch / warmup_epochs) if warmup_epochs > 0 else beta_max
        )
        current_beta_top = (
            min(beta_top_max, beta_top_max * epoch / warmup_epochs)
            if warmup_epochs > 0 else beta_top_max
        )
        current_beta_bottom = (
            min(beta_bottom_max, beta_bottom_max * epoch / warmup_epochs)
            if warmup_epochs > 0 else beta_bottom_max
        )

        model.train()
        total_loss_sum = recon_sum = kl_sum = perc_sum = 0.0
        kl_top_sum = kl_bottom_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}/{epochs}", leave=False)
        for x, _ in pbar:
            # 卷积模型保持 (B,C,H,W)；全连接模型展平为 (B, input_dim)
            if arch in ("conv", "hierarchical"):
                x = x.to(device)
            else:
                x = x.view(x.size(0), -1).to(device)

            optimizer.zero_grad()
            if is_hierarchical:
                recon_x, stats = model(x)
                (
                    loss, recon_loss, kl_loss, perc_loss,
                    kl_top_loss, kl_bottom_loss,
                ) = hierarchical_vae_loss(
                    recon_x, x, stats, dataset_size,
                    beta_top=current_beta_top,
                    beta_bottom=current_beta_bottom,
                    recon_type=recon_type,
                    free_bits_top=free_bits_top,
                    free_bits_bottom=free_bits_bottom,
                    perceptual_fn=perceptual_fn,
                    lambda_pixel=lambda_pixel,
                    lambda_perc=lambda_perc,
                )
            else:
                recon_x, mu, log_var = model(x)
                loss, recon_loss, kl_loss, perc_loss = vae_loss(
                    recon_x, x, mu, log_var, dataset_size,
                    beta=current_beta,
                    recon_type=recon_type,
                    free_bits=free_bits,
                    perceptual_fn=perceptual_fn,
                    lambda_pixel=lambda_pixel,
                    lambda_perc=lambda_perc,
                )
                kl_top_loss = torch.tensor(float("nan"))
                kl_bottom_loss = torch.tensor(float("nan"))
            loss.backward()
            optimizer.step()

            # 累计 batch 平均用的分子（后面除以 n_batches）
            total_loss_sum += loss.item()
            recon_sum += recon_loss.item()
            kl_sum += kl_loss.item()
            perc_sum += perc_loss.item()
            kl_top_sum += kl_top_loss.item()
            kl_bottom_sum += kl_bottom_loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss_sum / n_batches
        avg_recon = recon_sum / n_batches
        avg_kl = kl_sum / n_batches
        avg_perc = perc_sum / n_batches
        avg_kl_top = kl_top_sum / n_batches
        avg_kl_bottom = kl_bottom_sum / n_batches
        current_lr = optimizer.param_groups[0]["lr"]

        prior_std = (
            compute_prior_sample_std(model, device)
            if arch in ("conv", "hierarchical") else float("nan")
        )

        if is_hierarchical:
            log.info(
                f"Epoch {epoch:>3}/{epochs}  "
                f"loss={avg_loss:.4f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  "
                f"kl_top={avg_kl_top:.4f}  kl_bottom={avg_kl_bottom:.4f}  "
                f"perc={avg_perc:.4f}  beta_top={current_beta_top:.3f}  "
                f"beta_bottom={current_beta_bottom:.3f}  lr={current_lr:.2e}  "
                f"prior_std={prior_std:.4f}"
            )
        else:
            log.info(
                f"Epoch {epoch:>3}/{epochs}  "
                f"loss={avg_loss:.4f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  "
                f"perc={avg_perc:.4f}  beta={current_beta:.3f}  lr={current_lr:.2e}  "
                f"prior_std={prior_std:.4f}"
            )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, avg_loss, avg_recon, avg_kl, avg_perc,
                current_beta_top if is_hierarchical else current_beta,
                current_lr, prior_std, avg_kl_top, avg_kl_bottom,
            ])

        # 每 5 个 epoch 存一次中间 checkpoint
        if epoch % 5 == 0:
            ckpt_path = exp_dir / "checkpoints" / f"epoch_{epoch:02d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)

        # 每个 epoch 都保存采样网格，便于对比训练进程
        grid_path = exp_dir / "samples" / f"sample_epoch_{epoch:02d}.png"
        save_sample_grid(
            model, device, grid_path,
            input_dim=input_dim, pixel_range=pixel_range,
        )

    # 训练结束保存最终权重
    torch.save({"epoch": epochs, "model_state": model.state_dict()},
               exp_dir / "checkpoints" / "final.pt")
    log.info(f"Training complete. Results saved to: {exp_dir}")

    # 占位 README，提醒手动记录实验观察
    with open(exp_dir / "README.md", "w") as f:
        f.write(f"# {exp_dir.name}\n\nTODO: 记录本次实验的关键发现和观察。\n")

    return exp_dir


def main() -> None:
    """命令行入口：解析 --config 与 --tag，加载 YAML 后调用 train()。"""
    parser = argparse.ArgumentParser(description="Train a VAE")
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    parser.add_argument("--tag", default="baseline", help="Experiment tag")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.tag)


if __name__ == "__main__":
    main()

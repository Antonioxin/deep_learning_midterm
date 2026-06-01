"""
VAE 训练损失：负 ELBO = 重建项 + beta * KL 项。

归一化约定（影响 loss 数值与论文对照，务必保持一致）：
  - 对 batch 内所有样本、所有像素/特征维度求和，再除以 dataset_size（全集样本数）
  - 不除以 batch_size，这样换 batch_size 时 loss 量级不变
  - 与 Kingma & Welling (2013) 式 (10) 中「按数据点平均」的写法一致

参考文献：
  - Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
  - Higgins et al. (2017) beta-VAE（beta 加权 KL）
  - Kingma et al. (2016) Free Bits（KL 下界截断）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_per_dim(
    mu: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """
    解析形式的高斯后验 q(z|x) 与标准正态先验 p(z) 的逐维 KL 散度。

    对每一维 d：KL = -0.5 * (1 + log_var - mu^2 - exp(log_var))

    Args:
        mu:      (batch, latent_dim) 编码器输出的均值
        log_var: (batch, latent_dim) 编码器输出的对数方差 log(σ²)

    Returns:
        (batch, latent_dim)，每个元素为该维的 KL（单位：nats）
    """
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())


def kl_divergence_diag_gaussians(
    q_mu: torch.Tensor,
    q_log_var: torch.Tensor,
    p_mu: torch.Tensor,
    p_log_var: torch.Tensor,
) -> torch.Tensor:
    """
    逐元素 KL[q || p]，其中 q 与 p 都是对角高斯。

    用于分层 VAE 的 bottom latent：
      q(z_bottom | x, z_top) 对 p(z_bottom | z_top)
    """
    q_log_var = q_log_var.clamp(min=-8.0, max=8.0)
    p_log_var = p_log_var.clamp(min=-8.0, max=8.0)
    return 0.5 * (
        p_log_var - q_log_var
        + (q_log_var.exp() + (q_mu - p_mu).pow(2)) / p_log_var.exp()
        - 1
    )


def reconstruction_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    dataset_size: int,
    recon_type: str = "bce",
    perceptual_fn: nn.Module | None = None,
    lambda_pixel: float = 1.0,
    lambda_perc: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    共享的重建损失：像素项 + 可选感知项，并按 dataset_size 归一化。

    Returns:
        recon_loss, perc_loss，二者均已除以 dataset_size。
    """
    if recon_type == "mse":
        pixel_loss = F.mse_loss(recon_x, x, reduction="sum")
    else:
        pixel_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

    perc_loss = torch.tensor(0.0, device=x.device)
    if perceptual_fn is not None and lambda_perc > 0:
        perc_loss = perceptual_fn(recon_x, x)

    recon_loss = (
        lambda_pixel * pixel_loss + lambda_perc * perc_loss
    ) / dataset_size
    return recon_loss, perc_loss / dataset_size


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    dataset_size: int,
    beta: float = 1.0,
    recon_type: str = "bce",
    free_bits: float = 0.0,
    perceptual_fn: nn.Module | None = None,
    lambda_pixel: float = 1.0,
    lambda_perc: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 VAE 总损失（负变分下界 ELBO 的相反数，即最小化 -ELBO）。

    总损失 = recon_loss + beta * kl_loss

    重建项 recon_type：
      - "bce"：二元交叉熵，假设像素为 Bernoulli，适合 MNIST 等二值/灰度图
      - "mse"：均方误差，假设高斯似然，适合 CIFAR-10 等连续 RGB

    free_bits > 0：
      对每维 KL 做 max(KL_d, free_bits) 截断，防止部分潜维被 KL 压到 0 而「坍缩」

    perceptual_fn + lambda_perc：
      在 VGG 特征空间增加重建项（Johnson et al. 2016）

    Args:
        recon_x:        解码器重建输出
        x:              原始输入（与 recon_x 同形状）
        mu, log_var:    编码器输出的高斯参数
        dataset_size:   训练集总样本数，用于 loss 归一化
        beta:           KL 项权重（beta-VAE；1 为标准 VAE）
        recon_type:     "bce" 或 "mse"
        free_bits:      每维 KL 下限（nats），0 表示不启用
        perceptual_fn:  可选的 VGGPerceptualLoss 模块
        lambda_pixel:   像素重建项系数
        lambda_perc:    感知损失系数

    Returns:
        total_loss, recon_loss, kl_loss, perc_loss
        其中 perc_loss 已除以 dataset_size；未使用感知损失时 perc_loss 为 0
    """
    recon_loss, perc_loss = reconstruction_loss(
        recon_x, x, dataset_size,
        recon_type=recon_type,
        perceptual_fn=perceptual_fn,
        lambda_pixel=lambda_pixel,
        lambda_perc=lambda_perc,
    )

    # ---------- KL 散度：先逐维，可选 free bits，再对 batch 与维求和并归一化 ----------
    kl_dims = kl_divergence_per_dim(mu, log_var)
    if free_bits > 0:
        # 每维 KL 至少为 free_bits，避免 posterior collapse
        kl_dims = torch.clamp(kl_dims, min=free_bits)
    kl_loss = torch.sum(kl_dims) / dataset_size

    # 总损失 = 重建 + beta * KL（beta 可由 warmup 在 train.py 中调度）
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss, perc_loss


def hierarchical_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    stats: dict[str, torch.Tensor],
    dataset_size: int,
    beta_top: float = 1.0,
    beta_bottom: float = 1.0,
    recon_type: str = "mse",
    free_bits_top: float = 0.0,
    free_bits_bottom: float = 0.0,
    perceptual_fn: nn.Module | None = None,
    lambda_pixel: float = 1.0,
    lambda_perc: float = 0.0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    分层 VAE 损失。

    top KL:
      KL[q(z_top|x) || N(0,I)]

    bottom KL:
      KL[q(z_bottom|x,z_top) || p(z_bottom|z_top)]

    Returns:
        total_loss, recon_loss, kl_total, perc_loss, kl_top, kl_bottom
    """
    recon_loss, perc_loss = reconstruction_loss(
        recon_x, x, dataset_size,
        recon_type=recon_type,
        perceptual_fn=perceptual_fn,
        lambda_pixel=lambda_pixel,
        lambda_perc=lambda_perc,
    )

    kl_top_dims = kl_divergence_per_dim(stats["top_mu"], stats["top_log_var"])
    if free_bits_top > 0:
        kl_top_dims = torch.clamp(kl_top_dims, min=free_bits_top)
    kl_top = torch.sum(kl_top_dims) / dataset_size

    kl_bottom_dims = kl_divergence_diag_gaussians(
        stats["bottom_mu"],
        stats["bottom_log_var"],
        stats["bottom_prior_mu"],
        stats["bottom_prior_log_var"],
    )
    if free_bits_bottom > 0:
        kl_bottom_dims = torch.clamp(kl_bottom_dims, min=free_bits_bottom)
    kl_bottom = torch.sum(kl_bottom_dims) / dataset_size

    kl_total = kl_top + kl_bottom
    total_loss = recon_loss + beta_top * kl_top + beta_bottom * kl_bottom
    return total_loss, recon_loss, kl_total, perc_loss, kl_top, kl_bottom

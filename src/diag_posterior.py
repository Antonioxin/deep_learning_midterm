"""
一次性诊断：量化聚合后验(aggregate posterior)与先验 N(0,I) 的匹配程度。

回答的问题：v4 的「重建好、采样糊」到底是 VAE 上限，还是 prior hole / 后验坍塌？
输出：每维 KL、active units、聚合后验各维 std、编码器平均 sigma、
      以及把先验采样的 z 标准差缩到匹配聚合后验后再解码的 cross-sample std。
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import get_dataloader, get_device
from train import build_model

EXP = Path("experiments/exp_20260527_092330_cifar10_conv_v4")


def main() -> None:
    device = get_device()
    config = yaml.safe_load((EXP / "config.yaml").open())
    model, arch, _ = build_model(config, device)
    ckpt = torch.load(EXP / "checkpoints/final.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader, n = get_dataloader(
        "cifar10", batch_size=256, train=True,
        data_dir="src/data/", pixel_range=config["data"]["pixel_range"],
    )

    mus, logvars = [], []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            mu, log_var = model.encoder(x.to(device))
            mus.append(mu.cpu())
            logvars.append(log_var.cpu())
            if i >= 39:  # ~10k 张足够
                break
    mu = torch.cat(mus)          # (N, D)
    log_var = torch.cat(logvars)
    D = mu.shape[1]

    # 逐维 KL（对数据取平均），单位 nats
    kl_per_dim = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean(0)  # (D,)
    active = (kl_per_dim > 0.01).sum().item()

    # 聚合后验：把所有样本的 mu 看成点云，逐维 std（理想 VAE 应 ≈ 1）
    agg_std = mu.std(0)                       # (D,) 聚合后验各维标准差
    mean_sigma = torch.exp(0.5 * log_var).mean(0)  # (D,) 编码器平均后验 std

    print(f"样本数={mu.shape[0]}  latent_dim={D}")
    print(f"总 KL (sum over dims) = {kl_per_dim.sum():.2f} nats/图")
    print(f"active units (KL>0.01) = {active} / {D}")
    print(f"KL/dim: max={kl_per_dim.max():.3f}  mean={kl_per_dim.mean():.3f}  median={kl_per_dim.median():.3f}")
    print(f"聚合后验各维 std: mean={agg_std.mean():.3f}  min={agg_std.min():.3f}  max={agg_std.max():.3f}  (先验=1.0)")
    print(f"编码器平均后验 sigma: mean={mean_sigma.mean():.3f}  (越接近1越坍塌)")
    print(f"mu 全局范数均值 ||mu||={mu.norm(dim=1).mean():.2f}  (先验下 E||z||≈{np.sqrt(D):.2f})")

    # prior hole 量化：解码 z~N(0,I) vs z~N(0, agg_std) 的 cross-sample 多样性
    with torch.no_grad():
        z_prior = torch.randn(64, D, device=device)
        std_prior = model.decoder(z_prior).std(0).mean().item()
        # 用聚合后验的逐维 std 缩放（temperature / aggregate-posterior 采样）
        z_agg = torch.randn(64, D, device=device) * agg_std.to(device)
        std_agg = model.decoder(z_agg).std(0).mean().item()
        # 直接用真实编码点解码作为上界参考
        idx = torch.randperm(mu.shape[0])[:64]
        std_real = model.decoder(mu[idx].to(device)).std(0).mean().item()

    print("--- 解码图像 cross-sample std（越高=样本越多样，糊成均值图则低）---")
    print(f"  z~N(0,I)           : {std_prior:.4f}")
    print(f"  z~N(0, agg_std)    : {std_agg:.4f}   <- 匹配聚合后验后")
    print(f"  z=真实编码 mu      : {std_real:.4f}   <- 参考上界")


if __name__ == "__main__":
    main()

"""
FID 评测：对比一阶段 N(0,I) 基线、2-Stage VAE、以及一阶段重建 相对 CIFAR-10 测试集的 FID。

FID（Fréchet Inception Distance）：在 InceptionV3 pool3 (2048 维) 特征空间，
用真实图与生成图两组特征的高斯 (μ, Σ) 计算 Fréchet 距离，越低越好。
使用 pytorch-fid 的官方 InceptionV3 权重与实现。

用法：
    python src/eval_fid.py \
        --stage1 experiments/exp_20260527_092330_cifar10_conv_v4 \
        --stage2 experiments/exp_20260601_143312_cifar10_2stage_v5/checkpoints/final.pt \
        --n 10000
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from datasets import get_dataloader, get_device
from train import build_model
from models.stage2_vae import Stage2VAE


# ---------------------------------------------------------------------------
# 特征提取与统计量
# ---------------------------------------------------------------------------

@torch.no_grad()
def activations_from_batches(batch_iter, n_total, inception, device):
    """对一系列 [0,1] 值域的图像 batch 提取 InceptionV3 2048 维特征。"""
    feats = []
    pbar = tqdm(total=n_total, desc="Inception feats", leave=False)
    for imgs in batch_iter:
        imgs = imgs.to(device)
        # pytorch-fid 的 InceptionV3：normalize_input 期望 [0,1]，内部 resize 到 299
        out = inception(imgs)[0]            # (B, 2048, 1, 1)
        feats.append(out.squeeze(-1).squeeze(-1).cpu().numpy())
        pbar.update(imgs.shape[0])
    pbar.close()
    feats = np.concatenate(feats, axis=0)
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def to01(x, pixel_range):
    """模型输出 → [0,1]。"""
    return ((x + 1) * 0.5 if pixel_range == "11" else x).clamp(0, 1)


# ---------------------------------------------------------------------------
# 各来源的图像生成器（产出 [0,1] 值域 batch）
# ---------------------------------------------------------------------------

def real_batches(pixel_range_unused, batch=200):
    """CIFAR-10 测试集真实图（[0,1]）。"""
    loader, n = get_dataloader("cifar10", batch_size=batch, train=False,
                               data_dir="data/", pixel_range="01")
    def gen():
        for x, _ in loader:
            yield x
    return gen(), n


def baseline_batches(stage1, d1, n, pixel_range, device, batch=200):
    """一阶段 z~N(0,I) 采样。"""
    def gen():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            z = torch.randn(b, d1, device=device)
            yield to01(stage1.decoder(z), pixel_range).cpu()
    return gen(), n


def twostage_batches(stage1, stage2, z_mean, z_std, d2, n, pixel_range, device, batch=200):
    """2-Stage：u~N(0,I) → Stage2 → 反标准化 → Stage1。"""
    def gen():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            u = torch.randn(b, d2, device=device)
            z = stage2.decode(u) * z_std.to(device) + z_mean.to(device)
            yield to01(stage1.decoder(z), pixel_range).cpu()
    return gen(), n


def recon_batches(stage1, pixel_range, device, batch=200):
    """一阶段对测试集的重建（FID 的经验下界参考）。"""
    loader, n = get_dataloader("cifar10", batch_size=batch, train=False,
                               data_dir="data/", pixel_range=pixel_range)
    def gen():
        for x, _ in loader:
            mu, log_var = stage1.encoder(x.to(device))
            recon = stage1.decoder(stage1.reparameterize(mu, log_var))
            yield to01(recon, pixel_range).cpu()
    return gen(), n


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FID evaluation")
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--stage2", default=None, help="Stage-2 final.pt（含 z_mean/z_std）")
    parser.add_argument("--n", type=int, default=10000, help="每个来源生成的样本数")
    args = parser.parse_args()

    device = get_device()

    # --- 一阶段 ---
    s1_cfg = yaml.safe_load((Path(args.stage1) / "config.yaml").open())
    stage1, _, _ = build_model(s1_cfg, device)
    stage1.load_state_dict(torch.load(
        Path(args.stage1) / "checkpoints" / "final.pt",
        map_location=device, weights_only=True)["model_state"])
    stage1.eval()
    pixel_range = s1_cfg["data"]["pixel_range"]
    d1 = stage1.latent_dim

    # --- Inception 特征提取器 (2048 维) ---
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    # --- 真实测试集统计量 ---
    rb, nr = real_batches(None)
    print(f"[real] CIFAR-10 test: {nr} imgs")
    mu_r, sig_r = activations_from_batches(rb, nr, inception, device)

    results = {}

    # baseline
    bb, nb = baseline_batches(stage1, d1, args.n, pixel_range, device)
    mu, sig = activations_from_batches(bb, nb, inception, device)
    results["Stage-1  z~N(0,I)"] = calculate_frechet_distance(mu, sig, mu_r, sig_r)

    # recon 下界
    rcb, nrc = recon_batches(stage1, pixel_range, device)
    mu, sig = activations_from_batches(rcb, nrc, inception, device)
    results["Stage-1  reconstruction"] = calculate_frechet_distance(mu, sig, mu_r, sig_r)

    # 2-stage
    if args.stage2:
        ck = torch.load(args.stage2, map_location=device, weights_only=False)
        stage2 = Stage2VAE(dim=d1).to(device)
        # 用 checkpoint 里的实际超参重建结构（若与默认不同）
        stage2.load_state_dict(ck["model_state"])
        stage2.eval()
        z_mean, z_std = ck["z_mean"], ck["z_std"]
        d2 = stage2.latent_dim
        tb, nt = twostage_batches(stage1, stage2, z_mean, z_std, d2, args.n,
                                  pixel_range, device)
        mu, sig = activations_from_batches(tb, nt, inception, device)
        results["2-Stage VAE"] = calculate_frechet_distance(mu, sig, mu_r, sig_r)

    # --- 输出 ---
    print("\n========== FID vs CIFAR-10 test (越低越好) ==========")
    for k, v in results.items():
        print(f"  {k:28s} FID = {v:8.2f}")
    print("====================================================")


if __name__ == "__main__":
    main()

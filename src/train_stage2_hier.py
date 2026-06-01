"""
对分层 VAE 套用 2-Stage 思路（learned prior），修复诊断出的两个 prior hole：
  (1) 顶层 q(z_top) ≠ N(0,I)；
  (2) 底层条件先验 p(z_bottom|z_top) ≠ 底层聚合后验。

做法：把冻结分层模型的**联合 latent (z_top, z_bottom) 拼成向量**，用一个 Stage2VAE
学习其聚合后验分布。采样时 u~N(0,I) → Stage2 解码出 (z_top, z_bottom) → 分层解码出图，
两层都不再用原 N(0,I)/条件先验，从而同时补上两个洞。与 exp_*_2stage_v5 完全同构，
只是 latent 维度变成 d_top + d_bottom（默认 64 + 64*4*4 = 1088）。

用法：
    python src/train_stage2_hier.py \
        --hier experiments/exp_20260601_170405_cifar10_hier_v3 \
        --config configs/cifar10_2stage_hier_v6.yaml --tag cifar10_2stage_hier_v6
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
from train import build_model, make_exp_dir, set_seed, setup_logging, to_display_rgb
from eval_fid import activations_from_batches, to01
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance


# ---------------------------------------------------------------------------
# 分层 latent 的编码 / 解码（联合向量 z = [z_top | flat(z_bottom)]）
# ---------------------------------------------------------------------------

def load_hier(hier_dir: Path, device):
    cfg = yaml.safe_load((hier_dir / "config.yaml").open())
    model, arch, _ = build_model(cfg, device)
    assert arch == "hierarchical", "本脚本仅针对分层 VAE"
    ckpt = torch.load(hier_dir / "checkpoints" / "final.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def encode_joint_latent(model, x, mode="joint"):
    """
    x → latent 向量，z 取后验采样。
      mode="joint": 拼接 [z_top | flat(z_bottom)]（同时建模两层）
      mode="top"  : 只取 z_top（底层留给条件先验，因其后验 σ≈先验、信息少且高噪）
    """
    _, _, f4, f2 = model.encode_features(x)
    flat = f2.flatten(1)
    top_mu = model.top_mu(flat)
    top_log_var = model.top_log_var(flat).clamp(model.log_var_min, model.log_var_max)
    z_top = model.reparameterize(top_mu, top_log_var)
    if mode == "top":
        return z_top
    top4 = model.top_to_feature(z_top)
    bottom_mu, bottom_log_var = model.split_params(
        model.bottom_posterior(torch.cat([f4, top4], dim=1)))
    z_bottom = model.reparameterize(bottom_mu, bottom_log_var)  # (B, C, 4, 4)
    return torch.cat([z_top, z_bottom.flatten(1)], dim=1)


@torch.no_grad()
def decode_joint_latent(model, z, d_top, c, s, mode="joint"):
    """latent 向量 → 图像。mode="top" 时底层从条件先验 p(z_bottom|z_top) 采样。"""
    if mode == "top":
        z_top = z
        top4 = model.top_to_feature(z_top)
        bpmu, bplv = model.split_params(model.bottom_prior(top4))
        z_bottom = model.reparameterize(bpmu, bplv)
        return model.decode_from_top4(top4, z_bottom)
    z_top = z[:, :d_top]
    z_bottom = z[:, d_top:].view(-1, c, s, s)
    top4 = model.top_to_feature(z_top)
    return model.decode_from_top4(top4, z_bottom)


@torch.no_grad()
def two_stage_sample(hier, stage2, n, device, z_mean, z_std, d_top, c, s, mode="joint"):
    z_norm = stage2.sample_latent(n, device)
    z = z_norm * z_std.to(device) + z_mean.to(device)
    return decode_joint_latent(hier, z, d_top, c, s, mode)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_grid(imgs_raw, out_path, pixel_range):
    imgs = to_display_rgb(imgs_raw.cpu(), pixel_range)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img, vmin=0, vmax=1); ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=100); plt.close(fig)


@torch.no_grad()
def save_comparison(hier, stage2, device, out_path, pixel_range,
                    z_mean, z_std, d_top, c, s, mode="joint"):
    n = 32
    torch.manual_seed(0)
    base = hier.sample(n, device)  # 分层原生采样（N(0,I) + 条件先验）
    two = two_stage_sample(hier, stage2, n, device, z_mean, z_std, d_top, c, s, mode)
    base_d = to_display_rgb(base.cpu(), pixel_range)
    two_d = to_display_rgb(two.cpu(), pixel_range)
    fig, axes = plt.subplots(8, 8, figsize=(9, 9))
    for col_block, imgs in [(0, base_d), (4, two_d)]:
        for i in range(n):
            ax = axes[i // 4, i % 4 + col_block]
            ax.imshow(imgs[i], vmin=0, vmax=1); ax.axis("off")
    fig.text(0.27, 0.99, "Hier native (N(0,I)+cond prior)", ha="center", va="top", fontsize=11)
    fig.text(0.75, 0.99, "Hier + 2-Stage learned prior", ha="center", va="top", fontsize=11)
    plt.tight_layout(pad=0.2, rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_fid(hier, stage2, device, pixel_range, z_mean, z_std, d_top, c, s, n_fid, log, mode="joint"):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    # 真实测试集
    rloader, nr = get_dataloader("cifar10", 200, train=False, data_dir="data/", pixel_range="01")
    mu_r, sig_r = activations_from_batches((x for x, _ in rloader), nr, inception, device)

    def fid_of(gen, n):
        mu, sig = activations_from_batches(gen, n, inception, device)
        return calculate_frechet_distance(mu, sig, mu_r, sig_r)

    def hier_native(n, b=200):
        for i in range(0, n, b):
            yield to01(hier.sample(min(b, n - i), device), pixel_range).cpu()

    def two_stage(n, b=200):
        for i in range(0, n, b):
            yield to01(two_stage_sample(hier, stage2, min(b, n - i), device,
                                        z_mean, z_std, d_top, c, s, mode), pixel_range).cpu()

    def recon(b=200):
        loader, _ = get_dataloader("cifar10", b, train=False, data_dir="data/", pixel_range=pixel_range)
        for x, _ in loader:
            z = encode_joint_latent(hier, x.to(device), mode)
            yield to01(decode_joint_latent(hier, z, d_top, c, s, mode), pixel_range).cpu()

    res = {
        "Hier native (N(0,I)+cond prior)": fid_of(hier_native(n_fid), n_fid),
        "Hier + 2-Stage learned prior": fid_of(two_stage(n_fid), n_fid),
        "Hier reconstruction (下界)": fid_of(recon(), 10000),
    }
    log.info("===== FID vs CIFAR-10 test (越低越好) =====")
    for k, v in res.items():
        log.info(f"  {k:36s} FID = {v:8.2f}")
    return res


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hier", required=True, help="分层实验目录")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", default="2stage_hier")
    parser.add_argument("--n_fid", type=int, default=10000)
    parser.add_argument("--latent", choices=["joint", "top"], default="top",
                        help="learned prior 建模哪部分 latent（top=只建模 z_top，底层用条件先验）")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = get_device()
    set_seed(cfg["training"]["seed"], device)

    hier, hcfg = load_hier(Path(args.hier), device)
    pixel_range = hcfg["data"]["pixel_range"]
    mode = args.latent
    d_top = hier.top_latent_dim
    c, s = hier.bottom_latent_channels, hier.bottom_spatial
    d_joint = d_top if mode == "top" else d_top + c * s * s

    # --- 编码训练集（一次性缓存）---
    loader, N = get_dataloader("cifar10", 512, train=True, data_dir="data/", pixel_range=pixel_range)
    zs = []
    for x, _ in tqdm(loader, desc=f"Encoding {mode} latents", leave=False):
        zs.append(encode_joint_latent(hier, x.to(device), mode).cpu())
    Z = torch.cat(zs)  # (N, d_joint)
    z_mean, z_std = Z.mean(0), Z.std(0).clamp_min(1e-6)

    # --- Stage-2 ---
    stage2 = Stage2VAE(
        dim=d_joint,
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"].get("latent_dim", d_joint),
        depth=cfg["model"].get("depth", 3),
    ).to(device)

    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    warmup_epochs = cfg["training"].get("beta_warmup_epochs", 0)
    optimizer = torch.optim.Adam(stage2.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg["training"].get("lr_eta_min", 1e-5))

    exp_dir = make_exp_dir("experiments", args.tag)
    log = setup_logging(exp_dir / "logs" / "train.log")
    log.info(f"Experiment: {exp_dir.name}")
    log.info(f"Hier: {Path(args.hier).name}  mode={mode}  d_top={d_top}  d_bottom={c*s*s}  d_model={d_joint}  N={N}")
    log.info(f"Stage-2 params: {sum(p.numel() for p in stage2.parameters()):,}")
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump({**cfg, "hier_dir": str(args.hier)}, f, default_flow_style=False)

    metrics_path = exp_dir / "logs" / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "recon_nll", "kl", "beta", "gamma", "lr", "std_2stage"])

    Z = Z.to(device); z_mean_d, z_std_d = z_mean.to(device), z_std.to(device)
    Zn = (Z - z_mean_d) / z_std_d  # 标准化后的训练目标

    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
        stage2.train()
        perm = torch.randperm(N, device=device)
        ls = rs = ks = 0.0; nb = 0
        for i in range(0, N, batch_size):
            zb = Zn[perm[i:i + batch_size]]
            optimizer.zero_grad()
            loss, recon, kl = stage2.loss(zb, beta=beta)
            loss.backward(); optimizer.step()
            ls += loss.item(); rs += recon.item(); ks += kl.item(); nb += 1
        scheduler.step()

        stage2.eval()
        std2 = two_stage_sample(hier, stage2, 256, device, z_mean, z_std, d_top, c, s, mode).std(0).mean().item()
        gamma = stage2.log_gamma2.exp().sqrt().item()
        lr_now = optimizer.param_groups[0]["lr"]
        log.info(f"Epoch {epoch:>3}/{epochs}  loss={ls/nb:.4f}  recon_nll={rs/nb:.4f}  "
                 f"kl={ks/nb:.4f}  beta={beta:.3f}  gamma={gamma:.4f}  lr={lr_now:.2e}  std_2stage={std2:.4f}")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, ls/nb, rs/nb, ks/nb, beta, gamma, lr_now, std2])

        if epoch % 20 == 0 or epoch == epochs:
            save_grid(two_stage_sample(hier, stage2, 64, device, z_mean, z_std, d_top, c, s, mode),
                      exp_dir / "samples" / f"sample_epoch_{epoch:03d}.png", pixel_range)

    torch.save({"epoch": epochs, "model_state": stage2.state_dict(),
                "z_mean": z_mean, "z_std": z_std, "hier_dir": str(args.hier)},
               exp_dir / "checkpoints" / "final.pt")
    save_comparison(hier, stage2, device, exp_dir / "compare_native_vs_2stage.png",
                    pixel_range, z_mean, z_std, d_top, c, s, mode)

    run_fid(hier, stage2, device, pixel_range, z_mean, z_std, d_top, c, s, args.n_fid, log, mode)
    log.info(f"Done. Results in {exp_dir}")
    print(f"\nEXP_DIR={exp_dir}")


if __name__ == "__main__":
    main()

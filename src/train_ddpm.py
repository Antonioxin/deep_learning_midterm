"""
DDPM 训练入口（Ho et al. 2020）。

复用项目现有设施：datasets（CIFAR, pixel_range="11"→[-1,1]）、实验目录/日志、eval_fid。
关键：用 EMA 权重采样（显著提升样本质量），DDIM 少步采样做快速 FID。

用法：
    python src/train_ddpm.py --config configs/cifar10_ddpm.yaml --tag cifar10_ddpm
"""

import argparse
import copy
import csv
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import get_dataloader, get_device
from train import make_exp_dir, set_seed, setup_logging, to_display_rgb
from models.unet import UNet
from diffusion import GaussianDiffusion


class EMA:
    """参数指数滑动平均；采样时用 EMA 影子权重。"""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


@torch.no_grad()
def save_sample_grid(diffusion, model, out_path, pixel_range, n=64, ddim_steps=100):
    model.eval()
    imgs = diffusion.ddim_sample(model, (n, 3, 32, 32), ddim_steps=ddim_steps)
    disp = to_display_rgb(imgs.cpu(), pixel_range)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flat, disp):
        ax.imshow(img, vmin=0, vmax=1); ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=100); plt.close(fig)


@torch.no_grad()
def run_fid(diffusion, model, device, pixel_range, n, ddim_steps, log):
    from eval_fid import activations_from_batches, to01
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance

    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    rloader, nr = get_dataloader("cifar10", 200, train=False, data_dir="data/", pixel_range="01")
    mu_r, sig_r = activations_from_batches((x for x, _ in rloader), nr, inception, device)

    def gen(b=250):
        for i in range(0, n, b):
            bs = min(b, n - i)
            imgs = diffusion.ddim_sample(model, (bs, 3, 32, 32), ddim_steps=ddim_steps)
            yield to01(imgs, pixel_range).cpu()

    mu, sig = activations_from_batches(gen(), n, inception, device)
    fid = calculate_frechet_distance(mu, sig, mu_r, sig_r)
    log.info(f"FID (DDIM {ddim_steps} steps, n={n}) = {fid:.2f}")
    return fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", default="ddpm")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = get_device()
    set_seed(cfg["training"]["seed"], device)
    pixel_range = cfg["data"].get("pixel_range", "11")

    loader, n_data = get_dataloader(cfg["data"]["dataset"], cfg["data"]["batch_size"],
                                    train=True, data_dir="data/", pixel_range=pixel_range)

    m = cfg["model"]
    model = UNet(
        base_channels=m.get("base_channels", 128),
        channel_mults=tuple(m.get("channel_mults", [1, 2, 2, 2])),
        num_res_blocks=m.get("num_res_blocks", 2),
        attn_resolutions=tuple(m.get("attn_resolutions", [16])),
        dropout=m.get("dropout", 0.1),
    ).to(device)
    ema = EMA(model, decay=cfg["training"].get("ema_decay", 0.9999))

    t = cfg["training"]
    diffusion = GaussianDiffusion(timesteps=t.get("timesteps", 1000),
                                  schedule=t.get("schedule", "linear"), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    epochs = t["epochs"]
    grad_clip = t.get("grad_clip", 1.0)
    ddim_steps = t.get("ddim_steps", 100)
    use_amp = t.get("amp", True) and device.type == "cuda"

    exp_dir = make_exp_dir("experiments", args.tag)
    log = setup_logging(exp_dir / "logs" / "train.log")
    log.info(f"Experiment: {exp_dir.name}")
    log.info(f"Dataset size: {n_data}  |  UNet params: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"timesteps={diffusion.timesteps} schedule={t.get('schedule','linear')} "
             f"batch={cfg['data']['batch_size']} epochs={epochs}")
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    metrics_path = exp_dir / "logs" / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "lr"])

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        nb = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch:>3}/{epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = diffusion.p_losses(model, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)
            loss_sum += loss.item(); nb += 1; global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = loss_sum / nb
        lr_now = optimizer.param_groups[0]["lr"]
        log.info(f"Epoch {epoch:>3}/{epochs}  loss={avg:.4f}  lr={lr_now:.2e}  step={global_step}")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg, lr_now])

        if epoch % cfg["training"].get("sample_every", 10) == 0 or epoch == epochs:
            save_sample_grid(diffusion, ema.shadow,
                             exp_dir / "samples" / f"sample_epoch_{epoch:03d}.png",
                             pixel_range, ddim_steps=ddim_steps)
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "ema_state": ema.shadow.state_dict()},
                       exp_dir / "checkpoints" / "final.pt")

    # 最终 FID（用 EMA + DDIM）
    n_fid = cfg["training"].get("n_fid", 5000)
    fid = run_fid(diffusion, ema.shadow, device, pixel_range, n_fid, ddim_steps, log)
    log.info(f"Done. Results in {exp_dir}  |  final FID={fid:.2f}")
    print(f"\nEXP_DIR={exp_dir}  FID={fid:.2f}")


if __name__ == "__main__":
    main()

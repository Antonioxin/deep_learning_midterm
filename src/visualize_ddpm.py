"""
DDPM 去噪过程可视化（模仿 Ho et al. 2020 Figure 6 风格）。

展示 x̂₀ 随去噪步骤的演变：从纯高斯噪声（最左列）逐步还原为清晰图像（最右列）。
每行对应一张独立的随机生成图像，列为均匀采样的时间步快照。

用法示例：
    python src/visualize_ddpm.py \
        --exp experiments/exp_20260601_220750_cifar10_ddpm \
        --out experiments/exp_20260601_220750_cifar10_ddpm/samples/progressive.png
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent))
from datasets import get_device
from train import to_display_rgb
from models.unet import UNet
from diffusion import GaussianDiffusion


def load_ddpm(exp_dir: Path, device):
    cfg = yaml.safe_load((exp_dir / "config.yaml").open())
    m = cfg["model"]
    model = UNet(
        base_channels=m.get("base_channels", 128),
        channel_mults=tuple(m.get("channel_mults", [1, 2, 2, 2])),
        num_res_blocks=m.get("num_res_blocks", 2),
        attn_resolutions=tuple(m.get("attn_resolutions", [16])),
        dropout=m.get("dropout", 0.1),
    ).to(device)
    ckpt = torch.load(exp_dir / "checkpoints" / "final.pt",
                      map_location=device, weights_only=True)
    # 优先使用 EMA 权重
    key = "ema_state" if "ema_state" in ckpt else "model_state"
    model.load_state_dict(ckpt[key])
    model.eval()
    t = cfg["training"]
    diffusion = GaussianDiffusion(
        timesteps=t.get("timesteps", 1000),
        schedule=t.get("schedule", "linear"),
        device=device,
    )
    pixel_range = cfg["data"].get("pixel_range", "11")
    return model, diffusion, pixel_range


@torch.no_grad()
def generate_row(model, diffusion, pixel_range, seed: int, row_idx: int,
                 n_batch: int, n_cols: int, ddim_steps: int):
    """生成一批图像并取第 row_idx 行，返回 (n_frames, 3, 32, 32)。"""
    torch.manual_seed(seed)
    frames = diffusion.ddim_sample_progressive(
        model,
        shape=(n_batch, 3, 32, 32),
        ddim_steps=ddim_steps,
        eta=0.0,
        n_frames=n_cols - 1,
    )  # (n_frames, n_batch, 3, 32, 32)
    return frames[:, row_idx]  # (n_frames, 3, 32, 32)


@torch.no_grad()
def make_progressive_figure(
    exp_dir: Path,
    out_path: Path,
    n_cols: int = 16,
    ddim_steps: int = 100,
    seed: int = 0,
    row_specs: list | None = None,
):
    """
    row_specs: list of (seed, row_idx, n_batch) tuples，每个元素对应一行。
               若为 None，则使用 seed 统一生成 4 行。
    """
    device = get_device()
    model, diffusion, pixel_range = load_ddpm(exp_dir, device)

    if row_specs is None:
        torch.manual_seed(seed)
        frames_all = diffusion.ddim_sample_progressive(
            model, shape=(4, 3, 32, 32),
            ddim_steps=ddim_steps, eta=0.0, n_frames=n_cols - 1,
        )
        row_frames = [frames_all[:, r] for r in range(4)]
    else:
        row_frames = []
        for (s, ridx, nbatch) in row_specs:
            row_frames.append(
                generate_row(model, diffusion, pixel_range, s, ridx, nbatch, n_cols, ddim_steps)
            )

    n_rows = len(row_frames)
    n_frames_actual = row_frames[0].shape[0]

    # --- 绘图 ---
    pad = 0.03
    cell = 0.85
    fig_w = n_frames_actual * cell + (n_frames_actual - 1) * pad
    fig_h = n_rows * cell + (n_rows - 1) * pad

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs = gridspec.GridSpec(
        n_rows, n_frames_actual,
        figure=fig,
        left=0, right=1, top=1, bottom=0,
        wspace=pad / cell,
        hspace=pad / cell,
    )

    for r, frames in enumerate(row_frames):
        for c in range(n_frames_actual):
            ax = fig.add_subplot(gs[r, c])
            img = to_display_rgb(frames[c : c + 1].cpu(), pixel_range)[0]
            ax.imshow(img, vmin=0, vmax=1, interpolation="bilinear")
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved progressive visualization → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DDPM denoising progression (Figure 6 style)"
    )
    parser.add_argument("--exp", required=True, help="Path to experiment directory")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--rows", type=int, default=4, help="Number of image rows")
    parser.add_argument("--cols", type=int, default=16, help="Number of time-step columns")
    parser.add_argument("--ddim-steps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    exp_dir = Path(args.exp)
    out_path = Path(args.out) if args.out else exp_dir / "samples" / "progressive_denoising.png"

    make_progressive_figure(
        exp_dir=exp_dir,
        out_path=out_path,
        n_rows=args.rows,
        n_cols=args.cols,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

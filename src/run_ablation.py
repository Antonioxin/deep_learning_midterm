"""
单层 ConvVAE v4 架构上的受控消融：固定其余、每次只变一个因子，统一训练预算，用 FID 对比。

测试因子（基准：β=4, 感知损失 λ_perc=0.01, free_bits=0, latent_dim=64）：
  - β ∈ {1, 4, 8}
  - 感知损失：on / off
  - free_bits：0 / 0.5
  - latent_dim ∈ {32, 64, 128}

所有 run 用相同的缩短预算（默认 40 epochs），故 FID 绝对值与 100-epoch 的正式 v4(=155) 不同，
**只做组内相对比较**。结果增量写入 results/ablation.md，单个 run 失败不影响其余。

用法：
    python src/run_ablation.py --epochs 40 --n_fid 10000
"""

import argparse
import copy
import traceback
from pathlib import Path

import numpy as np
import torch

from datasets import get_dataloader, get_device
from train import build_model, train
from eval_fid import activations_from_batches, to01
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance


def base_config(epochs: int) -> dict:
    """基准配置：等同 v4，但训练预算缩短到 epochs。"""
    return {
        "data": {"dataset": "cifar10", "batch_size": 128, "pixel_range": "11"},
        "model": {"arch": "conv", "version": "v4", "latent_dim": 64},
        "training": {
            "beta": 4.0, "beta_warmup_epochs": 5, "epochs": epochs,
            "free_bits": 0.0, "lambda_perc": 0.01, "lambda_pixel": 1.0,
            "lr": 1e-3, "lr_eta_min": 1e-5, "lr_scheduler": "cosine",
            "optimizer": "adam", "recon_loss": "mse", "seed": 42,
        },
    }


def ablation_configs(epochs: int):
    """返回 [(name, 描述, config)]，每个只改一个因子。"""
    runs = []

    def mk(name, desc, **over):
        cfg = copy.deepcopy(base_config(epochs))
        for k, v in over.get("model", {}).items():
            cfg["model"][k] = v
        for k, v in over.get("training", {}).items():
            cfg["training"][k] = v
        runs.append((name, desc, cfg))

    # β 扫描（base 为 β=4）
    mk("beta1",   "β=1",                 training={"beta": 1.0})
    mk("beta4",   "β=4 (基准)",           )
    mk("beta8",   "β=8",                 training={"beta": 8.0})
    # 感知损失
    mk("noperc",  "无感知损失 λ_perc=0",  training={"lambda_perc": 0.0})
    # free bits
    mk("fb05",    "free_bits=0.5",       training={"free_bits": 0.5})
    # latent 维度
    mk("lat32",   "latent_dim=32",       model={"latent_dim": 32})
    mk("lat128",  "latent_dim=128",      model={"latent_dim": 128})
    return runs


@torch.no_grad()
def generation_fid(model, device, pixel_range, inception, mu_r, sig_r, n, b=200):
    def gen():
        for i in range(0, n, b):
            z = torch.randn(min(b, n - i), model.latent_dim, device=device)
            yield to01(model.decoder(z), pixel_range).cpu()
    mu, sig = activations_from_batches(gen(), n, inception, device)
    return calculate_frechet_distance(mu, sig, mu_r, sig_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n_fid", type=int, default=10000)
    args = parser.parse_args()

    device = get_device()

    # Inception + 真实测试集统计量（只算一次）
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    rloader, nr = get_dataloader("cifar10", 200, train=False, data_dir="data/", pixel_range="01")
    mu_r, sig_r = activations_from_batches((x for x, _ in rloader), nr, inception, device)

    out_md = Path("results/ablation.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# 消融实验（单层 ConvVAE v4 架构，{args.epochs} epochs，生成 FID vs CIFAR-10 test）\n\n"
              f"> 基准：β=4, λ_perc=0.01, free_bits=0, latent_dim=64。每行只改一个因子。\n"
              f"> FID 为 {args.epochs}-epoch 缩短训练所得，仅做组内相对比较（与 100-epoch 正式 v4=155 不可直接比）。\n\n"
              f"| 实验 | 改动 | β | λ_perc | free_bits | latent | 生成 FID |\n"
              f"|---|---|---:|---:|---:|---:|---:|\n")
    out_md.write_text(header)
    print(header)

    for name, desc, cfg in ablation_configs(args.epochs):
        t = cfg["training"]; m = cfg["model"]
        try:
            exp_dir = train(cfg, tag=f"ablation_{name}")
            model, _, _ = build_model(cfg, device)
            model.load_state_dict(torch.load(exp_dir / "checkpoints" / "final.pt",
                                             map_location=device, weights_only=True)["model_state"])
            model.eval()
            fid = generation_fid(model, device, cfg["data"]["pixel_range"],
                                 inception, mu_r, sig_r, args.n_fid)
            fid_str = f"{fid:.2f}"
        except Exception:
            traceback.print_exc()
            fid_str = "FAILED"
        row = (f"| {name} | {desc} | {t['beta']} | {t['lambda_perc']} | "
               f"{t['free_bits']} | {m['latent_dim']} | {fid_str} |\n")
        with open(out_md, "a") as f:
            f.write(row)
        print(row.strip())

    print(f"\nDone. Table written to {out_md}")


if __name__ == "__main__":
    main()

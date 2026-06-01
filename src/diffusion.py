"""
高斯扩散过程（Ho et al. 2020 DDPM；cosine 调度来自 Nichol & Dhariwal 2021）。

约定：
  - 预测噪声 ε（ε-parameterization），训练目标为 MSE(ε, ε_θ(x_t, t))。
  - 图像值域 [-1, 1]（配合数据 pixel_range="11"）。
  - 提供祖先采样（p_sample，完整 T 步）与 DDIM 确定性采样（少步，用于快速 FID）。
"""

import torch
import torch.nn.functional as F


def make_beta_schedule(timesteps: int, schedule: str = "linear") -> torch.Tensor:
    if schedule == "linear":
        # DDPM 原文：beta 从 1e-4 线性增到 0.02
        return torch.linspace(1e-4, 0.02, timesteps)
    if schedule == "cosine":
        # Nichol & Dhariwal 2021，s=0.008
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        acp = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown schedule {schedule!r}")


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """取 a[t] 并 reshape 成可对 (B,C,H,W) 广播的 (B,1,1,1)。"""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, schedule: str = "linear",
                 device: torch.device = torch.device("cpu")) -> None:
        self.timesteps = timesteps
        self.device = device
        betas = make_beta_schedule(timesteps, schedule).to(device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_cumprod = acp
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # 后验 q(x_{t-1}|x_t,x_0) 方差
        self.posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.acp_prev = acp_prev

    # ---------- 前向加噪 ----------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (_extract(self.sqrt_acp, t, x0.shape) * x0
                + _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    # ---------- 训练损失 ----------
    def p_losses(self, model, x0: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)

    # ---------- 祖先采样（完整 T 步）----------
    @torch.no_grad()
    def p_sample(self, model, shape, progress=False) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        iterator = reversed(range(self.timesteps))
        for i in iterator:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            pred = model(x, t)
            beta = _extract(self.betas, t, x.shape)
            sqrt_one_minus = _extract(self.sqrt_one_minus_acp, t, x.shape)
            sqrt_recip = _extract(self.sqrt_recip_alphas, t, x.shape)
            mean = sqrt_recip * (x - beta / sqrt_one_minus * pred)
            if i > 0:
                var = _extract(self.posterior_var, t, x.shape)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)

    # ---------- DDIM 确定性采样（少步，用于快速生成/FID）----------
    @torch.no_grad()
    def ddim_sample(self, model, shape, ddim_steps: int = 100, eta: float = 0.0) -> torch.Tensor:
        step_seq = torch.linspace(0, self.timesteps - 1, ddim_steps, dtype=torch.long).tolist()
        step_seq = list(reversed(step_seq))
        x = torch.randn(shape, device=self.device)
        for idx, i in enumerate(step_seq):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            pred = model(x, t)
            acp_t = _extract(self.alphas_cumprod, t, x.shape)
            # 由 ε 反推 x0
            x0 = (x - torch.sqrt(1 - acp_t) * pred) / torch.sqrt(acp_t)
            x0 = x0.clamp(-1, 1)
            if idx == len(step_seq) - 1:
                x = x0
                break
            i_prev = step_seq[idx + 1]
            t_prev = torch.full((shape[0],), i_prev, device=self.device, dtype=torch.long)
            acp_prev = _extract(self.alphas_cumprod, t_prev, x.shape)
            sigma = eta * torch.sqrt((1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev))
            dir_xt = torch.sqrt(1 - acp_prev - sigma ** 2) * pred
            x = torch.sqrt(acp_prev) * x0 + dir_xt + sigma * torch.randn_like(x)
        return x.clamp(-1, 1)

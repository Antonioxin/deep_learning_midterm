"""
DDPM 用的 U-Net 噪声预测器 ε_θ(x_t, t)。

结构（Ho et al. 2020 "Denoising Diffusion Probabilistic Models" 的标准配置）：
  - 正弦时间步嵌入 + MLP
  - 下采样/上采样路径，每级若干带时间条件的 ResBlock（GroupNorm + SiLU）
  - 指定分辨率（默认 16×16）插入自注意力
  - U-Net skip 连接：下路特征压栈、上路 concat 出栈
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = min(num_groups, channels)
    while channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, channels)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """正弦位置编码形式的时间步嵌入。t: (B,) 整数步 → (B, dim)。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """带时间条件的残差块。"""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """空间自注意力块（单头）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        k = k.reshape(B, C, H * W)                     # (B, C, HW)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)  # (B, HW, HW)
        out = (attn @ v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """ε 预测 U-Net。输入/输出同形状 (B, in_ch, H, W)。"""

    def __init__(
        self,
        in_ch: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),
        dropout: float = 0.1,
        img_size: int = 32,
    ) -> None:
        super().__init__()
        temb_dim = base_channels * 4
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        self.conv_in = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        # ---- 下采样路径 ----
        self.downs = nn.ModuleList()
        chs = [base_channels]
        cur = base_channels
        res = img_size
        for i, m in enumerate(channel_mults):
            out = base_channels * m
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(cur, out, temb_dim, dropout)])
                cur = out
                if res in attn_resolutions:
                    block.append(AttnBlock(cur))
                self.downs.append(block)
                chs.append(cur)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(cur))
                chs.append(cur)
                res //= 2

        # ---- 中间 ----
        self.mid1 = ResBlock(cur, cur, temb_dim, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, temb_dim, dropout)

        # ---- 上采样路径 ----
        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(channel_mults))):
            out = base_channels * m
            for _ in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(cur + chs.pop(), out, temb_dim, dropout)])
                cur = out
                if res in attn_resolutions:
                    block.append(AttnBlock(cur))
                self.ups.append(block)
            if i != 0:
                self.ups.append(Upsample(cur))
                res *= 2

        self.norm_out = _gn(cur)
        self.conv_out = nn.Conv2d(cur, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base_channels))
        h = self.conv_in(x)
        hs = [h]
        for module in self.downs:
            if isinstance(module, Downsample):
                h = module(h)
            else:
                h = module[0](h, temb)
                if len(module) > 1:
                    h = module[1](h)
            hs.append(h)  # 每个下路模块输出都压栈（含 downsample），与上路 pop 次数对齐
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for module in self.ups:
            if isinstance(module, Upsample):
                h = module(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module[0](h, temb)
                if len(module) > 1:
                    h = module[1](h)
        return self.conv_out(F.silu(self.norm_out(h)))

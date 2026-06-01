"""
卷积变分自编码器（ConvVAE），面向 32×32 RGB 图像（如 CIFAR-10）。

设计要点：
  - 编码器用步长卷积下采样，解码器用转置卷积上采样
  - BatchNorm 提升彩色图像训练稳定性
  - 提供 v2 / v3 两套结构，通过 version 参数切换

版本差异：
  v2: 3 层卷积，瓶颈 256×4×4，Sigmoid 输出，输入像素 [0, 1]
  v3: 4 层卷积 + 每层后 ResBlock，瓶颈 512×2×2，Tanh 输出，输入像素 [-1, 1]
  v4: 编码器同 v3；解码器用 GroupNorm 替代 BatchNorm，缓解 prior hole
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    固定空间分辨率下的残差块：x + Conv-BN-ReLU-Conv-BN，最后 ReLU。

    在 v3 编码器中每个下采样 stage 之后使用，加深网络而不改变特征图尺寸。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接：缓解深层梯度消失
        return torch.relu(x + self.conv(x))


class ConvEncoderV2(nn.Module):
    """
    v2 编码器：(B, 3, 32, 32) → 高斯参数 (μ, log_var)。

    三次 stride=2 的 4×4 卷积：32→16→8→4，通道 3→64→128→256，
    展平后维度 256×4×4=4096，再经全连接映射到 latent_dim。
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.flat_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(start_dim=1)  # (B, flat_dim)
        return self.fc_mu(h), self.fc_log_var(h)


class ConvDecoderV2(nn.Module):
    """
    v2 解码器：z → (B, 3, 32, 32)，输出经 Sigmoid 落在 (0, 1)。

    与 v2 编码器对称：Linear 展开到 256×4×4，三次 ConvTranspose2d 上采样。
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z)).view(-1, 256, 4, 4)
        return self.deconv(h)


class ConvEncoderV3(nn.Module):
    """
    v3 编码器：更深、更小瓶颈。

    四次下采样：32→16→8→4→2，每层后接 ResBlock 提炼特征；
    瓶颈 512×2×2=2048 维，比 v2 空间更小、通道更宽，感受野更大。
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        # 四个下采样 stage，通道逐级增加
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
        ])
        # 与每个 stage 输出通道匹配的残差块
        self.resblocks = nn.ModuleList([
            ResBlock(64),
            ResBlock(128),
            ResBlock(256),
            ResBlock(512),
        ])
        self.flat_dim = 512 * 2 * 2
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for stage, res in zip(self.stages, self.resblocks):
            x = res(stage(x))  # 先下采样，再残差精炼
        h = x.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)


class ConvDecoderV3(nn.Module):
    """
    v3 解码器：z → (B, 3, 32, 32)，输出经 Tanh 落在 (-1, 1)。

    与 v3 编码器及数据预处理 pixel_range="11" 配套使用。
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # 输出 [-1, 1]，配合 MSE 与 [-1,1] 输入
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z)).view(-1, 512, 2, 2)
        return self.deconv(h)


def _group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Pick a valid GroupNorm group count that divides channels."""
    groups = min(num_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvDecoderV4(nn.Module):
    """
    v4 解码器：与 v3 对称的上采样路径，Tanh 输出 [-1, 1]。

    相对 v3 的改动（修复 prior hole）：
      - BatchNorm2d → GroupNorm，避免 eval 时 running stats 与 N(0,I) 先验采样失配
      - fc→reshape 后第一层转置卷积后不加 norm，仅 ReLU，保留 z 带来的样本间差异
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            _group_norm(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            _group_norm(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z)).view(-1, 512, 2, 2)
        return self.deconv(h)


class ConvVAE(nn.Module):
    """
    卷积 VAE 封装：根据 version 选择 v2 / v3 / v4 的编码器/解码器对。

    接口与全连接 VAE 一致：forward 返回 (recon_x, mu, log_var)，sample 从 N(0,I) 生成。
    """

    def __init__(self, latent_dim: int = 128, version: str = "v2") -> None:
        super().__init__()
        self.version = version
        self.latent_dim = latent_dim
        if version == "v4":
            self.encoder = ConvEncoderV3(latent_dim)
            self.decoder = ConvDecoderV4(latent_dim)
        elif version == "v3":
            self.encoder = ConvEncoderV3(latent_dim)
            self.decoder = ConvDecoderV3(latent_dim)
        elif version == "v2":
            self.encoder = ConvEncoderV2(latent_dim)
            self.decoder = ConvDecoderV2(latent_dim)
        else:
            raise ValueError(
                f"Unknown ConvVAE version: {version!r}. Use 'v2', 'v3', or 'v4'."
            )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """与全连接 VAE 相同的重参数化采样。"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 32, 32)，像素范围由数据预处理决定（v2: [0,1]，v3: [-1,1]）

        Returns:
            recon_x: (B, 3, 32, 32)
            mu, log_var: (B, latent_dim)
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """从标准正态先验采样并解码，用于可视化生成质量。"""
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)

"""
全连接（Fully-Connected）变分自编码器（VAE）。

网络结构遵循 Kingma & Welling (2013) 附录 C：
  - 编码器/解码器各两层隐藏层，默认 hidden_dim=400
  - 潜变量维度默认 latent_dim=20
  - 适用于展平后的 28×28 灰度图（如 MNIST，input_dim=784）
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    识别模型 q_φ(z|x)：将观测 x 映射为潜变量高斯分布的参数 (μ, log σ²)。

    输出用于构造 q_φ(z|x) = N(μ, diag(exp(log_var)))，
    再通过重参数化技巧采样 z ~ q(z|x)。
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """
        Args:
            input_dim:  输入特征维度（MNIST 为 784）
            hidden_dim: 两个隐藏层的宽度
            latent_dim: 潜空间维度
        """
        super().__init__()
        # 共享的确定性特征提取：784 -> hidden -> hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 两个独立线性头：分别预测均值与对数方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)，已归一化到 [0,1] 的展平图像

        Returns:
            mu:      (batch, latent_dim)
            log_var: (batch, latent_dim)，log(σ²)，数值稳定用 log 而非 σ
        """
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """
    生成模型 p_θ(x|z)：将潜向量 z 映射回像素空间。

    最后一层 Sigmoid 将输出约束在 (0,1)，与 Bernoulli 似然及 BCE 重建损失一致。
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Args:
            latent_dim: 潜空间维度
            hidden_dim: 隐藏层宽度
            output_dim: 输出像素数（MNIST 为 784）
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # 像素概率 ∈ (0, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, latent_dim) -> recon: (batch, output_dim)"""
        return self.net(z)


class VAE(nn.Module):
    """
    标准变分自编码器：编码器 + 重参数化采样 + 解码器。

    训练时 forward 返回重建图与高斯参数；推理可用 sample() 从先验 p(z) 生成新图。
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：z = μ + ε ⊙ σ，其中 ε ~ N(0, I)，σ = exp(0.5 * log_var)。

        随机性只来自 ε（与网络参数无关），从而可对 μ、log_var 反传梯度。
        Kingma & Welling (2013) 式 (4)。
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 与 std 同形状的标准正态噪声
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步前向：编码 -> 采样 z -> 解码。

        Args:
            x: (batch, 784) 展平图像，像素值 [0, 1]

        Returns:
            recon_x: (batch, 784) 重建像素（Sigmoid 输出）
            mu:      (batch, latent_dim)
            log_var: (batch, latent_dim)
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """
        从先验 p(z)=N(0,I) 采样 n 个 z，解码得到 n 张生成图（无梯度）。

        Args:
            n:      生成样本数量
            device: 计算设备

        Returns:
            (n, output_dim) 重建像素张量
        """
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)

"""
第二阶段 VAE（2-Stage VAE，Dai & Wipf 2019）。

动机（见本项目 diag_posterior.py 的诊断结论）：
  第一阶段 ConvVAE 重建良好、64 维全 active、聚合后验逐维 std≈1，
  但从 N(0,I) 采样仍糊——因为编码点云只是 N(0,I) 球壳上的一层薄流形，
  其「联合结构」并非各向同性高斯，N(0,I) 的随机点大多落在流形缝隙里。

解决：用第二个 VAE 学习「第一阶段 latent z 的真实分布 q(z)」。
  采样时 u~N(0,I) → Stage2 解码出 z → Stage1 解码出图像，
  让随机 z 落回流形上，从而补满 prior hole。

关键设计（Dai & Wipf 2019 §3）：
  - 第二阶段对连续、无界的 z 用高斯似然，且**可学习**观测噪声 γ
    （p(z|u)=N(decoder(u), γ²I)），γ 自适应地平衡重建精度与 KL。
  - latent 维度 d2 默认与 d1 相同；网络为浅层 MLP。
"""

import math

import torch
import torch.nn as nn


class Stage2VAE(nn.Module):
    """
    在第一阶段 latent 向量 z∈R^{d1} 上训练的 MLP-VAE。

    encoder: z → (μ_u, logvar_u)，u∈R^{d2}
    decoder: u → ẑ，配合可学习标量 log γ² 构成高斯似然 N(ẑ, γ²I)
    """

    def __init__(self, dim: int, hidden_dim: int = 512, latent_dim: int | None = None,
                 depth: int = 3) -> None:
        """
        Args:
            dim:        第一阶段 latent 维度 d1（即本模型的输入/输出维度）
            hidden_dim: MLP 隐藏层宽度
            latent_dim: 第二阶段 latent 维度 d2，默认与 d1 相同
            depth:      编码器/解码器各自的隐藏层数量
        """
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim or dim

        def trunk(in_dim: int) -> nn.Sequential:
            """depth 层 (Linear→ReLU)，输出宽度 hidden_dim。"""
            layers: list[nn.Module] = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
                d = hidden_dim
            return nn.Sequential(*layers)

        # 编码器：共享主干 + 双头（μ_u, logvar_u）
        self.enc = trunk(dim)
        self.fc_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, self.latent_dim)

        # 解码器：主干 + 线性输出回 d1 维
        self.dec = nn.Sequential(trunk(self.latent_dim), nn.Linear(hidden_dim, dim))

        # 可学习的全局观测对数方差 log γ²（标量），初始化为 0 → γ=1
        self.log_gamma2 = nn.Parameter(torch.zeros(()))

    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(z)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return self.dec(u)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_u, log_var_u = self.encode(z)
        u = self.reparameterize(mu_u, log_var_u)
        z_recon = self.decode(u)
        return z_recon, mu_u, log_var_u

    def loss(self, z: torch.Tensor, beta: float = 1.0
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        负 ELBO（按样本平均）：高斯 NLL 重建 + beta * KL。

        重建项使用可学习 γ：
          NLL = 0.5 * [ (z-ẑ)²/γ² + log γ² + log 2π ]  （逐维求和）
        KL(q(u|z) || N(0,I)) 解析式。

        beta < 1（warmup 阶段）防止 KL 在重建学起来前压垮编码器导致坍塌：
          β=1 且 γ 自由时，模型易陷入「解码器恒输出均值、γ²→1 把一切当噪声、
          KL=0」的退化解（实测会发生）。先用 β≈0 学好 z 流形再升 β 可避免。

        Returns: total, recon_nll, kl  （均为 batch 平均的标量）
        """
        z_recon, mu_u, log_var_u = self.forward(z)
        # 下界 γ²，避免 log γ² → -inf 数值发散
        log_gamma2 = self.log_gamma2.clamp(min=-10.0)
        gamma2 = log_gamma2.exp()
        # 高斯 NLL，对 latent 维求和、对 batch 求平均
        recon = 0.5 * (
            ((z - z_recon) ** 2) / gamma2 + log_gamma2 + math.log(2 * math.pi)
        ).sum(dim=1).mean()
        kl = (-0.5 * (1 + log_var_u - mu_u.pow(2) - log_var_u.exp())).sum(dim=1).mean()
        return recon + beta * kl, recon, kl

    @torch.no_grad()
    def sample_latent(self, n: int, device: torch.device) -> torch.Tensor:
        """从 N(0,I) 采 u 并解码得到第一阶段 latent z（取解码均值，不加 γ 噪声）。"""
        u = torch.randn(n, self.latent_dim, device=device)
        return self.decode(u)

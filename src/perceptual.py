"""
基于 VGG 的感知损失（Perceptual Loss），用于图像重建任务。

参考：Johnson et al., ECCV 2016 — 在预训练网络的特征空间中
比较重建图与原图，而非仅在像素空间比较，能更好地保留纹理与结构。

实现要点：
  - 使用冻结的 VGG-16 特征提取器，截取至 relu3_3（前 16 层）
  - 训练图像的像素值域需先映射到 ImageNet 归一化空间，再送入 VGG
"""

import torch
import torch.nn as nn
from torchvision import models

# ImageNet 预训练权重对应的标准化均值与标准差（RGB 三通道）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# nn.Module作为基类
class VGGPerceptualLoss(nn.Module):
    """
    在 VGG-16 relu3_3 特征图上计算 L2 距离作为感知损失。

    所有 VGG 参数冻结，仅作为固定特征提取器；梯度只回传到 recon_x。
    """
    # ->None 函数不返回值
    def __init__(self, pixel_range: str = "11") -> None:
        """
        Args:
            pixel_range: 输入张量的像素值域。
                "11" 表示 [-1, 1]（与 Tanh 解码器 + CIFAR 预处理一致）；
                其他值表示已在 [0, 1]。
        """
        super().__init__()
        # 加载在 ImageNet 上预训练的 VGG-16 特征部分
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        # 只保留 features 的前 16 层，对应 relu3_3 输出
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        # 冻结参数：感知损失不更新 VGG 权重
        for p in self.features.parameters():
            p.requires_grad = False
        # 将均值/标准差注册为 buffer，随模型自动搬到正确 device
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        self.pixel_range = pixel_range

    def _to_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """
        将模型输入张量转换为 VGG 期望的 ImageNet 归一化格式。

        步骤：若像素在 [-1,1]，先线性映射到 [0,1]；再 (x - mean) / std。
        """
        if self.pixel_range == "11":
            # [-1, 1] -> [0, 1]
            x = (x + 1.0) * 0.5
        return (x - self.mean) / self.std

    def forward(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        计算重建图与真值图在 VGG 特征空间中的平方误差之和。

        注意：返回的是 batch 内所有元素的总和（未除以 batch_size 或 dataset_size），
        调用方在 vae_loss 中会除以 dataset_size 以与 ELBO 标度一致。

        Args:
            recon_x: 解码器输出，形状 (B, 3, H, W)
            x:       原始输入，形状与 recon_x 相同

        Returns:
            标量张量：sum((f(recon) - f(x))^2)
        """
        recon_f = self.features(self._to_imagenet(recon_x))
        target_f = self.features(self._to_imagenet(x))
        return torch.sum((recon_f - target_f) ** 2)

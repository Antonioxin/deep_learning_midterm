# VAE 模型改进策略

> 本文档记录当前模型在 CIFAR-10 数据集上的已知不足，以及对应的改进方向，供后续实验迭代参考。

---

## 一、现状：CIFAR-10 生成质量差的原因

### 1.1 根本原因：像素级重建损失无法捕捉高频细节

当前损失函数（`losses.py`）使用 MSE 作为重建项：

```
recon_loss = F.mse_loss(recon_x, x, reduction="sum") / dataset_size
```

MSE 等价于假设解码器服从逐像素独立的高斯分布 `p(x|z) = N(recon_x, I)`。  
当同一类别（如"猫"）在姿势、背景、光照上差异极大时，模型的最优策略是输出所有可能样本的**像素均值**，结果是模糊、无法辨认的图像。

**量化指标**（来自 `exp_20260507_164731_cifar10_conv_v2`）：

| 指标 | 值 |
|---|---|
| 收敛 recon_loss | ~0.124 |
| per-pixel RMSE | ~0.125（像素值范围 12.5%） |
| avg KL | ~0.060 |
| KL per latent dim | ~0.183 nats |

per-pixel RMSE 达 12.5%，说明平均每个像素存在严重误差，生成结果无法识别。

### 1.2 Sigmoid 输出层梯度饱和

`conv_vae.py` 解码器末层使用 `nn.Sigmoid()`：

- Sigmoid 在饱和区（输出接近 0 或 1）梯度趋于零
- CIFAR-10 存在大量高对比区域（深色背景、高亮物体），这些区域的梯度在训练中严重衰减
- 导致高对比区域重建质量尤为差

### 1.3 编码器感受野不足

当前编码器仅 3 个 stride-2 卷积层，空间分辨率变化为：

```
32×32 → 16×16 → 8×8 → 4×4
```

最终 4×4 特征图每点感受野约覆盖原图 13 像素，无法建立对全局语义（物体轮廓、类别整体结构）的理解，而 CIFAR-10 的辨识特征往往依赖全局结构。

### 1.4 后验坍塌（Posterior Collapse）

收敛时每个 latent 维度的 KL ≈ 0.183 nats，接近先验 N(0, I)，说明 128 维潜空间中大量维度未携带有效信息。解码器绕开潜变量直接"猜"均值图像（即后验坍塌）。Beta warmup 有所缓解，但未从根本上解决。

### 1.5 训练过早收敛，缺少学习率调度

Loss 曲线在约第 80 epoch 后完全平坦，但配置中无 lr scheduler。固定 lr=1e-3 在平坦区继续训练没有收益。

---

## 二、改进策略

### 改进 1：引入感知损失（Perceptual Loss）⭐⭐⭐

**针对问题**：1.1（像素级损失根本性不足）

**原理**：用预训练的 VGG 网络提取特征，在特征空间而非像素空间度量重建误差（Johnson et al., 2016）：

```
L_perceptual = || phi(recon_x) - phi(x) ||^2_F
```

其中 `phi` 是 VGG-16 relu3_3 层的输出。特征空间的距离对人类感知更敏感，可以捕捉纹理、结构等高频信息。

**实施要点**：
- 需要冻结 VGG 权重（`requires_grad=False`）
- 输入需要归一化到 ImageNet 均值/方差
- 与像素级 MSE 加权混合：`L_recon = λ_pixel * MSE + λ_perc * L_perceptual`

**预期效果**：生成样本清晰度显著提升，纹理更自然。

---

### 改进 2：替换激活函数，解决梯度饱和 ⭐⭐

**针对问题**：1.2（Sigmoid 梯度饱和）

**方案**：将解码器末层 `Sigmoid` 替换为 `Tanh`，同时将数据归一化从 [0, 1] 改为 [-1, 1]：

```python
# datasets.py 中修改 transform：
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → [-1, 1]

# conv_vae.py 中修改解码器末层：
nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
nn.Tanh(),  # 替换 Sigmoid
```

Tanh 在 (-1, 1) 区间梯度更均匀，饱和区更窄，有利于高对比区域的梯度传播。  
对应重建损失改用 MSE（[-1, 1] 范围内 MSE 仍适用）。

---

### 改进 3：加深编码器，增加感受野 ⭐⭐

**针对问题**：1.3（感受野不足）

**方案**：在现有 3 层卷积基础上增加 Residual Block，或增加至 4 层卷积：

```
32×32 → 16×16 → 8×8 → 4×4 → 2×2
```

推荐在每个 stride-2 卷积后增加一个 1×1 或 3×3 的残差卷积（不改变分辨率），以增加非线性表达能力而不引入更多下采样。

---

### 改进 4：Free Bits 约束，缓解后验坍塌 ⭐⭐

**针对问题**：1.4（posterior collapse）

**原理**：Kingma et al. (2016) "Improving Variational Inference with Inverse Autoregressive Flow" 提出 Free Bits 机制——对每个维度单独施加 KL 下界 λ，使模型至少利用每个维度传递 λ nats 的信息：

```
L_KL = sum_j max(λ, KL_j)
```

典型取值 λ = 0.25 ~ 2.0 nats/dim。

与当前 beta warmup 配合使用效果更好：warmup 防止训练初期 KL 项主导，free bits 防止收敛后的坍塌。

---

### 改进 5：学习率衰减 ⭐

**针对问题**：1.5（提前收敛）

**方案**：在 `train.py` 中加入 `CosineAnnealingLR`：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-5
)
# 每个 epoch 末调用：
scheduler.step()
```

Cosine 衰减在训练后期平滑降低学习率，帮助模型在 loss 平坦区找到更优的局部极小值。

---

### 改进 6（进阶）：VAE-GAN 混合框架 ⭐⭐⭐

**针对问题**：1.1（根本性限制）

**原理**：Larsen et al. (2016) "Autoencoding beyond pixels using a learned similarity metric" 将 VAE 重建损失与 GAN 判别器结合：

```
L_total = L_VAE_KL + L_GAN_recon
```

判别器提供比 MSE/BCE 更丰富的重建信号，可以学习感知相关的相似度度量。  
这是从架构层面解决"平均图像"问题的最彻底方案。

**注意**：引入 GAN 训练稳定性问题，需要较多调参经验，建议在前几项改进验证有效后再考虑。

---

## 三、改进优先级汇总

| 优先级 | 改进项 | 难度 | 预期收益 |
|---|---|---|---|
| P0 | 感知损失（Perceptual Loss） | 中 | 高（清晰度质变） |
| P1 | Sigmoid → Tanh + 数据归一化 | 低 | 中（改善梯度流） |
| P1 | 加深编码器（+ResBlock） | 低 | 中（更好语义理解） |
| P2 | Free Bits 防坍塌 | 中 | 中（latent 利用率提升） |
| P2 | CosineAnnealingLR | 极低 | 低-中（训练收敛改善） |
| P3 | VAE-GAN 混合框架 | 高 | 极高（但训练不稳定） |

---

## 四、参考文献

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv:1312.6114*
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution. *ECCV 2016*
- Larsen, A. B. L., Sønderby, S. K., Larochelle, H., & Winther, O. (2016). Autoencoding beyond pixels using a learned similarity metric. *ICML 2016*
- Kingma, D. P., et al. (2016). Improving Variational Inference with Inverse Autoregressive Flow. *NeurIPS 2016*
- Higgins, I., et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*
- Bowman, S. R., et al. (2016). Generating Sentences from a Continuous Space. *CoNLL 2016*

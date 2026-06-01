# CIFAR-10 VAE 生成质量改进：实验过程与教训

本文件记录从「单层 ConvVAE 采样糊」出发，经诊断、2-Stage 修复、分层 VAE、到分层 + learned
prior 的完整实验链，含**失败的尝试与教训**。所有结论以 **FID**（vs CIFAR-10 测试集 10k，
pytorch-fid 官方 InceptionV3）为准。代表性图见 [`figures/`](figures/)。

> 复现脚本均在 `src/`，配置在 `configs/`。实验产物（权重/逐 epoch 图/日志）在 `experiments/`
> 下（已 gitignore，本地生成）。

---

## 总览：FID 横向对比（越低越好）

| 模型 / 采样方式 | FID | 备注 |
|---|---:|---|
| 单层 ConvVAE v4，z~N(0,I) | 155 | 基线 |
| 单层 + 2-Stage learned prior (v5) | 129 | 单层最佳 |
| 分层 VAE v3，原生采样 | 95 | 分层原生已是大跃升 |
| **分层 v3 + 2-Stage(只对 z_top)** | **85** | **全项目最佳** |
| 分层 v3 重建（z_bottom 用后验，真·下界） | 33 | 生成的理论上界参考 |
| 分层 + 2-Stage(联合 latent) | 198 | ❌ 失败，见第 4 节 |

**一句话**：分层 + 对顶层套 learned prior 得到最佳 FID 85；过程中最大的教训是「采样糊」长期被
错误指标误导，以及 2-Stage 技巧被错误地套用到高维高噪 latent 上而失败。

---

## 1. 诊断：单层 ConvVAE「重建好、采样糊」的真正原因

图 [`figures/01_single_level_diagnosis.png`](figures/01_single_level_diagnosis.png)（real / recon / 先验采样三行）：
重建清晰、先验采样糊。量化诊断（`src/diag_posterior.py`）：

- 64/64 维全 active、总 KL≈213 nats/图 → **无后验坍塌**；
- 编码器后验 σ≈0.038 → 近乎确定性编码（VAE 退化成 AutoEncoder）；
- 聚合后验逐维 std≈1.0、‖μ‖≈√64 → 一阶/二阶矩已匹配 N(0,I)；
- 但 N(0,I) 采样解码多样性 0.34 ≪ 真实编码 0.44 → **洞在 latent 的联合结构**。

**结论**：不是 MSE/重建上限、也不是坍塌，而是聚合后验与先验的**联合结构失配（prior hole）**。
拟合一个全协方差高斯/GMM 到编码点云，采样多样性当场从 0.34 拉到 0.44——验证了诊断。

## 2. 修复：2-Stage VAE（单层，成功）

`src/train_stage2.py` + `configs/cifar10_2stage_v5.yaml`。在冻结一阶段 latent 上训第二个
MLP-VAE 建模 q(z)，采样 u~N(0,I)→Stage2→Stage1。
图 [`figures/02_single_level_2stage.png`](figures/02_single_level_2stage.png)：右侧明显更清晰多样。

- **FID 155 → 129**；多样性 std 0.36→0.46。
- **教训（坑）**：Stage-2 首训**完全后验坍塌**（KL=0、γ=1、采样塌成一点）——γ 可学且初始为 1 时
  模型偷懒把一切当观测噪声。**修复 = beta warmup**（KL 权重 0→1），先学好重建再加 KL。

## 3. 分层 VAE（v1→v2→v3）：先被误判，后被 FID 翻案

`src/models/hierarchical_vae.py`，两级（z_top 向量 + z_bottom 空间 64×4×4）。三轮调参
（v1 bottom 成 AE 码 → v2 高 beta 压制致先验「方差逃逸」作弊 → v3 钳制方差）。

图 [`figures/03_hier_decompose_diagnosis.png`](figures/03_hier_decompose_diagnosis.png)（real / recon / 真实z_top+条件先验底层 / 全先验采样）
定位出两个洞：顶层 q(z_top)≠N(0,I)，且底层条件先验弱于后验。

- **当时的误判**：基于样本「看着糊」和 `prior_std≈0.40` 认为分层没改善。
- **FID 翻案**：分层**原生采样 FID=95**，远优于单层基线 155 与单层 2-stage 129。
  **`prior_std`（pixel 跨样本 std）衡量的是对比度，不是分布保真度；FID 才是对的指标。**
  分层那种「低对比、偏柔」的样本，在 Inception 特征分布上其实与真实 CIFAR 相当接近。

> **核心教训**：不要用 pixel 方差/肉眼判断生成质量好坏，要用 FID。

## 4. 分层 + 2-Stage（联合 latent）：失败 ❌

`src/train_stage2_hier.py --latent joint`。对 1088 维联合 latent (z_top 64 + z_bottom 1024)
套 learned prior，意图同时补两个洞。
图 [`figures/04_hier_2stage_joint_FAILED.png`](figures/04_hier_2stage_joint_FAILED.png)：右侧塌成重复网格、几无多样性。

- **FID 95 → 198（恶化）**。收敛 γ≈0.94 → Stage-2 几乎无法重建联合 latent，只输出均值 latent。
- **原因 / 教训**：hier_v3 **底层后验 σ≈0.9**，z_bottom 的 1024 维大部分是**采样噪声**而非可建模
  结构。**2-Stage / learned prior 只在低维、近确定性 latent 上成立**（对比 v5：σ=0.038、64 维，
  成功）。对高维高噪 latent 套 2-Stage，不仅学不动，还破坏了「z_bottom 由条件先验依赖 z_top」
  这一原本正确的结构。

## 5. 分层 + 2-Stage（只对 z_top）：正确用法，最佳结果 ✓

`src/train_stage2_hier.py --latent top`。只对 64 维 z_top 套 learned prior，底层保留条件先验。
图 [`figures/05_hier_2stage_top_BEST.png`](figures/05_hier_2stage_top_BEST.png)：右侧更多样、对比更强，且**未崩坏**。

- **FID 95 → 85**；γ≈0.66、KL≈23.8、健康无坍塌（同 v5 行为）。
- 闭合了「z_top~N(0,I)」与「真实 z_top」之间约 46% 的差距。

## 6. 潜空间分析（单层 ConvVAE v4）

`src/visualize.py`，在 64 维向量 latent 上产出三张图：

- **插值** [`figures/06_latent_interpolation.png`](figures/06_latent_interpolation.png)：两张真实图编码后
  线性插值再解码，过渡平滑且语义连续 → latent 空间是连续的、没有明显空洞断裂。
- **维度遍历** [`figures/07_latent_traversal.png`](figures/07_latent_traversal.png)：围绕一张真实图，
  逐个改变最活跃的 12 维（±3σ）。每维主要调制颜色/明暗/背景等外观属性，物体结构保持；但**没有
  单一维度干净地对应某个语义因子** → 标准 VAE 未实现解耦（与 β-VAE 不同，符合预期）。
  注：若以数据集均值（z≈0）为基码，会解码成「均值绿斑 mush」，这反向印证了第 1 节的 prior-mean 退化。
- **t-SNE** [`figures/08_latent_tsne.png`](figures/08_latent_tsne.png)：2500 张测试图编码 μ 降到 2D 按类别
  上色，**类别并未清晰分簇**（混成一团）→ 无监督 VAE 的 latent 主要按低层外观（颜色/亮度/构图）组织，
  而非类别语义。这与「CIFAR 难生成」一致：latent 没把类别结构编码进去。

复现：`python src/visualize.py --exp experiments/exp_20260527_092330_cifar10_conv_v4`

## 7. 消融实验（单层 ConvVAE v4 架构）

`src/run_ablation.py`：固定基准（β=4, λ_perc=0.01, free_bits=0, latent_dim=64），每次只改一个因子，
统一 40-epoch 预算，比生成 FID。完整表见 [`ablation.md`](ablation.md)。

| 改动 | 生成 FID | 结论 |
|---|---:|---|
| β = 1 / 4 / 8 | 179.5 / 177.2 / 176.8 | **β 在 [1,8] 几乎不影响生成 FID** |
| 去掉感知损失 (λ_perc=0) | 209.8 | **影响最大：+33 FID**，感知损失是关键 |
| free_bits = 0.5 | 178.7 | 无影响（本就无后验坍塌可救，呼应第 1 节） |
| latent_dim = 32 / 64 / 128 | 207.6 / 177.2 / 183.0 | **64 接近最优**；32 欠拟合，128 无增益 |

> 注：FID 为 40-epoch 缩短训练所得，绝对值高于 100-epoch 正式 v4(=155)，仅用于组内相对比较。

**要点**：(1) **感知损失是最有效的单一因子**，去掉显著变差；(2) **β 不是这里的瓶颈**——在很宽范围内
对 FID 几乎无影响，印证「问题在 prior hole 而非 KL 权重」的诊断；(3) latent_dim=64 选得合适；
(4) free_bits 无用，再次说明本模型不存在后验坍塌。

## 8. 剩余空间与后续方向

- 本模式下界 73（z_top 真实、底层走条件先验），top-only 2-Stage 已逼到 85。
- 真·重建下界 **33**，差距全在**底层**（z_bottom 条件先验 vs 后验）。
- 要逼近 33：需**更强的底层先验**（NVAE 式 prior/posterior 残差耦合、空间自回归）+ **更好的
  似然**（离散 logistic 混合），而非简单 learned prior（第 4 节已证此路不通）。

---

## 附：关键脚本

| 脚本 | 作用 |
|---|---|
| `src/diag_posterior.py` | 单层聚合后验诊断（KL/active units/联合结构） |
| `src/train_stage2.py` | 单层 2-Stage VAE 训练 + 对比图 |
| `src/train_stage2_hier.py` | 分层 2-Stage（`--latent top/joint`）+ 对比图 + FID |
| `src/eval_fid.py` | 通用 FID 评测（pytorch-fid InceptionV3） |

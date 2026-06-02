# 实验环境配置

生成时间：2026-06-02 18:51 UTC  
项目路径：`/root/autodl-tmp/deep_learning_midterm`

## 1. 系统环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 22.04.3 LTS (Jammy Jellyfish) |
| Linux Kernel | 5.15.0-78-generic |
| 架构 | x86_64 |
| 主机名 | autodl-container-8f0d40a27c-e6e0f249 |
| Python | 3.10.8 |
| Python 路径 | `/root/miniconda3/bin/python` |
| Conda 环境 | `base` (`/root/miniconda3`) |
| pip | 22.3.1 |

## 2. 硬件环境

| 项目 | 配置 |
|---|---|
| CPU | Intel(R) Xeon(R) Platinum 8470Q |
| CPU 核心/线程 | 2 sockets, 52 cores/socket, 2 threads/core，共 208 逻辑 CPU |
| 内存 | 754 GiB |
| Swap | 0 B |
| 项目所在磁盘 | `/dev/md0`, 50 GiB 总容量，20 GiB 可用 |

## 3. GPU 与 CUDA

| 项目 | 配置 |
|---|---|
| GPU 型号 | NVIDIA GeForce RTX 4080 |
| GPU 数量 | 1 |
| 显存 | 32760 MiB，PyTorch 识别约 31.47 GiB |
| CUDA Compute Capability | 8.9 |
| NVIDIA Driver | 580.105.08 |
| nvidia-smi CUDA Version | 13.0 |
| nvcc CUDA Toolkit | 12.1, V12.1.105 |
| PyTorch CUDA | 12.1 |
| cuDNN | 8902 |
| CUDA 可用性 | `torch.cuda.is_available() == True` |

## 4. 深度学习框架与主要依赖

### 当前环境实测版本

| 包 | 版本 |
|---|---|
| torch | 2.1.2+cu121 |
| torchvision | 0.16.2+cu121 |
| numpy | 1.26.3 |
| scipy | 1.14.1 |
| matplotlib | 3.8.2 |
| pillow | 10.2.0 |
| PyYAML | 6.0.1 |
| tqdm | 4.67.3 |
| pytorch-fid | 0.3.0 |
| scikit-learn | 1.7.2 |
| opencv/cv2 | 4.10.0 |

备注：`requirements.txt` 中声明了 `seaborn>=0.12.0`，但当前 Python 环境未检测到 `seaborn`。

### 项目依赖要求

项目根目录 `requirements.txt` 声明的最低依赖如下：

```txt
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
pillow>=10.0.0
pyyaml>=6.0
seaborn>=0.12.0
scipy>=1.10.0
pytorch-fid>=0.3.0
```

README 中说明的基础环境要求：

| 项目 | 要求 |
|---|---|
| Python | 3.10 或 3.11 |
| PyTorch | >= 2.1.0 |
| GPU | CUDA 推荐，也支持 Apple MPS |

## 5. 项目信息

| 项目 | 信息 |
|---|---|
| Git 分支 | `main` |
| Git commit | `f4534cd` |
| 当前工作区状态 | 存在未提交改动：`src/diffusion.py`；未跟踪文件：`src/visualize_ddpm.py` |

## 6. 环境提取命令

本文件中的环境信息主要来自以下命令：

```bash
uname -a
cat /etc/os-release
python --version
python -m pip --version
nvidia-smi
nvcc --version
lscpu
free -h
df -h /root/autodl-tmp/deep_learning_midterm
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

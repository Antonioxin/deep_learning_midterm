"""
数据加载与设备选择工具模块。

职责：
  - get_device(): 按 MPS > CUDA > CPU 优先级选择训练设备
  - get_dataloader(): 统一入口，支持 MNIST / Fashion-MNIST / CIFAR-10 / QuickDraw
  - QuickDraw 自定义 Dataset 与 90/10 划分逻辑
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def get_device() -> torch.device:
    """
    按优先级选择计算设备：MPS（Apple Silicon）> CUDA > CPU。

    MPS 需要 macOS 12.3+ 且 PyTorch >= 1.12；Linux 服务器通常走 CUDA 分支。
    选择结果会打印到 stdout，便于日志中确认实际硬件。
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[device] Using CPU")
    return device


def get_dataloader(
    dataset_name: str,
    batch_size: int,
    train: bool = True,
    data_dir: str = "data/",
    pixel_range: str = "01",
) -> tuple[DataLoader, int]:
    """
    加载指定数据集并返回 DataLoader 与当前划分的样本总数。

    像素归一化策略（与解码器激活、损失函数配套）：
      pixel_range="01": ToTensor 后像素 ∈ [0, 1]，配合 Sigmoid + BCE/MSE
      pixel_range="11": 再 Normalize(0.5,0.5) 映射到 [-1, 1]，配合 Tanh + MSE

    Args:
        dataset_name: "mnist" | "fashion_mnist" | "cifar10" | "quickdraw"
        batch_size:   小批量大小
        train:        True 训练集 / False 测试集（QuickDraw 为 90/10 划分）
        data_dir:     数据根目录，torchvision 会在此下载或读取
        pixel_range:  "01" 或 "11"

    Returns:
        loader:       DataLoader，num_workers=0（避免 MPS 多进程问题）
        dataset_size: 该 split 的样本数，用于 loss 除以 dataset_size
    """
    # 基础变换：PIL/ndarray uint8 [0,255] → float tensor [0,1]
    transform_steps = [transforms.ToTensor()]
    if pixel_range == "11":
        # (x - 0.5) / 0.5：将 [0,1] 线性映射到 [-1, 1]
        transform_steps.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    transform = transforms.Compose(transform_steps)

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    name = dataset_name.lower()
    if name == "mnist":
        dataset = datasets.MNIST(root=str(root), train=train, download=True, transform=transform)
    elif name == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root=str(root), train=train, download=True, transform=transform
        )
    elif name == "quickdraw":
        # QuickDraw 为自定义 npy 格式，走独立加载路径
        return get_quickdraw_dataloader(batch_size=batch_size, train=train, data_dir=data_dir)
    elif name == "cifar10":
        # download=False：假定数据已放在 data_dir 下，避免重复下载
        dataset = datasets.CIFAR10(
            root=str(root), train=train, download=False, transform=transform
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            "Choose 'mnist', 'fashion_mnist', 'quickdraw', or 'cifar10'."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,       # 训练集打乱，测试集顺序固定
        num_workers=0,       # macOS MPS 下 num_workers>0 易出问题
        pin_memory=False,
    )
    return loader, len(dataset)


class QuickDrawDataset(Dataset):
    """
    从 data/QuickDraw/*.npy 加载 Google Quick Draw 简笔画数据。

    每个 .npy 文件对应一个类别，形状 (N, 784)，uint8 像素 [0,255]；
    合并后归一化为 float32 [0,1]，标签为文件排序后的类别索引 0..K-1。
    """

    def __init__(self, data_dir: str = "data/") -> None:
        qd_dir = Path(data_dir) / "QuickDraw"
        arrays, labels = [], []
        # 按文件名排序保证类别索引可复现
        for idx, f in enumerate(sorted(qd_dir.glob("*.npy"))):
            arr = np.load(f)                          # (N, 784) uint8
            arrays.append(arr)
            labels.append(np.full(len(arr), idx, dtype=np.int64))
        # 拼接所有类别，像素缩放到 [0, 1]
        self.data   = np.concatenate(arrays, axis=0).astype(np.float32) / 255.0
        self.labels = np.concatenate(labels, axis=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回展平向量 (784,) 与 long 类型标签（VAE 训练通常忽略 y）
        x = torch.from_numpy(self.data[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def get_quickdraw_dataloader(
    batch_size: int,
    train: bool = True,
    data_dir: str = "data/",
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[DataLoader, int]:
    """
    加载完整 QuickDraw 后做固定种子的 90/10 训练/测试划分。

    Args:
        batch_size:  批量大小
        train:       True 返回训练子集，False 返回测试子集
        data_dir:    数据根目录
        train_ratio: 训练集比例，默认 0.9
        seed:        random_split 的随机种子，保证可复现

    Returns:
        loader, len(split)
    """
    full_dataset = QuickDrawDataset(data_dir=data_dir)
    n_train = int(len(full_dataset) * train_ratio)
    n_test  = len(full_dataset) - n_train
    train_set, test_set = random_split(
        full_dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    split = train_set if train else test_set
    loader = DataLoader(
        split,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=False,
    )
    return loader, len(split)

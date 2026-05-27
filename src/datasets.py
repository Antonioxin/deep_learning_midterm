"""
Device selection and dataset loading utilities.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def get_device() -> torch.device:
    """
    Select compute device in priority order: MPS > CUDA > CPU.

    MPS is Apple Silicon's Metal Performance Shaders backend, available
    on macOS 12.3+ with PyTorch >= 1.12.
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
) -> tuple[DataLoader, int]:
    """
    Load MNIST or Fashion-MNIST and return (DataLoader, dataset_size).

    Images are normalised to [0, 1] (no further standardisation) to match
    the Sigmoid output of the decoder and the Bernoulli BCE loss.

    Args:
        dataset_name: "mnist" or "fashion_mnist"
        batch_size:   mini-batch size
        train:        True for training split, False for test split
        data_dir:     root directory where torchvision downloads data

    Returns:
        loader:       DataLoader with num_workers=0 (MPS compatibility)
        dataset_size: total number of samples in this split
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # scales uint8 [0,255] → float [0,1]
    ])

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
        return get_quickdraw_dataloader(batch_size=batch_size, train=train, data_dir=data_dir)
    elif name == "cifar10":
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
        shuffle=train,
        num_workers=0,   # num_workers>0 conflicts with MPS on macOS
        pin_memory=False,
    )
    return loader, len(dataset)


class QuickDrawDataset(Dataset):
    """
    Loads all .npy files under data/QuickDraw/ and concatenates them.

    Each .npy file has shape (N, 784), dtype uint8, values in [0, 255].
    We normalise to float32 [0, 1] to match the Sigmoid decoder output.
    Labels are class indices (0..K-1), one per .npy file.
    """

    def __init__(self, data_dir: str = "data/") -> None:
        qd_dir = Path(data_dir) / "QuickDraw"
        arrays, labels = [], []
        for idx, f in enumerate(sorted(qd_dir.glob("*.npy"))):
            arr = np.load(f)                          # (N, 784) uint8
            arrays.append(arr)
            labels.append(np.full(len(arr), idx, dtype=np.int64))
        self.data   = np.concatenate(arrays, axis=0).astype(np.float32) / 255.0
        self.labels = np.concatenate(labels, axis=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.data[idx])          # (784,) float32
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
    Load all QuickDraw .npy files, do an 90/10 train/test split, return DataLoader.
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

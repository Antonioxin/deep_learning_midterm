# VAE Project — Variational Autoencoder for Image Generation

Course midterm project: implement a standard VAE and perform image generation,
latent space analysis, and ablation studies on MNIST / Fashion-MNIST / CIFAR-10.

---

## Environment Requirements

| Item | Requirement |
|---|---|
| Python | 3.10 or 3.11 |
| PyTorch | >= 2.1.0 |
| GPU | CUDA (recommended) or Apple MPS |

---

## Quick Start on a Remote Server

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL> vae-project
cd vae-project
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA servers, install the CUDA-enabled PyTorch first:

```bash
# Example for CUDA 12.1 — adjust the index URL for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Prepare data

MNIST and Fashion-MNIST are downloaded automatically on first run.

For CIFAR-10, download manually and place under `data/`:

```bash
mkdir -p data
# torchvision will auto-download MNIST/FashionMNIST to data/ on first run
```

For QuickDraw (optional), place `.npy` files under `data/QuickDraw/`:

```
data/QuickDraw/apple.npy
data/QuickDraw/cat.npy
...
```

### 4. Train a model

All training is launched from the project root via `src/train.py`.
Pass a config yaml and an experiment tag:

```bash
# Baseline FC-VAE on MNIST (latent_dim=20, 20 epochs)
python src/train.py --config configs/baseline.yaml --tag baseline

# Convolutional VAE on CIFAR-10
python src/train.py --config configs/cifar10_conv_v2.yaml --tag cifar10_conv_v2
```

Training output is saved automatically to:

```
experiments/exp_<YYYYMMDD_HHMMSS>_<tag>/
├── config.yaml        # snapshot of hyperparameters
├── checkpoints/       # .pt files every 5 epochs + final.pt
├── samples/           # generated image grids per epoch
└── logs/
    ├── train.log      # human-readable log
    └── metrics.csv    # epoch / loss / recon / kl / beta
```

### 5. Available configs

| Config file | Description |
|---|---|
| `configs/baseline.yaml` | FC-VAE, MNIST, latent_dim=20, 20 epochs |
| `configs/cifar10.yaml` | FC-VAE on CIFAR-10 |
| `configs/cifar10_conv.yaml` | ConvVAE on CIFAR-10 (v1) |
| `configs/cifar10_conv_v2.yaml` | ConvVAE on CIFAR-10 (v2, recommended) |
| `configs/quickdraw.yaml` | FC-VAE on QuickDraw sketches |

---

## Project Structure

```
vae-project/
├── requirements.txt
├── configs/               # YAML hyperparameter files (one per experiment)
├── src/
│   ├── models/
│   │   ├── vae.py         # Fully-connected VAE
│   │   └── conv_vae.py    # Convolutional VAE
│   ├── losses.py          # VAE ELBO loss (reconstruction + KL)
│   ├── datasets.py        # Data loading (MNIST / CIFAR-10 / QuickDraw)
│   ├── train.py           # Training entry point
│   ├── sample.py          # Sampling / generation utilities
│   └── visualize.py       # Latent space visualization and interpolation
├── notebooks/             # Exploratory notebooks (not part of final code)
└── experiments/           # Auto-generated per training run (gitignored)
```

---

## Notes

- The training script automatically selects the best available device:
  MPS (Apple Silicon) → CUDA → CPU.
- `num_workers=0` is set in all DataLoaders for compatibility with MPS;
  on CUDA servers you can increase this to speed up data loading.
- Random seeds are set via the `training.seed` field in each config yaml.

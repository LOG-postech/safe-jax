# SAFE: Finding Sparse and Flat Minima to Improve Pruning

**Authors:** Dongyeop Lee, Kwanhee Lee, Jinseok Chung, Namhoon Lee

**Venue:** ICML 2025, *Spotlight* poster

**Contact:** [dylee23@postech.ac.kr](mailto:kwanhee.lee@postech.ac.kr)

This repository contains the official JAX implementation for the paper [SAFE: Finding Sparse and Flat Minima to Improve Pruning](https://arxiv.org/abs/2506.06866). Our work introduces SAFE, an algorithm designed to find sparse and flat minima, leading to improved model pruning performance.

## 1. Abstract

Sparsifying neural networks often suffers from seemingly inevitable performance degradation, and it remains challenging to restore the original performance despite much recent progress.Motivated by recent studies in robust optimization, we aim to tackle this problem by finding subnetworks that are both sparse and flat at the same time.Specifically, we formulate pruning as a sparsity-constrained optimization problem where flatness is encouraged as an objective.We solve it explicitly via an augmented Lagrange dual approach and extend it further by proposing a generalized projection operation, resulting in novel pruning methods called SAFE and its extension, SAFE+. Extensive evaluations on standard image classification and language modeling tasks reveal that SAFE consistently yields sparse networks with improved generalization performance, which compares competitively to well-established baselines.In addition, SAFE demonstrates resilience to noisy data, making it well-suited for real-world conditions.

## 2. Requirements and Environment Setup

### Environments

- Python 3.8
- CUDA 11.x + cuDNN 8.x
- JAX 0.4.4 / Flax 0.6.5 / Optax 0.1.3
- [uv](https://docs.astral.sh/uv/getting-started/installation/) or conda

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/log-postech/safe-jax.git
    cd safe-jax
    ```

2.  **Install dependencies:**

    With [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended):
    ```bash
    uv sync
    source .venv/bin/activate
    ```

    With conda + pip:
    ```bash
    conda create -n safe python=3.8
    conda activate safe
    pip install -r requirements.txt
    ```

## 3. Usage

### Training

Train with default hyperparameters. Override with flags (e.g. `--sp`, `--seed`, `--checkpoint_name`).

```bash
bash scripts/cifar10_resnet20.sh --sp .95 --seed 1
```

**CIFAR-10:**

| Model | Default checkpoint name | Script |
| ----- | ----------------------- | ------ |
| ResNet-20x2 | `cifar10_ResNet20x2_safe_0.95_s1_<timestamp>` | `scripts/cifar10_resnet20.sh` |
| VGG19-bn | `cifar10_VGG19-bn_safe_0.95_s1_<timestamp>` | `scripts/cifar10_vgg19.sh` |

**CIFAR-100:**

| Model | Default checkpoint name | Script |
| ----- | ----------------------- | ------ |
| ResNet-32x2 | `cifar100_ResNet32x2_safe_0.95_s1_<timestamp>` | `scripts/cifar100_resnet32.sh` |
| VGG19-bn | `cifar100_VGG19-bn_safe_0.95_s1_<timestamp>` | `scripts/cifar100_vgg19.sh` |

### Evaluation

Evaluate a trained checkpoint by name. Use `list_checkpoints.sh` to see available checkpoints.

```bash
bash scripts/list_checkpoints.sh
```

| Eval | Script |
| ---- | ------ |
| Clean (BNT) | `bash scripts/eval_clean.sh --checkpoint_name <NAME>` |
| CIFAR-10C | `bash scripts/eval_cifar10c.sh --checkpoint_name <NAME>` |
| l∞-PGD | `bash scripts/eval_adversarial_linf.sh --checkpoint_name <NAME>` |
| l₂-PGD | `bash scripts/eval_adversarial_l2.sh --checkpoint_name <NAME>` |

For language implementation, please refer to [safe-torch](https://github.com/log-postech/safe-torch)


## 4. Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
    lee2025safe,
    title={SAFE: Finding Sparse and Flat Minima to Improve Pruning},
    author={Doegyeop, Lee and Kwanhee, Lee and Jinseok, Chung and Namhoon, Lee},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=10l1pGeOcK}
}
```

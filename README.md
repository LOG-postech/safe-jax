# SAFE: Finding Sparse and Flat Minima to Improve Pruning

**Authors:** Dongyeop Lee, Kwanhee Lee, Jinseok Chung, Namhoon Lee

**Venue:** ICML 2025, *Spotlight* poster

**Contact:** [dylee23@postech.ac.kr](mailto:kwanhee.lee@postech.ac.kr)

This repository contains the official JAX implementation for the paper [SAFE: Finding Sparse and Flat Minima to Improve Pruning](https://arxiv.org/abs/2506.06866). Our work introduces SAFE, an algorithm designed to find sparse and flat minima, leading to improved model pruning performance.

## 1. Abstract

Sparsifying neural networks often suffers from seemingly inevitable performance degradation, and it remains challenging to restore the original performance despite much recent progress.Motivated by recent studies in robust optimization, we aim to tackle this problem by finding subnetworks that are both sparse and flat at the same time.Specifically, we formulate pruning as a sparsity-constrained optimization problem where flatness is encouraged as an objective.We solve it explicitly via an augmented Lagrange dual approach and extend it further by proposing a generalized projection operation, resulting in novel pruning methods called SAFE and its extension, SAFE+. Extensive evaluations on standard image classification and language modeling tasks reveal that SAFE consistently yields sparse networks with improved generalization performance, which compares competitively to well-established baselines.In addition, SAFE demonstrates resilience to noisy data, making it well-suited for real-world conditions.

## 2. Requirements and Environment Setup

### Environments

- Python 3.8.0
- cuda 11.4.4
- cudnn 8.6.0
- nccl 2.11.4

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/log-postech/safe-jax.git 
    cd safe-jax
    ```

2.  **Create a virtual environment (recommended):**
    * Conda:
        ```bash
        conda create -n safe python=3.8
        conda activate safe
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    <!-- Please beware of JAX, jaxlib, flax, CUDA, CuDNN compatibility.
    >If you encounter issues with specific versions, especially for PyTorch with CUDA, please refer to the official PyTorch installation guide: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) -->

## 3. Usage

### Running SAFE

```bash
python train.py \
 --workdir=./logdir \
 --model VGG19-bn \
 --dataset cifar10 \
 --num_epochs 300 \
 --sparsifier safe \
 --lambda 0.0001 \
 --lambda_schedule cosine \
 --rho 0.1 \
 --dual_update_interval 32 \
 --sp .95 \
 --seed 1
```

### Evalation

```bash
python eval.py
 --workdir=./logdir \
 --model VGG19-bn \
 --dataset cifar10 \
 --num_epochs 300 \
 --sparsifier safe \
 --lambda 0.0001 \
 --lambda_schedule cosine \
 --rho 0.1 \
 --dual_update_interval 32 \
 --sp .95 \
 --seed 1
```

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

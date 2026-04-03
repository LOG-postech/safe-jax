#!/bin/bash
# Train SAFE on CIFAR-10 ResNet-20x2.
# Usage: bash scripts/cifar10_resnet20.sh --checkpoint_name my_model [--sp .9] [--seed 2] ...

python train.py \
 --workdir=./logdir \
 --model ResNet20x2 \
 --dataset cifar10 \
 --num_epochs 200 \
 --sparsifier safe \
 --lambda 0.001 \
 --lambda_schedule cosine \
 --rho 0.1 \
 --dual_update_interval 32 \
 --sp .95 \
 --seed 1 \
 "$@"

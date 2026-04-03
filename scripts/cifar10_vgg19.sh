#!/bin/bash
# Train SAFE on CIFAR-10 VGG19-bn.
# Usage: bash scripts/cifar10_vgg19.sh --checkpoint_name my_model [--sp .9] [--seed 2] ...

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
 --seed 1 \
 "$@"

#!/bin/bash
# Usage: bash scripts/eval_cifar10c.sh --checkpoint_name my_model [--corrupt_intensity 5]
python eval.py --mode corrupt --corrupt_intensity 3 "$@"

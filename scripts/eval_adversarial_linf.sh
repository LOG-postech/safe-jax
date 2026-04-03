#!/bin/bash
# Usage: bash scripts/eval_adversarial_linf.sh --checkpoint_name my_model
python eval.py --mode adversarial --attack_norm linf \
  --attack_bound $(python -c "print(1/255)") \
  --attack_lr $(python -c "print(1/255/4)") \
  "$@"

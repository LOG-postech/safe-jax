#!/bin/bash
# Usage: bash scripts/eval_adversarial_l2.sh --checkpoint_name my_model
python eval.py --mode adversarial --attack_norm l2 \
  --attack_bound $(python -c "print(3/255)") \
  --attack_lr $(python -c "print(3/255/4)") \
  "$@"

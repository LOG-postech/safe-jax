#!/bin/bash
# Usage: bash scripts/eval_clean.sh --checkpoint_name my_model
python eval.py --mode clean "$@"

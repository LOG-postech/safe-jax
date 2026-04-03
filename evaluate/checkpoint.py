"""Checkpoint resolution and shared flag configuration."""

import os
from absl import flags
import ml_collections

from train_utils import cfg2ckpt


def configure_flags():
    """Define all flags shared across train.py and eval scripts."""
    flags.DEFINE_string('workdir', './logdir', "Working directory for checkpoints.")
    flags.DEFINE_bool('half_precision', False, "Whether to use half precision.")

    flags.DEFINE_string('model', 'ResNet20', "Model architecture.")
    flags.DEFINE_string('dataset', 'cifar10', "Dataset (mnist, cifar10, cifar100).")
    flags.DEFINE_integer('num_epochs', 200, "Number of training epochs.")
    flags.DEFINE_integer('batch_size', 128, "Mini-batch size.")
    flags.DEFINE_integer('seed', 1, "Random seed.")

    flags.DEFINE_string('optimizer', 'sgd', "Optimizer (sgd, adam).")
    flags.DEFINE_float('lr', 0.1, "Learning rate.")
    flags.DEFINE_string('lr_schedule', 'cosine', "LR schedule (constant, linear, cosine, step).")
    flags.DEFINE_integer('warmup_epochs', 5, "Warmup epochs.")
    flags.DEFINE_float('wd', 0.0001, "Weight decay.")
    flags.DEFINE_float('momentum', 0.9, "Momentum.")

    flags.DEFINE_string('sparsifier', 'safe', "Sparsifier (safe, admm, gmp, iht, none).")
    flags.DEFINE_float('sp', 0.95, "Target sparsity.")
    flags.DEFINE_string('sp_scope', 'global', "Sparsity scope (global, layerwise).")
    flags.DEFINE_integer('bnt_sample_size', 10000, "BNT calibration samples.")

    flags.DEFINE_float('lambda', 0.0001, "Lambda for SAFE/ADMM.")
    flags.DEFINE_string('lambda_schedule', 'cosine', "Lambda schedule (constant, linear, cosine).")
    flags.DEFINE_integer('dual_update_interval', 32, "Dual update interval for SAFE/ADMM.")
    flags.DEFINE_string('sp_schedule', 'cubic', "Sparsity schedule for GMP/IHT.")
    flags.DEFINE_float('rho', 0.1, "Rho for SAFE.")

    flags.DEFINE_float('label_noise_ratio', 0.0, "Label noise ratio.")

    flags.DEFINE_string('checkpoint_dir', '', "Explicit checkpoint directory.")
    flags.DEFINE_string('checkpoint_name', '', "Named checkpoint from ckpt_index.json.")


def _parse_config_from_path(path):
    """Extract config key-values from a checkpoint directory path."""
    from pathlib import Path
    known_keys = ['dataset', 'optimizer', 'model', 'num_epochs', 'lr', 'wd',
                  'momentum', 'label_noise_ratio', 'sparsifier', 'sp', 'sp_scope',
                  'sp_schedule', 'lambda', 'lambda_schedule', 'rho',
                  'dual_update_interval', 'seed']
    type_map = {
        'num_epochs': int, 'lr': float, 'wd': float, 'momentum': float,
        'label_noise_ratio': float, 'sp': float, 'lambda': float, 'rho': float,
        'dual_update_interval': int, 'seed': int,
    }
    config = {}
    for part in Path(path).parts:
        for k in sorted(known_keys, key=len, reverse=True):
            if part.startswith(k + '_'):
                val = part[len(k) + 1:]
                cast = type_map.get(k, str)
                try:
                    config[k] = cast(val)
                except ValueError:
                    config[k] = val
                break
    return config


def resolve_checkpoint(flag_values):
    """Resolve checkpoint path and config from flags.

    Returns (output_dir, configs) where configs is a ConfigDict
    with the training config (either from flags or from the checkpoint index).
    """
    flag_dict = flag_values.flag_values_dict()
    checkpoint_name = flag_dict.pop('checkpoint_name', '')
    checkpoint_dir = flag_dict.pop('checkpoint_dir', '')

    # Remove non-config flags
    for k in ['checkpoint_every_epoch', 'resume_training']:
        flag_dict.pop(k, None)

    if checkpoint_name:
        from ckpt_index import resolve_name
        output_dir, saved_config = resolve_name(checkpoint_name)
        if saved_config:
            configs = ml_collections.ConfigDict(saved_config)
        else:
            # Auto-registered: parse config from path components
            configs = ml_collections.ConfigDict(flag_dict)
            configs.update(_parse_config_from_path(output_dir))
    elif checkpoint_dir:
        output_dir = checkpoint_dir
        configs = ml_collections.ConfigDict(flag_dict)
    else:
        configs = ml_collections.ConfigDict(flag_dict)
        output_dir, *_ = cfg2ckpt(configs, flag_values['workdir'].value)

    assert os.path.exists(output_dir), f'Checkpoint {output_dir} not found.'
    return output_dir, configs

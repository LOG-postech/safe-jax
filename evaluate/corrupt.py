"""CIFAR-10C common corruption evaluation."""

import numpy as np
from absl import logging
from flax import jax_utils

from train_utils import sync_batch_stats, batch_norm_tuning
from sparsify import projection
from datasets import TFDataLoader, CIFAR10C_CORRUPT_TYPES


def evaluate_corruptions(
    config, state, p_eval_step, p_bn_step,
    train_iter, model_info
):
    """Evaluate model on all CIFAR-10C corruption types."""
    get_metric_item = lambda s: s.metric.reduce().compute().items()
    results = {}

    original_batch_stats = state.batch_stats

    for sp in config.eval_sparsities:
        logging.info(f"Sparsity {sp} eval:")

        masked_params, _ = projection(
            jax_utils.unreplicate(state.params), sp, scope=config.sp_scope
        )

        # Non-BNT
        logging.info("Start non-BNT eval:")
        state = state.replace(
            params=jax_utils.replicate(masked_params),
            batch_stats=original_batch_stats,
            metric=jax_utils.replicate(state.metric.empty())
        )

        corr_accs = []
        for corrupt in CIFAR10C_CORRUPT_TYPES:
            state = state.replace(metric=jax_utils.replicate(state.metric.empty()))
            dataset = f'cifar10_corrupted/{corrupt}_{config.corrupt_intensity}'
            eval_iter = TFDataLoader(dataset, config.batch_size, train=False)

            for batch in eval_iter:
                state = p_eval_step(state, batch)

            for m, v in get_metric_item(state):
                if 'accuracy' in m.lower():
                    results[f'sp_{sp}/{corrupt}_{config.corrupt_intensity}/masked_val_{m}'] = float(v)
                    corr_accs.append(float(v))

        results[f'sp_{sp}/mean_masked_val_accuracy'] = float(np.mean(corr_accs))
        logging.info(f"  Non-BNT mean accuracy: {results[f'sp_{sp}/mean_masked_val_accuracy']:.4f}")

        # BNT
        if model_info['batch_norm']:
            logging.info("Start BNT eval:")
            state = state.replace(
                params=jax_utils.replicate(masked_params),
                batch_stats=original_batch_stats,
                metric=jax_utils.replicate(state.metric.empty())
            )
            state = batch_norm_tuning(
                state, train_iter, p_bn_step,
                bnt_sample_size=config.bnt_sample_size
            )
            state = sync_batch_stats(state)

            corr_accs_bnt = []
            for corrupt in CIFAR10C_CORRUPT_TYPES:
                state = state.replace(metric=jax_utils.replicate(state.metric.empty()))
                dataset = f'cifar10_corrupted/{corrupt}_{config.corrupt_intensity}'
                eval_iter = TFDataLoader(dataset, config.batch_size, train=False)

                for batch in eval_iter:
                    state = p_eval_step(state, batch)

                for m, v in get_metric_item(state):
                    if 'accuracy' in m.lower():
                        results[f'sp_{sp}/{corrupt}_{config.corrupt_intensity}/masked_bnt_val_{m}'] = float(v)
                        corr_accs_bnt.append(float(v))

            results[f'sp_{sp}/mean_masked_bnt_val_accuracy'] = float(np.mean(corr_accs_bnt))
            logging.info(f"  BNT mean accuracy: {results[f'sp_{sp}/mean_masked_bnt_val_accuracy']:.4f}")

    return results

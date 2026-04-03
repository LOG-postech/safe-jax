"""Adversarial PGD evaluation."""

import jax
import jax.numpy as jnp
from absl import logging
from flax import jax_utils

from train_utils import sync_batch_stats, batch_norm_tuning
from sparsify import projection


def evaluate_adversarial(
    config, state, p_eval_step, p_bn_step, p_attack_step,
    train_iter, eval_iter, model_info, tx, attack_steps=10
):
    """Evaluate model under adversarial attack at each sparsity level."""
    get_metric_item = lambda s: s.metric.reduce().compute().items()
    results = {}

    original_batch_stats = state.batch_stats

    for sp in config.eval_sparsities:
        logging.info(f"Sparsity {sp} adversarial eval:")

        masked_params, _ = projection(
            jax_utils.unreplicate(state.params), sp, scope=config.sp_scope
        )

        state = state.replace(
            params=jax_utils.replicate(masked_params),
            batch_stats=original_batch_stats,
            metric=jax_utils.replicate(state.metric.empty())
        )

        if model_info['batch_norm']:
            state = batch_norm_tuning(
                state, train_iter, p_bn_step,
                bnt_sample_size=config.bnt_sample_size
            )
        state = sync_batch_stats(state)

        for i, batch in enumerate(eval_iter):
            logging.info(f"  Attacking batch {i}/{len(eval_iter)}")

            tx_state = tx.init(jnp.zeros(()))
            tx_state = jax_utils.replicate(tx_state)

            sample_perturbation = jnp.zeros_like(batch['sample'], dtype=jnp.float32)
            for _ in range(attack_steps):
                sample_perturbation, tx_state = p_attack_step(
                    state, tx_state, sample_perturbation, batch
                )

            batch['sample'] = jnp.clip(batch['sample'] + sample_perturbation, 0, 1)
            state = p_eval_step(state, batch)

        for m, v in get_metric_item(state):
            logging.info(f"  Adversarial {m}: {v:.4f}")
            results[f'sp_{sp}/adv_bnt_val_{m}'] = float(v)

    return results

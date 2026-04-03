"""Clean CIFAR-10 test evaluation."""

from absl import logging
from flax import jax_utils
import ml_collections

from train_utils import sync_batch_stats, batch_norm_tuning
from sparsify import projection, param_count, weight_sparsity, tree_norm
from jax.tree_util import tree_map


def evaluate(
    config, state, p_eval_step, p_bn_step,
    train_iter, eval_iter,
    model_info, new_metrics=None, compute_bnt=False
):
    """Evaluate the model on the clean validation set."""
    get_metric_item = lambda s: s.metric.reduce().compute().items()
    new_metrics = new_metrics or {}

    eval_state = sync_batch_stats(state) if model_info['batch_norm'] else state
    eval_state = eval_state.replace(metric=jax_utils.replicate(eval_state.metric.empty()))

    for batch in eval_iter:
        eval_state = p_eval_step(eval_state, batch)

    new_metrics.update({f'val_{m}': float(v) for m, v in get_metric_item(eval_state)})

    # masked eval
    for sp in config.eval_sparsities:
        masked_params, _ = projection(
            jax_utils.unreplicate(state.params), sp, scope=config.sp_scope
        )
        eval_state = eval_state.replace(
            params=jax_utils.replicate(masked_params),
            metric=jax_utils.replicate(eval_state.metric.empty())
        )
        for batch in eval_iter:
            eval_state = p_eval_step(eval_state, batch)

        sign_m = lambda m: -1 if m == 'loss' else 1
        new_metrics.update({
            f'sp_{int(sp*100)}%': {
                **{f'masked_val_{m}': float(v) for m, v in get_metric_item(eval_state)},
                **{f'masked_val_{m}_diff': sign_m(m) * (new_metrics[f'val_{m}'] - float(v))
                   for m, v in get_metric_item(eval_state)},
                'proj_dev': float(tree_norm(tree_map(lambda x, z: x - z, state.params, eval_state.params)))
            }
        })

    # BNT eval
    if compute_bnt and model_info['batch_norm']:
        for sp in config.eval_sparsities:
            masked_params, _ = projection(
                jax_utils.unreplicate(state.params), sp, scope=config.sp_scope
            )
            eval_state = eval_state.replace(
                params=jax_utils.replicate(masked_params),
                metric=jax_utils.replicate(eval_state.metric.empty())
            )
            eval_state = batch_norm_tuning(
                eval_state, train_iter, p_bn_step, bnt_sample_size=config.bnt_sample_size
            )
            eval_state = sync_batch_stats(eval_state)

            for batch in eval_iter:
                eval_state = p_eval_step(eval_state, batch)

            sign_m = lambda m: -1 if m == 'loss' else 1
            final_bnt_metrics = {
                **{f'final_masked_bnt_val_{m}': float(v) for m, v in get_metric_item(eval_state)},
                **{f'final_masked_bnt_val_{m}_diff': sign_m(m) * (new_metrics[f'val_{m}'] - float(v))
                   for m, v in get_metric_item(eval_state)}
            }
            new_metrics[f'sp_{int(sp*100)}%'].update(**final_bnt_metrics)

    state = jax_utils.unreplicate(state)
    new_metrics = {
        'param_count': param_count(state.params),
        'weight_sparsity': weight_sparsity(state.params),
        **new_metrics
    }

    logging.info(f'Results: \n{ml_collections.ConfigDict(new_metrics)}')
    return new_metrics

"""Shared eval and batch-norm steps."""

from functools import partial
import jax

@jax.jit
def bn_step(state, batch):
    dropout_key = jax.random.fold_in(key=state.key, data=state.step)
    _, new_state = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['sample'],
        train=True,
        mutable=['batch_stats'],
        rngs={'dropout': dropout_key}
    )
    state = state.replace(batch_stats=new_state['batch_stats'])
    return state


@partial(jax.jit, static_argnums=(2, 3))
def eval_step(state, batch, loss_type, metrics):
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['sample'],
        train=False,
        mutable=False
    )
    metric = state.metric.merge(
        metrics.gather_from_model_output(
            loss=loss_type(logits, batch['target']),
            logits=logits,
            labels=batch['target']
        )
    )
    state = state.replace(metric=metric)
    return state

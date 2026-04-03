"""PGD adversarial attack utilities."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import optax
import chex


class EmptyState(NamedTuple):
    pass


def init_empty_state(params) -> EmptyState:
    del params
    return EmptyState()


def scale_by_sign() -> optax.GradientTransformation:
    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree_map(jnp.sign, updates)
        return updates, state
    return optax.GradientTransformation(init_empty_state, update_fn)


def fgsm(lr):
    return optax.chain(scale_by_sign(), optax.scale(-lr))


class ProjState(NamedTuple):
    count: chex.Array


def get_proj_fn(epsilon, norm='linf'):
    def project_ball(x):
        if norm == 'linf':
            x = jnp.clip(x, -epsilon, epsilon)
        elif norm == 'l2':
            norm_x = jnp.maximum(jnp.linalg.norm(x), epsilon)
            x = x * (epsilon / norm_x)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        return x
    return project_ball


def pgd_attack(base_tx, epsilon, norm='linf') -> optax.GradientTransformation:
    proj = get_proj_fn(epsilon=epsilon, norm=norm)

    def init_fn(params):
        tx_state = base_tx.init(params)
        return ProjState(count=jnp.zeros([], jnp.int32)), tx_state

    def update_fn(updates, state, params):
        state, tx_state = state
        updates, tx_state = base_tx.update(updates, tx_state, params)
        params_ = tree_map(lambda p, u: p + u, params, updates)
        proj_params = proj(params_)
        updates = tree_map(lambda p, pp: pp - p, params, proj_params)
        count_inc = optax.safe_int32_increment(state.count)
        return updates, (ProjState(count=count_inc), tx_state)

    return optax.GradientTransformation(init_fn, update_fn)


@partial(jax.jit, static_argnums=(4, 5))
def attack_step(state, tx_state, sample_perturbation, batch, loss_type, tx):
    def loss_fn(sample_perturbation):
        logits = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            batch['sample'] + sample_perturbation,
            train=False,
            mutable=False
        )
        loss = loss_type(logits, batch['target'])
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(sample_perturbation)
    updates, tx_state = tx.update(grads, tx_state, sample_perturbation)
    sample_perturbation = sample_perturbation + updates
    return sample_perturbation, tx_state

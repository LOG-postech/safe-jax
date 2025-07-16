import copy
from typing import NamedTuple, Any

import jax, optax, chex
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map

from .utils import projection, only_weights

KeyArray = Any

class AdmmState(NamedTuple):
    """State for the pAdmm algorithm."""
    count: chex.Array  # shape=(), dtype=jnp.int32.
    z: optax.Updates
    u: optax.Updates

def admm(
    lmda: float, # penalty parameter
    primal_tx: optax.GradientTransformation, # primal optim
    target_sparsity: float,
    sp_scope: str = 'global',
    dual_update_interval: int = 1, # :: count -> bool
) -> optax.GradientTransformation:
    r"""Alternating Direction Method of Multipliers (ADMM) for sparsification.
    Args:
        lmda: penalty parameter.
        primal_tx: Optimizer for primal update (default: optax.sgd)
        target_sparsity: target sparsity level.
        sp_scope: Sparsity scope (global, layerwise).
        dual_update_interval: Update interval for z and u. Returns True in some iteration.

    Returns:
        The corresponding `GradientTransformation`.
    """
    
    def init_fn(params):
        w = only_weights(params)
        z = projection(copy.deepcopy(w), target_sparsity, scope=sp_scope)[0] # projected primal varible (masked)
        u = jax.tree_util.tree_map(jnp.zeros_like, w)  # dual variable

        ptx_state = primal_tx.init(params)
        return AdmmState(count=jnp.zeros([], jnp.int32), z=z, u=u), ptx_state

    def update_fn(updates, state, params):
        admm_state, ptx_state = state
        w, z, u = only_weights(params), admm_state.z, admm_state.u

        # update primal variable `w`
        updates = tree_map(lambda g, w, z, u: g+lmda*(w-z+u), updates, w, z, u)
        updates, ptx_state = primal_tx.update(updates, ptx_state, params)
        
        # update primal variable `z` or dual variable `u` upon schedule
        indicator = admm_state.count%dual_update_interval==dual_update_interval-1

        _wu = tree_map(lambda w, up, u: (w+up)+u, w, only_weights(updates), u)
        z = lax.cond(indicator, lambda: projection(_wu, target_sparsity, scope=sp_scope)[0], lambda: z)
        u = lax.cond(indicator, lambda: tree_map(lambda wu, z: wu-z, _wu, z), lambda: u)
            
        count_inc = optax.safe_int32_increment(admm_state.count)
        return updates, (AdmmState(count=count_inc, z=z, u=u), ptx_state)

    return optax.GradientTransformation(init_fn, update_fn)



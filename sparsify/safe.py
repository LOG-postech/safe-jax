import copy
from typing import NamedTuple, Any, Literal, Optional

import jax, optax, chex
from jax import lax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from .utils import projection, only_weights, BaseTrainState

KeyArray = Any

# TrainState.
# -----------------------------------------------------------------------------

class SAFETrainState(BaseTrainState):

    def apply_gradients(self, *, grads, loss_fn, **kwargs):
        """
        Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.
        """
        grads_with_opt = grads
        params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt, loss_fn
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)
        
        new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

# SAFE Optimizer.
# -----------------------------------------------------------------------------

class SAFEState(NamedTuple):
    """State for the pAdmm algorithm."""
    count: chex.Array  # shape=(), dtype=jnp.int32.
    z: optax.Updates
    u: optax.Updates

def safe(
    lmda: float, # penalty parameter
    primal_tx: optax.GradientTransformation, # primal optim
    target_sparsity: float,
    sp_scope: str = 'global',
    dual_update_interval: int = 1, # :: count -> bool
    total_steps: Optional[int] = None,
    lmda_schedule: Literal['constant', 'cosine', 'linear'] = 'cosine',
    rho: float = 0.1, # perturbation bound radius
) -> optax.GradientTransformation:
    r"""Sparsification via ADMM with Flatness Enforcement (SAFE). 
    Args:
        lmda: penalty parameter.
        primal_tx: Optimizer for primal update
        target_sparsity: target sparsity level.
        sp_scope: Sparsity scope (global, layerwise) (default: global).
        dual_update_interval: Update interval for z and u (default: 1).
        total_steps: total number of steps for cosine and linear schedule (default: None).
        lmda_schedule: schedule for lmda (constant, cosine, linear) (default: cosine)
        rho: perturbation bound radius (default: 0.1)

    Returns:
        The corresponding `GradientTransformation`.
    """
    assert (rho >= 0.0), 'rho value should be a positive value.'
    
    # lmda Schedules. 
    if lmda_schedule=='constant':
        lmda_fn = lambda _: lmda
    elif lmda_schedule=='cosine':
        assert total_steps is not None, 'total_steps must be provided for cosine schedule'
        lmda_fn = lambda count: lmda*0.5*(1-jnp.cos(jnp.pi*count/total_steps))
    elif lmda_schedule=='linear':
        assert total_steps is not None, 'total_steps must be provided for linear schedule'
        lmda_fn = lambda count: lmda*count/total_steps
    
    def init_fn(params):
        w = only_weights(params)
        z = projection(copy.deepcopy(w), target_sparsity, scope=sp_scope)[0] # projected primal varible (masked)
        u = jax.tree_util.tree_map(jnp.zeros_like, w)  # dual variable

        ptx_state = primal_tx.init(params)
        return SAFEState(count=jnp.zeros([], jnp.int32), z=z, u=u), ptx_state
    
    def update_fn(updates, state, params, loss_fn):
        admm_state, ptx_state = state
        w, z, u = only_weights(params), admm_state.z, admm_state.u
        
        # ascent step
        updates = jax.lax.pmean(updates, 'batch')
        gn = norm(ravel_pytree(updates)[0], ord=2) + 1e-12
        ps = tree_map(lambda p, g: p+rho*(g/gn), params, updates)
        _, updates = jax.value_and_grad(loss_fn, has_aux=True)(ps)

        # update primal variable `w`
        updates = tree_map(lambda g, w, z, u: g+lmda_fn(admm_state.count)*(w-z+u), updates, w, z, u)
        updates, ptx_state = primal_tx.update(updates, ptx_state, params)
        
        # update primal variable `z` or dual variable `u` upon schedule
        indicator = admm_state.count%dual_update_interval==dual_update_interval-1

        _wu = tree_map(lambda w, up, u: (w+up)+u, w, only_weights(updates), u)
        z = lax.cond(indicator, lambda: projection(_wu, target_sparsity, scope=sp_scope)[0], lambda: z)
        u = lax.cond(indicator, lambda: tree_map(lambda wu, z: wu-z, _wu, z), lambda: u)
            
        count_inc = optax.safe_int32_increment(admm_state.count)
        return updates, (SAFEState(count=count_inc, z=z, u=u), ptx_state)

    return optax.GradientTransformation(init_fn, update_fn)




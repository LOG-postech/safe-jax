
from typing import Sequence, NamedTuple, Union, Literal

import jax.numpy as jnp
from jax.tree_util import tree_map
import optax, chex

from .utils import projection, sparsity2count, BaseTrainState


# TrainState.
# -----------------------------------------------------------------------------

class SparsifierTrainState(BaseTrainState):

    def apply_gradients(self, *, grads, target_count, **kwargs):
        """
        Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.
        """
        grads_with_opt = grads
        params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt, target_count
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)
        
        new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )
        
    @classmethod
    def create(cls, *, apply_fn, params, target_count, tx, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        opt_state = tx.init(params, target_count)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


# PGD, GMP Optimizer.
# -----------------------------------------------------------------------------

class IHTState(NamedTuple):
    """State for the pAdmm algorithm."""
    count: chex.Array  # shape=(), dtype=jnp.int32.

def iht(base_tx, sp_scope: str) -> optax.GradientTransformation:
    r"""Iterative hard threshold.
    Args:
        base_tx: Optimizer for primal update (default: optax.sgd)
        sp_scope: Sparsity scope (global, layerwise)

    Returns:
        The corresponding `GradientTransformation`.
    """

    def init_fn(params, grads, target_count):
        del grads
        tx_state = base_tx.init(params)
        return IHTState(count=jnp.zeros([], jnp.int32)), tx_state

    def update_fn(updates, state, params, target_count):
        state, tx_state = state
        
        updates, tx_state = base_tx.update(updates, tx_state, params)
        params_ = tree_map(lambda p, u: p+u, params, updates)
        proj_params = projection(params_, target_count, scope=sp_scope, by_count=True)[0]
        updates = tree_map(lambda p, pp: pp-p, params, proj_params)
        
        count_inc = optax.safe_int32_increment(state.count)
        return updates, (IHTState(count=count_inc), tx_state)

    return optax.GradientTransformation(init_fn, update_fn)


class GMPState(NamedTuple):
    """State for masking algorithm."""
    masks: optax.Updates
    count: chex.Array  # shape=(), dtype=jnp.int32.

def gmp(base_tx, sp_scope: str) -> optax.GradientTransformation:
    r"""Gradual magnitude pruning.
    Args:
        base_tx: Optimizer for primal update.
        sp_scope: Sparsity scope (global, layerwise)

    Returns:
        The corresponding `GradientTransformation`.
    """
    def init_fn(params, target_count):
        tx_state = base_tx.init(params)
        masks = projection(params, target_count, scope=sp_scope, by_count=True)[1]
        return GMPState(masks=masks, count=jnp.zeros([], jnp.int32)), tx_state

    def update_fn(updates, state, params, target_count):
        state, tx_state = state
        
        updates, tx_state = base_tx.update(updates, tx_state, params)
        proj_params = tree_map(lambda p, u, m: (p+u)*m, params, updates, state.masks)
        proj_params, new_masks = projection(proj_params, target_count, scope=sp_scope, by_count=True)
        updates = tree_map(lambda p, pp: pp-p, params, proj_params)
        
        count_inc = optax.safe_int32_increment(state.count)
        return updates, (GMPState(masks=new_masks, count=count_inc), tx_state)

    return optax.GradientTransformation(init_fn, update_fn)


# Sparsity Schedules. 
# -----------------------------------------------------------------------------

def sp_schedules(
    target_sp: Union[float, Sequence[float]], 
    steps: int, 
    weight_count: Union[int, Sequence[int]],
    schedule_type: Literal['constant', 'cosine', 'linear', 'cubic'],
    init_sp=0.0
):
    r"""Sparsity schedules."""
    def seq_wrap(fn):
        def func(*args):
            if isinstance(target_sp, Sequence):
                return sparsity2count(weight_count, 
                                      tuple([fn(*args, sp=sp) for sp in target_sp]), 
                                      scope='layerwise')
            elif isinstance(target_sp, float):
                return sparsity2count(weight_count, 
                                      fn(*args, sp=target_sp),
                                      scope='global')
        return func
    
    if schedule_type=='constant':
        return lambda _: target_sp
    
    if schedule_type=='cosine':
        @seq_wrap
        def cosine(count, sp):
            count = min(count, steps)
            sch_sp = 0.5*(1-jnp.cos(jnp.pi*count/steps))
            return float(sp*sch_sp + init_sp*(1-sch_sp))
        
        return cosine
    
    if schedule_type=='linear':
        @seq_wrap
        def linear(count, sp):
            count = min(count, steps)
            return float(sp*count/steps + init_sp*(1-count/steps))
        
        return linear
    
    if schedule_type=='cubic':
        # Reference: To prune, or not to prune: exploring the efficacy of pruning for model compression, Zhu et al., ICLR workshop 2018
        @seq_wrap
        def cubic(count, sp):
            count = min(count, steps)
            return float(sp + (init_sp-sp)*(1-count/steps)**3)
        
        return cubic
    
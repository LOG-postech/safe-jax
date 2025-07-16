
from typing import Sequence, Any

from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
from flax.training import train_state
from clu import metrics

KeyArray = Any

# Sparsification utils.
# -----------------------------------------------------------------------------

def projection(params, target_sparsity, scope='global', by_count=False):
    scores = tree_map(lambda w: lax.abs(w), params)
    masks = compute_mask(scores, target_sparsity, scope=scope, by_count=by_count)
    projected_params = tree_map(lambda p, m: p*m, params, masks)
    return projected_params, masks


def only_weights(params):
    # make all bias terms 0
    layers, trdef = tree_flatten(params, lambda tr: 'bias' in tr)

    for wb in layers:
        if 'bias' in wb:
            wb['bias'] = jnp.zeros_like((wb['bias']))
    w = tree_unflatten(trdef, layers)
    return w


def compute_mask(scores, target_sparsity, scope='global', by_count=False):
  """Generate pruning mask based on given scores, eliminate ``target_count``-number of weights.
    For not raising ConcretizationTypeError from jax.jit"""
  
  assert scope in {'global', 'layerwise'}
  if not by_count:
    assert 0<=target_sparsity<=1, f'target_sparsity should be in [0, 1], got {target_sparsity}'
  
  # mask computing function given score and threshold
  def _mask_dict(sc, thr):
    if 'kernel' not in sc: return jnp.full(sc.shape, True)
    
    mask_dict = {'kernel': sc['kernel']>=thr}
    if 'bias' in sc:
      mask_dict['bias'] = jnp.full(sc['bias'].shape, True)

    return mask_dict

  # flatten scores pytree, leaf being dict containing 'kernel' instead of jnp.arrays   
  flat_tr, trdef = tree_flatten(scores, lambda tr: 'kernel' in tr)

  # sort by scores, use only kernel/weight parameters
  if scope=='global':
    flat_sc, _  = ravel_pytree([sc['kernel'] for sc in flat_tr if 'kernel' in sc])
    thr = jnp.sort(flat_sc)[target_sparsity if by_count else int(target_sparsity*len(flat_sc))] # compute global threshold

    _mask_dict_g = lambda sc: _mask_dict(sc, thr)
    flat_mask = [*map(_mask_dict_g, flat_tr)] # compute mask

  elif scope=='layerwise':
    sort_scs = [(jnp.sort(sc['kernel'].ravel()) if 'kernel' in sc else None) for sc in flat_tr]
    # compute layer thresholds
    assert isinstance(target_sparsity, Sequence), f'target_sparsity should be of type Sequence, got {type(target_sparsity)}'
    i, cntn = 0, []
    assert (kern_sclen:=len([sc for sc in sort_scs if sc!=None]))==len(target_sparsity), f'Layerwise parameter target_count sequence length (target_count={len(target_count)}) should be equal to the number of kernel layers ({kern_sclen})'
    for s in sort_scs:
      if s is not None:
        s = target_sparsity[i] if by_count else int(target_sparsity[i]*len(s))
        i += 1
      cntn.append(s)
    thrs = [sc if sc==None else sc[c] for sc, c in zip(sort_scs, cntn)] # different sp over layers
      
    flat_mask = [*map(_mask_dict, flat_tr, thrs)] # compute mask
      
  mask = tree_unflatten(trdef, flat_mask)

  return mask
        
        
def weight_count(params, layerwise=False):
  flat_tr, _ = tree_flatten(params, lambda tr: 'kernel' in tr)
  layers = [m['kernel'] for m in flat_tr if 'kernel' in m]
  if layerwise:
    return [l.size for l in layers]
  else:
    flat_w, _ = ravel_pytree(layers)
    return len(flat_w)


def sparsity2count(total_count, sp, scope='global'):
  '''Computes target parameter count (int) given target sparsity (float). Use this outside ``jax.jit``.'''
  assert scope in {'global', 'layerwise'}
  if isinstance(sp, float):
    assert 0 <= sp <= 1, f'sp should be in [0, 1], got {sp}'
  elif isinstance(sp, Sequence):
    assert all(0 <= s <= 1 for s in sp), f'All sp should be in [0, 1], got {sp}'
  
  if scope=='global':
    assert isinstance(sp, float), f'sp should be of type: float, got {type(sp)}'
    assert isinstance(total_count, int), f'weigth_count should be of type: int, got {type(total_count)}'
    return int(sp*total_count)

  elif scope=='layerwise':
    assert isinstance(total_count, Sequence), f'weigth_count should be of type: Sequence[int], got {type(total_count)}'
    if isinstance(sp, float):
      return [int(sp*c) for c in total_count]
    elif isinstance(sp, Sequence):
      assert len(total_count)== len(sp), f'Layerwise sparsity level sequence sp (len={len(sp)}) should be equal to the number of layers ({len(total_count)})'
      return [int(s*c) for s, c in zip(sp, total_count)]


def weight_sparsity(params, scope='global'):
  assert scope in {'global', 'local'}
  
  if scope=='global':
    flat_tr, _ = tree_flatten(params, lambda tr: 'kernel' in tr)
    flat_w, _ = ravel_pytree([m['kernel'] for m in flat_tr if 'kernel' in m])
    return (flat_w == 0).sum().item() / len(flat_w)
  
  elif scope=='local':
    flat_tr, _ = tree_flatten(params, lambda tr: 'kernel' in tr)
    flat_ws = [m['kernel'].ravel() for m in flat_tr if 'kernel' in m]
    return [(flat_w == 0).sum().item() / len(flat_w) for flat_w in flat_ws]
  
  
def param_count(params, only_weights=True):
  if only_weights:
    flat_tr, _ = tree_flatten(params, lambda tr: 'kernel' in tr)
    flat_w, _ = ravel_pytree([m['kernel'] for m in flat_tr if 'kernel' in m])
    return len(flat_w)
  else:
    return len(ravel_pytree(params)[0])
  
  
def tree_norm(params, get_only_weights=True):
    if get_only_weights:
        w = only_weights(params)
    return jnp.linalg.norm(ravel_pytree(w)[0])

  
# Base train state.
# -----------------------------------------------------------------------------

class BaseTrainState(train_state.TrainState):
    metric: metrics.Collection
    batch_stats: Any
    key: KeyArray

  
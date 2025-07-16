
import os, shutil, copy
from typing import Any, List

from absl import logging
import numpy as np
import tensorflow as tf

import jax, flax, optax
from jax import random, lax
import jax.numpy as jnp
from flax.training import checkpoints
from clu import metrics
import ml_collections

from models import initialized, get_model
from datasets import TFDataLoader, get_cifar10_lable_noise_datasets
from sparsify import (safe, admm, iht, gmp, SAFETrainState, SparsifierTrainState, BaseTrainState,
                      sparsity2count, weight_count, sp_schedules)
KeyArray = Any

# Configurator for dataloader, model, etc.
# -----------------------------------------------------------------------------

def configure_train(config: ml_collections.ConfigDict,
                    workdir: str,
                    resume_checkpoint=False):
    
    logging.info(config)
    
    ############################# Random Seed #############################
    
    rng = random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    
    ############################# Device settings #############################
    
    if (config.batch_size or 1) % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    
    ############################# Prepare Dataset #############################
    
    assert 0<=config.label_noise_ratio<=1, 'Label noise ratio should be in [0, 1]'
    if config.label_noise_ratio==0:
      train_iter   = TFDataLoader(config.dataset, config.batch_size, train=True)
      eval_iter    = TFDataLoader(config.dataset, config.batch_size, train=False)
    else:
        assert config.dataset=='cifar10', 'Label noise is only supported for CIFAR10 dataset'
        train_iter, eval_iter = get_cifar10_lable_noise_datasets(config.batch_size, config.label_noise_ratio, config.seed)
    
    ############################# Prepare model, loss, and metric #############################
    
    num_classes = train_iter.data_info['num_classes'] or 1
    model, model_info = get_model(config.model, config.half_precision, num_classes=num_classes)
    loss_type, Metrics = get_loss_and_metric(train_iter.data_info['task'])
        
    ############################# Prepare / Restore State #############################
    
    rng, init_rng = jax.random.split(rng)
    state, sp_schedule = create_train_state(init_rng, 
                                            model, model_info, train_iter.data_info, 
                                            Metrics, len(train_iter), config)
    if resume_checkpoint:
        checkpoints.restore_checkpoint(workdir, state)
        
    return state, (loss_type, Metrics, train_iter, eval_iter, sp_schedule, model_info)
    

# Optimizer and learning rate schedules.
# -----------------------------------------------------------------------------

def get_learning_rate_fn(steps_per_epoch: int, **kwargs):
    """Create learning rate schedule."""

    if kwargs['lr_schedule'] == 'step':
        step_ratio = lambda ratio: int(kwargs['num_epochs'] * ratio * steps_per_epoch)
        return optax.piecewise_constant_schedule(
            kwargs['lr'], 
            boundaries_and_scales={
                step_ratio(0.25): 0.1, 
                step_ratio(0.5):  0.1, 
                step_ratio(0.75): 0.1
            })
        
    elif kwargs['lr_schedule'] == 'cosine': 
        # cosine learning rate scheduler (kwargs['ration'] in 'efficient sam' and 'cram' paper)
        if kwargs['warmup_epochs']:
            return optax.warmup_cosine_decay_schedule(
                init_value = 0,
                peak_value = kwargs['lr'],
                warmup_steps = kwargs['warmup_epochs'] * steps_per_epoch,
                decay_steps = (kwargs['num_epochs']-kwargs['warmup_epochs']) * steps_per_epoch,
                end_value = 0,
            )
        else:
            return optax.cosine_decay_schedule(kwargs['lr'], kwargs['num_epochs'] * steps_per_epoch)
        
    elif kwargs['lr_schedule'] == 'constant':
        return optax.constant_schedule(kwargs['lr'])


def get_optimizer(optimizer: str, steps_per_epoch: int, **kwargs):
    optimizer = optimizer.lower()
    learning_rate = get_learning_rate_fn(steps_per_epoch=steps_per_epoch, **kwargs)
    
    if optimizer=='sgd':
        return optax.chain(
                optax.add_decayed_weights(kwargs['wd']),
                optax.sgd(
                learning_rate =  learning_rate,
                momentum =       kwargs['momentum'],
                nesterov =       False,
            )
        )
    elif optimizer=='adam':
        return optax.adamw(
            learning_rate =  learning_rate,
            b1 =             kwargs['momentum'],
            weight_decay =   kwargs['wd'],
        )
        

# Sparsifiers and trainstates.
# -----------------------------------------------------------------------------

    
def create_train_state(rng, 
                       model, model_info, data_info, 
                       Metrics, steps_per_epoch, config):
    """Create initial training state."""
    
    init_rng, dropout_rng = jax.random.split(rng)

    params, batch_stats = initialized(init_rng, 
                                      data_info['input_shape'], 
                                      model, 
                                      batch_stats=model_info['batch_norm'], 
                                      has_dropout=model_info['dropout'])
    
    tx, TrainState = get_sparsifier_and_trainstate(steps_per_epoch=steps_per_epoch, **config)

    if config.sparsifier in {'gmp', 'iht'}:
        w_count = weight_count(params, layerwise=(config.sp_scope=='layerwise'))
        sp_schedule = sp_schedules(config.sp, steps_per_epoch*config.num_epochs, w_count, schedule_type=config.sp_schedule)
        
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            target_count=sp_schedule(0),
            tx=tx,
            batch_stats=batch_stats,
            key=dropout_rng,
            metric=Metrics.empty())
    else:
        sp_schedule=None
        
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
            key=dropout_rng,
            metric=Metrics.empty())

    return state, sp_schedule


def get_sparsifier_and_trainstate(sparsifier: str, 
                                  optimizer: str, 
                                  steps_per_epoch: int, 
                                  **kwargs):
    sparsifier = sparsifier.lower()
    tx = get_optimizer(optimizer, steps_per_epoch, **kwargs)

    if sparsifier=='safe':
        return safe(
            lmda = kwargs['lambda'],
            primal_tx = tx,
            target_sparsity = kwargs['sp'],
            sp_scope = kwargs['sp_scope'],
            dual_update_interval = kwargs['dual_update_interval'],
            lmda_schedule = kwargs['lambda_schedule'],
            total_steps = kwargs['num_epochs'] * steps_per_epoch,
            rho = kwargs['rho'],
        ), SAFETrainState
        
    elif sparsifier=='admm':
        return admm(
            lmda = kwargs['lambda'],
            primal_tx = tx,
            target_sparsity = kwargs['sp'],
            sp_scope = kwargs['sp_scope'],
            dual_update_interval = kwargs['dual_update_interval']
        ), BaseTrainState
        
    elif sparsifier=='iht':
        return iht(
            base_tx = tx,
            sp_scope = kwargs['sp_scope'],
        ), SparsifierTrainState

    elif sparsifier=='gmp':
        return gmp(
            base_tx = tx,
            sp_scope = kwargs['sp_scope'],
        ), SparsifierTrainState
    
    elif sparsifier=='none':
        return tx, BaseTrainState


# Task specific (loss_fns and metrics).
# -----------------------------------------------------------------------------

def cross_entropy_loss(logits, labels):
    xentropy = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(xentropy)

def mse_loss(preds, targets):
    mse = 2*optax.l2_loss(predictions=preds, targets=targets)
    return jnp.mean(mse)

def get_loss_and_metric(task):
    """Returns loss and metric corresponding to task"""
    assert task in (t:={'regression', 'classification'}), f'task should be one of {t}'
    
    if task=='regression':
        # Loss
        loss_type = mse_loss
        
        # Metric
        @flax.struct.dataclass
        class Metrics(metrics.Collection):
            loss: metrics.Average.from_output("loss")
    
    elif task=='classification':
        # Loss
        loss_type = cross_entropy_loss
        
        # Metric
        @flax.struct.dataclass
        class Metrics(metrics.Collection):
            loss: metrics.Average.from_output("loss")
            accuracy: metrics.Accuracy
        
    return loss_type, Metrics


# Batch Norm Statistics.
# -----------------------------------------------------------------------------

def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    if state.batch_stats == {}:
        return state
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def batch_norm_tuning(state, data_iter, p_bn_step, bnt_sample_size=None):
    zero_batch_stats = jax.tree_map(lambda p: jnp.zeros_like(p), state.batch_stats)
    state = state.replace(batch_stats=zero_batch_stats)
    for it, batch in enumerate(data_iter):
        if bnt_sample_size is not None and it*data_iter.n_data/len(data_iter)>=bnt_sample_size: break
        state = p_bn_step(state, batch)
    return state


# Checkpoint path string.
# -----------------------------------------------------------------------------
def cfg2ckpt(config, workdir, **custom_cnfg)->List[str]:

    # set custom config
    if custom_cnfg:
      config = copy.deepcopy(config)
      for k, v in custom_cnfg.items():
        config[k] = v
    
    # Generate workdir given keys
    sfxgen = lambda lst: [f'{l}_{str(config[l]).replace("/", "-")}' for l in lst]
    
    # Generate directory    
    suffix_list = [
        'dataset', 'optimizer', 'model', 'num_epochs',
        'lr', 'wd', 'momentum', 'label_noise_ratio'
    ]
    if config.sparsifier in {'safe', 'admm', 'gmp', 'iht'}:
        suffix_list += ['sparsifier', 'sp', 'sp_scope']
    if config.sparsifier in {'gmp', 'iht'}:
        suffix_list.append('sp_schedule')
    if config.sparsifier in {'safe', 'admm'}:
        suffix_list.append('lambda')
    if config.sparsifier=='safe':
        suffix_list += ['lambda_schedule', 'rho']
    suffix_list.append('seed')

    output_dirs = [os.path.join(os.getcwd(), workdir.replace('./', ''), *sfxgen(suffix_list))]
        
    return output_dirs


# Other utilities.
# -----------------------------------------------------------------------------

def create_dir(dir):
  if os.path.exists(dir):
    for f in os.listdir(dir):
      if os.path.isdir(os.path.join(dir, f)):
        shutil.rmtree(os.path.join(dir, f))
      else:
        os.remove(os.path.join(dir, f))
  else:
    os.makedirs(dir)

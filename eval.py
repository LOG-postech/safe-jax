import os
from typing import Any
from functools import partial

from absl import app, flags, logging
import tensorflow as tf

import jax
from jax.tree_util import tree_map
from flax import jax_utils
from clu import platform
import ml_collections

from train_utils import configure_train, sync_batch_stats, batch_norm_tuning, cfg2ckpt
from sparsify import projection, param_count,  weight_sparsity, tree_norm

KeyArray = Any

def configure_flags(): 
  flags.DEFINE_string('workdir', './logdir', "Workding directory to save model and tensorboard data")
  flags.DEFINE_bool('half_precision', False, "Whether to use half precision.")

  flags.DEFINE_string('model', 'ResNet20', "Model to sparsify (ResNet{20, 32, 44, 56}, VGG{11, 13, 16, 19}(-bn), LeNet300-100).")
  flags.DEFINE_string('dataset', 'cifar10', "Dataset (mnist, cifar{10, 100}).")
  flags.DEFINE_integer('num_epochs', 200, "Number of epochs to train.")
  flags.DEFINE_integer('batch_size', 128, "Mini-batch size.")
  flags.DEFINE_integer('seed', 1, "Seed for controlling randomness.")

  flags.DEFINE_string('optimizer', 'sgd', "Optimizer/sparsifier to use (sgd, adam).")
  flags.DEFINE_float('lr', 0.1, "Learning rate.")
  flags.DEFINE_string('lr_schedule', 'cosine', "Learning rate schedule (constant, linear, cosine).")
  flags.DEFINE_integer('warmup_epochs', 5, "Number of warm-up epochs.")
  flags.DEFINE_float('wd', 0.0001, "Weight decay parameter.")
  flags.DEFINE_float('momentum', 0.9, "Momentum parameter.")

  flags.DEFINE_string('sparsifier', 'safe', "Sparsifier to use (safe, admm, gmp, iht, none).")
  flags.DEFINE_float('sp', 0.95, "Target sparsity.")
  flags.DEFINE_string('sp_scope', 'global', "Sparsity scope (global, layerwise).")
  flags.DEFINE_integer('bnt_sample_size', 10000, "Calibration sample size for batch-norm tuning.")

  flags.DEFINE_float('lambda', 0.0001, "Lambda parameter for SAFE and ADMM.")
  flags.DEFINE_string('lambda_schedule', 'cosine', "Lambda schedule for SAFE (constant, linear, cosine).")
  flags.DEFINE_integer('dual_update_interval', 32, "Update interval for z, u in SAFE and ADMM.")
  flags.DEFINE_string('sp_schedule', 'cubic', "Gradual sparsity schedule for GMP and IHT (constant, linear, cosine, cubic).")
  flags.DEFINE_float('rho', 0.1, "Rho parameter for SAFE")

  flags.DEFINE_float('label_noise_ratio', 0.0, "Label noise ratio for cifar10 within [0.0, 1.0].")


# Batch_norm and Eval steps.
# -----------------------------------------------------------------------------

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

@partial(jax.jit, static_argnums = (2, 3))
def eval_step(state, batch, loss_type, metrics):        
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats}, 
        batch['sample'], 
        train=False, 
        mutable=False
    )
    # update metric
    metric = state.metric.merge(
        metrics.gather_from_model_output(
            loss=loss_type(logits, batch['target']), 
            logits=logits,
            labels=batch['target']
        )
    )
    state = state.replace(metric=metric)
    return state


# Train and eval pipelines
# -----------------------------------------------------------------------------


def evaluate(
    config: ml_collections.ConfigDict,
    state,
    p_eval_step, p_bn_step,
    train_iter, eval_iter,
    model_info, new_metrics=None, compute_bnt=False
):
    """Evaluate the model on the validation set."""
    
    get_metric_item = lambda s: s.metric.reduce().compute().items()

    new_metrics = new_metrics or {}
    #####------------------------ eval ------------------------#####
    
    eval_state = sync_batch_stats(state) if model_info['batch_norm'] else state
    eval_state = eval_state.replace(metric=jax_utils.replicate(eval_state.metric.empty()))
    
    # record eval performance
    for batch in eval_iter:
        eval_state = p_eval_step(eval_state, batch)
    
    new_metrics.update({f'val_{m}': float(v) for m, v in get_metric_item(eval_state)})

        
    #####------------------------ masked eval ------------------------#####
    # record masked eval performance and pruning statistics
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
        
        sign_m = lambda m: -1 if m=='loss' else 1
        new_metrics.update({
            f'sp_{int(sp*100)}%': {
                **{f'masked_val_{m}': float(v) for m, v in get_metric_item(eval_state)},
                **{f'masked_val_{m}_diff': sign_m(m) * (new_metrics[f'val_{m}'] - float(v))
                   for m, v in get_metric_item(eval_state)},
                'proj_dev': float(tree_norm(tree_map(lambda x, z: x-z, state.params, eval_state.params)))
            }
        })

    # batch-norm-retuned performance
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
            
            sign_m = lambda m: -1 if m=='loss' else 1
            final_bnt_metrics = {
                **{f'final_masked_bnt_val_{m}': float(v) for m, v in get_metric_item(eval_state)},
                **{f'final_masked_bnt_val_{m}_diff': sign_m(m) * (new_metrics[f'val_{m}'] - float(v)) 
                   for m, v in get_metric_item(eval_state)}
            }
            
            new_metrics[f'sp_{int(sp*100)}%'].update(**final_bnt_metrics)

    state = jax_utils.unreplicate(state)
    
    new_metrics  = {
        'param_count':     param_count(state.params),
        'weight_sparsity': weight_sparsity(state.params),
        **new_metrics
    }

    logging.info(f'Results: \n{ml_collections.ConfigDict(new_metrics)}')
    
    return new_metrics
  

FLAGS = flags.FLAGS

def main(argv):
  configs = ml_collections.ConfigDict(FLAGS.flag_values_dict())
  configs['eval_sparsities'] = [configs['sp']]
  
  if len(argv)>2:
    raise app.UsageError('Too many command-line arguments.')
  
  logging.info(configs.to_dict())
  
  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info(f'JAX local devices ({len(jax.local_devices())}): {jax.local_devices()}')

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f'process_index: {jax.process_index()}, '
      f'process_count: {jax.process_count()}'
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY,
      FLAGS.workdir, 'workdir'
  )

  # get checkpoint
  output_dir, *_ = cfg2ckpt(configs, FLAGS.workdir)
  assert os.path.exists(output_dir), f'Checkpoint {output_dir} not found.'

  # get state, replicate over devices, and configure pmap functions
  state, (
      loss_type, Metrics, train_iter, eval_iter, _, model_info
  ) = configure_train(configs, output_dir, resume_checkpoint=True)  
  p_eval_step = jax.pmap(
      partial(eval_step, loss_type=loss_type, metrics=Metrics),
      axis_name='batch'
  )
  p_bn_step = jax.pmap(bn_step, axis_name='batch')
  state = jax_utils.replicate(state)
  
  evaluate(
      configs, state, p_eval_step, p_bn_step,
      train_iter, eval_iter, model_info, compute_bnt=True
  )


if __name__=='__main__':
  configure_flags()
  app.run(main)

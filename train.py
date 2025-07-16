import os, time
from typing import Any
from functools import partial

from absl import app, flags, logging
import tensorflow as tf

import jax
from jax import lax
from flax import jax_utils
from flax.training import checkpoints
from clu import platform
import ml_collections

from train_utils import configure_train, sync_batch_stats, cfg2ckpt, create_dir
from eval import evaluate, eval_step, bn_step

KeyArray = Any

def configure_flags():
  flags.DEFINE_string('workdir', './logdir', "Workding directory to save model and tensorboard data")
  flags.DEFINE_bool('resume_training', False, "Resume training from checkpoint.")
  flags.DEFINE_bool('checkpoint_every_epoch', 1, "Epoch interval for checkpointing.")
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


# Train, Batch_norm, and Eval steps.
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnums = (3, 4, 5, 6))
def train_step(state, batch, target_count, loss_type, metrics, train=True, sparsifier='none'):
    """Perform a single training step."""
    
    dropout_key = jax.random.fold_in(key=state.key, data=state.step)
    def loss_fn(params):
        """Loss function used for training"""
        logits, model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['sample'],
            train=train,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_key}
        )
        loss = loss_type(logits, batch['target'])
        return loss, (model_state, logits)
    
    # get loss and gradient
    (loss, (model_state, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # update params and metric
    metric = state.metric.merge(
        metrics.gather_from_model_output(
        loss=loss,
        logits=logits,
        labels=batch['target'])
    )
    
    # sync over parallel process by mean
    grads = lax.pmean(grads, axis_name='batch')
    batch_stats = lax.pmean(model_state['batch_stats'], axis_name='batch')

    if sparsifier=='safe':
        state = state.apply_gradients(
            grads=grads,
            batch_stats=batch_stats,
            metric=metric,
            loss_fn=loss_fn
        )

    elif sparsifier in {'gmp', 'iht'}:
        state = state.apply_gradients(
            grads=grads,
            target_count=target_count,
            batch_stats=batch_stats,
            metric=metric
        )
    else:
        state = state.apply_gradients(
            grads=grads,
            batch_stats=batch_stats,
            metric=metric
        )
        
    return state


# Train and eval pipelines
# -----------------------------------------------------------------------------


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str,
    state,
    loss_type,
    Metrics,
    train_iter,
    eval_iter,
    sp_schedule,
    model_info
):
    '''Training and evaluting model'''
    
    ############################# pmap train_step #############################
    
    p_train_step = jax.pmap(
        partial(
            train_step, loss_type=loss_type, metrics=Metrics, 
            train=not model_info['dropout'], sparsifier=config.sparsifier.lower()
        ), 
        axis_name='batch'
    )
    p_eval_step = jax.pmap(
        partial(eval_step, loss_type=loss_type, metrics=Metrics),
        axis_name='batch'
    )
    p_bn_step = jax.pmap(bn_step, axis_name='batch')
    
    state = jax_utils.replicate(state)
    
    ############################# Start Training #############################
    
    logging.info('Initial compilation, this might take some minutes...')

    best_val_acc = best_epoch = -1
    metric_types = state.metric.reduce().compute().keys()
    get_metric_item = lambda s: s.metric.reduce().compute().items()
    
    # resume from checkpoint
    start_epoch = start_epoch = int(jax_utils.unreplicate(state.step))//len(train_iter)
    
    #####------------------------ first eval ------------------------#####
    
    eval_state = state.replace(metric=jax_utils.replicate(state.metric.empty()))
    for batch in eval_iter:
        eval_state = p_eval_step(eval_state, batch)    
    logging.info({f'val_{m}': float(v) for m, v in get_metric_item(eval_state)})
    
    #####------------------------ training loop ------------------------#####
    
    for epoch in range(start_epoch, config.num_epochs):
        logging.info("Epoch %d / %d " % (epoch + 1, int(config.num_epochs)))
        new_metrics = {}
        
        #####------------------------ Train ------------------------#####
        
        # reset train_metric for next epoch
        state = state.replace(metric=jax_utils.replicate(state.metric.empty())) 
        
        start_time = time.time()
        for step, batch in enumerate(train_iter):
            if step % 100 == 0: cur_time = time.time()
            
            if config.sparsifier in {'gmp', 'iht'}:
                target_count = jax_utils.replicate(sp_schedule(jax_utils.unreplicate(state.step)))
                state = p_train_step(state, batch, target_count)
            else:
                state = p_train_step(state, batch, jax_utils.replicate(0))
            
            if (step + 1) % 100 == 0:
                logging.info(
                    f"Epoch[{epoch+1}] Step [{step+1}/{len(train_iter)}] ({time.time()-cur_time:.3f}): " + \
                    ' '.join([f"{m} {v:.4f}" for m, v in get_metric_item(state)])
                )
                cur_time = time.time()
        cur_time = time.time()
        
        # record train performance
        new_metrics.update({f'train_{m}': float(v) for m, v in get_metric_item(state)})
        
        #####------------------------ eval ------------------------#####
        
        new_metrics = evaluate(
            config, state, p_eval_step, p_bn_step, 
            train_iter, eval_iter, model_info, new_metrics,
            compute_bnt=((epoch + 1) == int(config.num_epochs))
        )
        
        #####------------------------ record metrics ------------------------#####
        
        # record best epoch
        if new_metrics['val_accuracy'] > best_val_acc:
            best_epoch, best_val_acc = epoch, new_metrics['val_accuracy']
        
        target_sp = max(config.eval_sparsities) if ('sp' not in config or config.sp==0) else config.sp
        logging.info(
            f"Train: " + ' '.join([f'{k} {new_metrics[f"train_{k}"]:.4f}' for k in metric_types]) + ';' +\
            f" Val: " + ' '.join([f'{k} {new_metrics[f"val_{k}"]:.4f}' for k in metric_types]) + ';' +\
            (f" Masked Val ({target_sp}): " + ' '.join([f'{k} {new_metrics[f"sp_{int(target_sp*100)}%"][f"masked_val_{k}"]:.4f}' for k in metric_types])) +\
            f"(took {cur_time - start_time:.2f} seconds) \n"
        )

        #####------------------------ save checkpoint ------------------------#####
        
        if (epoch + 1) % config.checkpoint_every_epoch == 0 or (epoch + 1) == int(config.num_epochs):
            state = sync_batch_stats(state)
            state = jax.device_get(state)
            checkpoints.save_checkpoint(workdir, state, int(state.step), keep=3, overwrite=True)
            
    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    logging.info("Best test acc %.4f at epoch %d" % (best_val_acc, best_epoch + 1))


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
    if not (FLAGS.resume_training and os.path.exists(output_dir)):
        create_dir(output_dir)

    # train and eval
    state, aux = configure_train(configs, output_dir, resume_checkpoint=FLAGS.resume_training)
    train_and_evaluate(configs, output_dir, state, *aux)


if __name__=='__main__':
    configure_flags()
    app.run(main)

"""Evaluation script for clean, CIFAR-10C corruption, and adversarial PGD.

Usage:
  python eval.py --checkpoint_name safe_0.95_s1
  python eval.py --checkpoint_name safe_0.95_s1 --mode corrupt --corrupt_intensity 3
  python eval.py --checkpoint_name safe_0.95_s1 --mode adversarial --attack_norm linf
"""

from functools import partial

from absl import app, flags, logging
import tensorflow as tf
import jax
from flax import jax_utils
import ml_collections

from train_utils import configure_train
from evaluate import (
    eval_step, bn_step, evaluate, evaluate_corruptions, evaluate_adversarial,
    fgsm, pgd_attack, attack_step, configure_flags, resolve_checkpoint
)

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 2:
        raise app.UsageError('Too many command-line arguments.')

    tf.config.experimental.set_visible_devices([], 'GPU')

    output_dir, configs = resolve_checkpoint(FLAGS)
    d = configs.to_dict()
    d['eval_sparsities'] = [float(s) for s in FLAGS.eval_sparsities.split(',')] if FLAGS.eval_sparsities else [configs.sp]
    if FLAGS.mode == 'corrupt':
        d['corrupt_intensity'] = FLAGS.corrupt_intensity
    configs = ml_collections.ConfigDict(d)

    state, (
        loss_type, Metrics, train_iter, eval_iter, _, model_info
    ) = configure_train(configs, output_dir, resume_checkpoint=True)
    p_eval_step = jax.pmap(
        partial(eval_step, loss_type=loss_type, metrics=Metrics),
        axis_name='batch'
    )
    p_bn_step = jax.pmap(bn_step, axis_name='batch')
    state = jax_utils.replicate(state)

    if FLAGS.mode == 'clean':
        results = evaluate(
            configs, state, p_eval_step, p_bn_step,
            train_iter, eval_iter, model_info, compute_bnt=True
        )

    elif FLAGS.mode == 'corrupt':
        results = evaluate_corruptions(
            configs, state, p_eval_step, p_bn_step,
            train_iter, model_info
        )
        logging.info(f'Results:\n{results}')

    elif FLAGS.mode == 'adversarial':
        if FLAGS.attack_opt == 'fgsm':
            base_tx = fgsm(lr=-FLAGS.attack_lr)
        else:
            raise ValueError(f"Unknown attack_opt: {FLAGS.attack_opt}")
        tx = pgd_attack(base_tx, FLAGS.attack_bound, norm=FLAGS.attack_norm)

        p_attack_step = jax.pmap(
            partial(attack_step, loss_type=loss_type, tx=tx),
            axis_name='batch'
        )
        results = evaluate_adversarial(
            configs, state, p_eval_step, p_bn_step, p_attack_step,
            train_iter, eval_iter, model_info, tx, attack_steps=FLAGS.attack_steps
        )
        logging.info(f'Results:\n{results}')

    else:
        raise ValueError(f"Unknown mode: {FLAGS.mode}. Use clean, corrupt, or adversarial.")


if __name__ == '__main__':
    configure_flags()
    flags.DEFINE_string('mode', 'clean', "Evaluation mode: clean, corrupt, or adversarial.")
    flags.DEFINE_string('eval_sparsities', '', "Comma-separated sparsities.")

    # corrupt mode
    flags.DEFINE_integer('corrupt_intensity', 3, "Corruption intensity (1-5).")

    # adversarial mode
    flags.DEFINE_string('attack_opt', 'fgsm', "Attack optimizer type.")
    flags.DEFINE_integer('attack_steps', 10, "PGD attack steps.")
    flags.DEFINE_float('attack_bound', 1/255, "Epsilon bound.")
    flags.DEFINE_float('attack_lr', 1/255/4, "Attack step size.")
    flags.DEFINE_string('attack_norm', 'linf', "Norm type: linf or l2.")

    app.run(main)

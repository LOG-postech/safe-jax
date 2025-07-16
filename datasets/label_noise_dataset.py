# coding=utf-8
"""https://github.com/google-research/google-research/blob/master/ieg/dataset_utils/datasets.py#L564"""

from math import ceil
from dataclasses import dataclass
from typing import Any

import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

import jax
from flax import jax_utils


def verbose_data(which_set, data, label):
  """Prints the number of data per class for a dataset."""
  text = [f'{which_set} size: {data.shape[0]}']
  text += [f'class{i}-{len(np.where(label == i)[0])}' for i in range(label.max() + 1)]
  text = ' '.join(text) + '\n'
  tf.logging.info(text)


def shuffle_dataset(data, label, others=None, class_balanced=False):
  """Shuffles the dataset with class balancing option.

  Args:
    data: A numpy 4D array.
    label: A numpy array.
    others: Optional array corresponded with data and label.
    class_balanced: If True, after shuffle, data of different classes are
      interleaved and balanced [1,2,3...,1,2,3.].

  Returns:
    Shuffled inputs.
  """
  if class_balanced:
    sorted_ids = []

    for i in range(label.max() + 1):
      tmp_ids = np.where(label == i)[0]
      np.random.shuffle(tmp_ids)
      sorted_ids.append(tmp_ids)

    sorted_ids = np.stack(sorted_ids, 0)
    sorted_ids = np.transpose(sorted_ids, axes=[1, 0])
    ids = np.reshape(sorted_ids, (-1,))

  else:
    ids = np.arange(data.shape[0])
    np.random.shuffle(ids)

  if others is None:
    return data[ids], label[ids]
  else:
    return data[ids], label[ids], others[ids]


def load_asymmetric(x, y, noise_ratio, n_val, random_seed=12345):
  """Create asymmetric noisy data."""

  def _generate_asymmetric_noise(y_train, n):
    """Generate cifar10 asymmetric label noise.

    Asymmetric noise confuses
      automobile <- truck
      bird -> airplane
      cat <-> dog
      deer -> horse

    Args:
      y_train: label numpy tensor
      n: noise ratio

    Returns:
      corrupted y_train.
    """
    assert y_train.max() == 10 - 1
    classes = 10
    p = np.eye(classes)

    # automobile <- truck
    p[9, 9], p[9, 1] = 1. - n, n
    # bird -> airplane
    p[2, 2], p[2, 0] = 1. - n, n
    # cat <-> dog
    p[3, 3], p[3, 5] = 1. - n, n
    p[5, 5], p[5, 3] = 1. - n, n
    # automobile -> truck
    p[4, 4], p[4, 7] = 1. - n, n
    tf.logging.info('Asymmetric corruption p:\n {}'.format(p))

    noise_y = y_train.copy()
    r = np.random.RandomState(random_seed)

    for i in range(noise_y.shape[0]):
      c = y_train[i]
      s = r.multinomial(1, p[c, :], 1)[0]
      noise_y[i] = np.where(s == 1)[0]

    actual_noise = (noise_y != y_train).mean()
    assert actual_noise > 0.0

    return noise_y

  n_img = x.shape[0]
  n_classes = 10

  # holdout balanced clean
  val_idx = []
  if n_val > 0:
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  trainlabel = trainlabel.squeeze()
  label_corr_train = trainlabel.copy()

  trainlabel = _generate_asymmetric_noise(trainlabel, noise_ratio)

  if len(trainlabel.shape) == 1:
    trainlabel = np.reshape(trainlabel, [trainlabel.shape[0], 1])

  traindata, trainlabel, label_corr_train = shuffle_dataset(  # pylint: disable=unbalanced-tuple-unpacking
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


def load_train_val_uniform_noise(x, y, n_classes, n_val, noise_ratio):
  """Make noisy data and holdout a clean val data.

  Constructs training and validation datasets, with controllable amount of
  noise ratio.

  Args:
    x: 4D numpy array of images
    y: 1D numpy array of labels of images
    n_classes: The number of classes.
    n_val: The number of validation data to holdout from train.
    noise_ratio: A float number that decides the random noise ratio.

  Returns:
    traindata: Train data.
    trainlabel: Train noisy label.
    label_corr_train: True clean label.
    valdata: Validation data.
    vallabel: Validation label.
  """
  n_img = x.shape[0]
  val_idx = []
  if n_val > 0:
    # Splits a clean holdout set
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  # split validation set
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  # Copies the true label for verification
  label_corr_train = trainlabel.copy()
  # Adds uniform noises
  mask = np.random.rand(len(trainlabel)) <= noise_ratio
  random_labels = np.random.choice(n_classes, mask.sum())
  flipped_labels = (trainlabel.squeeze() + np.random.choice(np.arange(1, n_classes), len(trainlabel))) % n_classes

  trainlabel = np.expand_dims(np.where(mask, flipped_labels, trainlabel.squeeze()), 1)

  # Shuffles dataset
  traindata, trainlabel, label_corr_train = shuffle_dataset(  # pylint: disable=unbalanced-tuple-unpacking
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


@dataclass
class TFIterWrapper:
    tfds_iter: Any
    data_info: dict
    n_data: int
    step_per_epoch: int
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count > len(self):
            raise StopIteration
        else:
            self.count += 1
            return next(self.tfds_iter)
        
    def __len__(self):
        return self.step_per_epoch


def inject_noise(x_train, y_train, target_ratio, seed, data_info, noise_type='uniform', data_ratio=1.0):
    if noise_type=='asymmetric':
      (x_train, y_train_noisy, y_train_true), _ = load_asymmetric(
                                                    x_train, y_train,
                                                    random_seed=seed,
                                                    noise_ratio=target_ratio,
                                                    n_val=0)
    elif noise_type=='uniform':
      (x_train, y_train_noisy, y_train_true), _ = load_train_val_uniform_noise(
                                                    x_train, y_train,
                                                    n_classes=data_info['num_classes'],
                                                    noise_ratio=target_ratio,
                                                    n_val=0)
    
    if data_ratio != 1.0:
      x_train, _, y_train_noisy, _ = train_test_split(x_train, y_train_noisy, test_size=int(x_train.shape[0] * (1 - data_ratio)), random_state=42, stratify=y_train_noisy)
      y_train_true = y_train_noisy
    
    conf_mat = sklearn_metrics.confusion_matrix(y_train_true, y_train_noisy)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
    print('Corrupted confusion matirx\n {}'.format(conf_mat))
    
    return x_train, y_train_noisy
    


def get_cifar10_lable_noise_datasets(batch_size, target_ratio, seed, noise_type='uniform', data_ratio=1.0):
    """Creates loader as tf.data.Dataset."""
    np.random.seed(seed)

    data_info = {'dataset': 'cifar10', 'input_shape': (1, 32, 32, 3), 'num_classes': 10, 
                 'task': 'classification',
                 'rgb_mean': 255*tf.constant([0.4914, 0.4822, 0.4465], shape=[1, 1, 3], dtype=tf.float32),
                 'rgb_sdv': 255*tf.constant([0.2023, 0.1994, 0.2010], shape=[1, 1, 3], dtype=tf.float32)}
    
    def create_ds(data, is_train=True):
        ds = tf.data.Dataset.from_tensor_slices({"image": data[0], "label": data[1]}).cache()
        
        def decode_example(sample):
            image = tf.cast(sample['image'], tf.float32)
            image = (image - data_info['rgb_mean']) / data_info['rgb_sdv']
            if is_train:
                image = tf.pad(image, [[4, 4],
                                        [4, 4], [0, 0]], 'CONSTANT')
                image = tf.image.random_crop(image, [32, 32, 3])
                image = tf.image.random_flip_left_right(image)
            batch = {'sample': image, 'target': sample['label']}
            return batch
        
        ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_train:
            ds = ds.shuffle(data[1].shape[0], seed=0, reshuffle_each_iteration=True)
        
        ds = ds.batch(batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(10)
        
        # For multiple device
        def _prepare(x):
            x = x._numpy()
            return x.reshape((jax.local_device_count(), -1) + x.shape[1:])
        it = map(lambda xs: jax.tree_util.tree_map(_prepare, xs), ds)
        it = jax_utils.prefetch_to_device(it, 2)
        return it
    
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train, y_train = shuffle_dataset(x_train, y_train.astype(np.int32))
    x_train, y_train_noisy = inject_noise(x_train, y_train, target_ratio, seed, data_info, noise_type=noise_type, data_ratio=data_ratio)
    train_size = x_train.shape[0]
    train_it = create_ds((x_train, y_train_noisy.squeeze()), is_train=True)
    tf_train_it = TFIterWrapper(train_it, data_info, train_size, ceil(train_size/batch_size))
    
    x_test, y_test = shuffle_dataset(x_test, y_test.astype(np.int32))
    val_size = x_test.shape[0]
    val_it= create_ds((x_test, y_test.squeeze()), is_train=False)
    tf_val_it = TFIterWrapper(val_it, data_info, val_size, ceil(val_size/batch_size))

    verbose_data('train', x_train, y_train)
    verbose_data('test', x_test, y_test)

    return tf_train_it, tf_val_it

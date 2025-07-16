from .image_process import *
from .label_noise_dataset import *

from math import ceil

import jax
from flax import jax_utils
import tensorflow as tf
import tensorflow_datasets as tfds


class TFDataLoader:
    def __init__(
        self,
        dataset: str,
        batch_size: int,
        train=True,
        valid=False,
        cache=False
    ):

        dataset_builder, data_info, as_dataset_kwargs, decode_example = self.__prepare_data(dataset, train)
        
        self.dataset_builder = dataset_builder
        self.data_info = data_info
        self.decode_example = decode_example
        
        self.__dset = 'train' if train else data_info['test_set']
        self.n_data = self.dataset_builder.info.splits[self.__dset].num_examples
        self.step_per_epoch = ceil(self.n_data/batch_size)
        
        self.tfds_iter = self.__get_tf_dataset(batch_size, train, valid, cache, as_dataset_kwargs)
        self.count = 0
        
    @staticmethod
    def __prepare_data(dataset: str, train=True):
        dataset_builder = tfds.builder(dataset)
            
        if dataset in 'mnist':
            dataset_builder.download_and_prepare()
            data_info = {'dataset': dataset, 'input_shape': (1, 28, 28, 1), 'num_classes': 10, 
                         'task': 'classification', 'test_set': 'test'}
            as_dataset_kwargs = {}
            
            def decode_example(sample):
                image = tf.cast(sample['image'], tf.float32) / 255.
                batch = {'sample': image, 'target': sample['label']}
                return batch
            
        elif dataset=='cifar10':
            dataset_builder.download_and_prepare()
            data_info = {'dataset': dataset, 'input_shape': (1, 32, 32, 3), 'num_classes': 10, 
                         'task': 'classification','test_set': 'test', 
                         'rgb_mean': 255*tf.constant([0.4914, 0.4822, 0.4465], shape=[1, 1, 3], dtype=tf.float32),
                         'rgb_sdv': 255*tf.constant([0.2023, 0.1994, 0.2010], shape=[1, 1, 3], dtype=tf.float32)}
            as_dataset_kwargs = {}

            def decode_example(sample):
                image = tf.cast(sample['image'], tf.float32)
                image = (image - data_info['rgb_mean']) / data_info['rgb_sdv']
                if train:
                    image = tf.pad(image, [[4, 4],
                                           [4, 4], [0, 0]], 'CONSTANT')
                    image = tf.image.random_crop(image, [32, 32, 3])
                    image = tf.image.random_flip_left_right(image)
                batch = {'sample': image, 'target': sample['label']}
                return batch
        
            
        elif dataset=='cifar100':
            dataset_builder.download_and_prepare()
            data_info = {'dataset': dataset, 'input_shape': (1, 32, 32, 3), 'num_classes': 100, 
                         'task': 'classification','test_set': 'test', 
                         'rgb_mean': 255* tf.constant([0.5071, 0.4867, 0.4408], shape=[1, 1, 3], dtype=tf.float32),
                         'rgb_sdv': 255*tf.constant([0.2675, 0.2565, 0.2761], shape=[1, 1, 3], dtype=tf.float32)}
            as_dataset_kwargs = {}

            def decode_example(sample):
                image = tf.cast(sample['image'], tf.float32)
                image = (image - data_info['rgb_mean']) / data_info['rgb_sdv']
                if train:
                    image = tf.pad(image, [[4, 4],
                                           [4, 4], [0, 0]], 'CONSTANT')
                    image = tf.image.random_crop(image, [32, 32, 3])
                    image = tf.image.random_flip_left_right(image)
                batch = {'sample': image, 'target': sample['label']}
                return batch
        
        return dataset_builder, data_info, as_dataset_kwargs, decode_example
    
    def __get_tf_dataset(self, batch_size, train, valid, cache, as_dataset_kwargs):
        
        split_size = self.n_data // jax.process_count()
        start = jax.process_index()*split_size
        split = f'{self.__dset}[{start}:{start+split_size}]'
        
        if valid:
            split = 'train[10%:]' if train else 'train[:10%]'
        
        ds = self.dataset_builder.as_dataset(split=split, **as_dataset_kwargs)
        ds = ds.map(self.decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                
        if train:
            buff_size = self.n_data
            ds = ds.shuffle(buff_size, seed=0, reshuffle_each_iteration=True)
        
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.repeat()
        ds = ds.prefetch(10)
        
        # For multiple device
        def _prepare(x):
            x = x._numpy()
            return x.reshape((jax.local_device_count(), -1) + x.shape[1:])
        it = map(lambda xs: jax.tree_util.tree_map(_prepare, xs), ds)
        it = jax_utils.prefetch_to_device(it, 2)
        
        return it
    
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

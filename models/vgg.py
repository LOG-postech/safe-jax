from functools import partial
from typing import Any, Sequence

from jax import random
import jax.numpy as jnp
import flax.linen as nn
import warnings

# Code taken from https://github.com/matthias-wright/flaxmodels/blob/main/flaxmodels/vgg/vgg.py

class VGG(nn.Module):
    """
    VGG.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the VGG activations
        num_classes (int):
            Number of classes. Only relevant if 'include_head' is True.
        dtype (str): Data type.
    """
    num_classes: int
    stage_size: Sequence[int]
    num_filters: int = 64
    dtype: str='float32'

    def setup(self):
        assert len(self.stage_size)==5

    @nn.compact
    def __call__(self, x, train=True):
        """
        Args:
            x (tensor of shape [N, H, W, 3]):
                Batch of input images (RGB format). Images must be in range [0, 1].
                If 'include_head' is True, the images must be 224x224.
            train (bool): Training mode.

        Returns:
            (tensor): Output tensor of shape [N, num_classes] (softmax).
        """

        x = self._conv_block(x, features=self.num_filters, num_layers=self.stage_size[0], block_num=1, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=self.num_filters * 2, num_layers=self.stage_size[1], block_num=2, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=self.num_filters * 4, num_layers=self.stage_size[2], block_num=3, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=self.num_filters * 8, num_layers=self.stage_size[3], block_num=4, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=self.num_filters * 8, num_layers=self.stage_size[4], block_num=5, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x, (-1, x.shape[1] * x.shape[2] * x.shape[3]))
        # x = self._fc_block(x, features=4096, block_num=6, relu=True, dropout=True, train=train, dtype=self.dtype)
        # x = self._fc_block(x, features=4096, block_num=7, relu=True, dropout=True, train=train, dtype=self.dtype)
        x = self._fc_block(x, features=512, block_num=6, relu=True, dropout=True, train=train, dtype=self.dtype)
        x = self._fc_block(x, features=512, block_num=7, relu=True, dropout=True, train=train, dtype=self.dtype)
        x = self._fc_block(x, features=self.num_classes, block_num=8, relu=False, dropout=False, train=train, dtype=self.dtype)

        return x

    def _conv_block(self, x, features, num_layers, block_num, dtype='float32'):
        for l in range(num_layers):
            layer_name = f'conv{block_num}_{l + 1}'
            x = nn.Conv(features=features, kernel_size=(3, 3), kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros, padding='same', name=layer_name, dtype=dtype)(x)
            x = nn.relu(x)
        return x
    
    def _fc_block(self, x, features, block_num, relu=False, dropout=False, train=True, dtype='float32'):
        layer_name = f'fc{block_num}'
        x = nn.Dense(features=features, kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros, name=layer_name, dtype=dtype)(x)
        if relu: x = nn.relu(x)
        if dropout: x = nn.Dropout(rate=0.5)(x, deterministic=not train)
        return x  


class VGGBN(nn.Module):
    """
    VGG.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the VGG activations
        num_classes (int):
            Number of classes. Only relevant if 'include_head' is True.
        dtype (str): Data type.
    """
    num_classes: int
    stage_size: Sequence[int]
    num_filters: int = 64
    dtype: str='float32'

    def setup(self):
        assert len(self.stage_size)==5

    @nn.compact
    def __call__(self, x, train=True):
        """
        Args:
            x (tensor of shape [N, H, W, 3]):
                Batch of input images (RGB format). Images must be in range [0, 1].
                If 'include_head' is True, the images must be 224x224.
            train (bool): Training mode.

        Returns:
            (tensor): Output tensor of shape [N, num_classes] (softmax).
        """
        norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

        x = self._conv_block(x, features=self.num_filters, num_layers=self.stage_size[0], norm=norm, block_num=1, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=self.num_filters * 2, num_layers=self.stage_size[1], norm=norm, block_num=2, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=self.num_filters * 4, num_layers=self.stage_size[2], norm=norm, block_num=3, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=self.num_filters * 8, num_layers=self.stage_size[3], norm=norm, block_num=4, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self._conv_block(x, features=self.num_filters * 8, num_layers=self.stage_size[4], norm=norm, block_num=5, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x, (-1, x.shape[1] * x.shape[2] * x.shape[3]))
        # x = self._fc_block(x, features=4096, block_num=6, relu=True, dropout=True, train=train, dtype=self.dtype)
        # x = self._fc_block(x, features=4096, block_num=7, relu=True, dropout=True, train=train, dtype=self.dtype)
        x = self._fc_block(x, features=512, block_num=6, relu=True, dropout=False, train=train, dtype=self.dtype)
        x = self._fc_block(x, features=512, block_num=7, relu=True, dropout=False, train=train, dtype=self.dtype)
        x = self._fc_block(x, features=self.num_classes, block_num=8, relu=False, dropout=False, train=train, dtype=self.dtype)

        return x

    def _conv_block(self, x, features, num_layers, norm, block_num, dtype='float32'):
        for l in range(num_layers):
            layer_name = f'conv{block_num}_{l + 1}'
            x = nn.Conv(features=features, kernel_size=(3, 3), kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros, padding='same', name=layer_name, dtype=dtype)(x)
            x = norm()(x)
            x = nn.relu(x)
        return x
    
    def _fc_block(self, x, features, block_num, relu=False, dropout=False, train=True, dtype='float32'):
        layer_name = f'fc{block_num}'
        x = nn.Dense(features=features, kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros, name=layer_name, dtype=dtype)(x)
        if relu: x = nn.relu(x)
        if dropout: x = nn.Dropout(rate=0.5)(x, deterministic=not train)
        return x  


VGG11 = partial(VGG, stage_size=[1, 1, 2, 2, 2])
VGG13 = partial(VGG, stage_size=[2, 2, 2, 2, 2])
VGG16 = partial(VGG, stage_size=[2, 2, 3, 3, 3])
VGG19 = partial(VGG, stage_size=[2, 2, 4, 4, 4])

VGG11_bn = partial(VGGBN, stage_size=[1, 1, 2, 2, 2])
VGG13_bn = partial(VGGBN, stage_size=[2, 2, 2, 2, 2])
VGG16_bn = partial(VGGBN, stage_size=[2, 2, 3, 3, 3])
VGG19_bn = partial(VGGBN, stage_size=[2, 2, 4, 4, 4])
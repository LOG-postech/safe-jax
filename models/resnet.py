# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

from jax import random
from jax._src import core
from jax._src import dtypes

Array = Any
ModuleDef = Any
KeyArray = Any
DTypeLikeInexact = Any

def custom_bias_init(key: KeyArray,
                      shape: core.Shape,
                      dtype:DTypeLikeInexact) -> Array:
  dtype = dtypes.canonicalize_dtype(dtype)
  named_shape = core.as_named_shape(shape)
  fan_in = shape[-1]
  variance = jnp.array(1 / fan_in, dtype=dtype)
  return random.uniform(key, named_shape, dtype, -1) * jnp.sqrt(variance)


class BasicCifarBlock(nn.Module):
  """ResNet block."""
  block_gates: Sequence[bool]
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)
  downsample: bool = False
  option: str = 'B'

  @nn.compact
  def __call__(self, x,):
    residual = x

    if self.block_gates[0]:
      y = self.conv(self.filters, (3, 3), self.strides, padding=[(1, 1), (1, 1)])(x)
      y = self.norm()(y)
      y = self.act(y)

    if self.block_gates[1]:
      y = self.conv(self.filters, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(y)
      y = self.norm()(y)

    if self.downsample:
      if self.option == 'A':
        residual = jnp.pad(x[:, ::2, ::2, :], ((0, 0), (0, 0), (0, 0), (self.filters//4, self.filters//4)))
      elif self.option == 'B':
        residual = self.conv(self.filters, (1, 1),
                            self.strides, 
                            padding=[(0, 0), (0, 0)], name='conv_proj')(residual)
        residual = self.norm(name='norm_proj')(residual)

    y += residual
    y = self.act(y)
    return y
  
class ResNetCifar(nn.Module):
  """For ResNet20, 32, 44"""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 16
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, train: bool = True):
    layer_gates = []
    for layer in range(3):
      layer_gates.append([])
      for _ in range(self.stage_sizes[layer]):
        layer_gates[layer].append([True, True])
    
    conv = partial(
      self.conv,
      use_bias=False,
      dtype=self.dtype,
      kernel_init=nn.initializers.variance_scaling(
        scale=2., mode="fan_out", distribution="normal"
      )
    ) # 'fan_in' in ToST and 'fan_out' in CrAM and GraSP
    norm = partial(
      nn.BatchNorm,
      use_running_average=not train,
      momentum=0.9,
      epsilon=1e-5,
      dtype=self.dtype
    )

    x = conv(
      self.num_filters, (3, 3), (1, 1),
      padding=[(1, 1), (1, 1)],
      name='conv_init'
    )(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    
    for i, block_size in enumerate(self.stage_sizes):
      downsample = (i > 0)
      strides = (2, 2) if i > 0 else (1, 1)
      x = self.block_cls(layer_gates[i][0], self.num_filters * 2 ** i, strides=strides, conv=conv, norm=norm, act=self.act, downsample=downsample)(x)
      for j in range(1, block_size):
        x = self.block_cls(
          layer_gates[i][j],
          self.num_filters * 2 ** i,
          strides=(1, 1),
          conv=conv,
          norm=norm,
          act=self.act
        )(x)
    x = nn.avg_pool(x, (8, 8), strides=(8, 8), padding=[(0, 0), (0, 0)])
    x = x.reshape(x.shape[0], -1)
    x = nn.Dense(self.num_classes, dtype=self.dtype, kernel_init=nn.initializers.normal(), bias_init=nn.initializers.constant(0))(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet20 = partial(ResNetCifar, stage_sizes=[3, 3, 3], block_cls=BasicCifarBlock) # original ResNet20
ResNet32 = partial(ResNetCifar, stage_sizes=[5, 5, 5], block_cls=BasicCifarBlock) # original ResNet32
ResNet44 = partial(ResNetCifar, stage_sizes=[7, 7, 7], block_cls=BasicCifarBlock)
ResNet56 = partial(ResNetCifar, stage_sizes=[9, 9, 9], block_cls=BasicCifarBlock)

ResNet20x2 = partial(ResNetCifar, stage_sizes=[3, 3, 3], block_cls=BasicCifarBlock, num_filters=32)  # ResNet20 with double kernel count
ResNet32x2 = partial(ResNetCifar, stage_sizes=[5, 5, 5], block_cls=BasicCifarBlock, num_filters=32)  # ResNet32 with double kernel count
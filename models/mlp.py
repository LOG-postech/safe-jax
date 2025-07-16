# Code partially taken from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py

from typing import Any, Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp

class MLP(nn.Module):
  num_classes: int
  num_neurons: Sequence[int]
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, x, train: bool = True):
    x = x.reshape((x.shape[0], -1))
    for i, num_neuron in enumerate(self.num_neurons):
      x = nn.Dense(num_neuron, dtype=self.dtype)(x)
      x = self.act(x)
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x

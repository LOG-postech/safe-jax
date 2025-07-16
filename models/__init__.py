from .mlp import MLP
from .resnet import ResNet20, ResNet32, ResNet44, ResNet56, ResNet20x2, ResNet32x2
from .vgg import VGG11, VGG13, VGG16, VGG19, VGG11_bn, VGG13_bn, VGG16_bn, VGG19_bn

import jax
import jax.numpy as jnp

def precision_dtype(half_precision):
    platform = jax.local_devices()[0].platform
    if half_precision:
        return jnp.bfloat16 if platform=='tpu' else jnp.float16
    else:
        return jnp.float32

def get_model(model_str, half_precision, num_classes):
    '''Return Model Object & Model Info of given model_str'''
    model_dtype = precision_dtype(half_precision)

    # MNIST
    if 'LeNet300-100' in model_str:
        model = MLP(num_classes=num_classes, num_neurons=(300, 100), dtype=model_dtype)
        model_info = {'batch_norm': False, 'dropout': False}
    
    # CIFAR10 / CIFAR100    
    elif model_str=='VGG11': 
        model = VGG11(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': False, 'dropout': True}
    elif model_str=='VGG13': 
        model = VGG13(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': False, 'dropout': True}
    elif model_str=='VGG16': 
        model = VGG16(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': False, 'dropout': True}
    elif model_str=='VGG19': 
        model = VGG19(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': False, 'dropout': True}

    elif model_str=='VGG11-bn': 
        model = VGG11_bn(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    elif model_str=='VGG13-bn': 
        model = VGG13_bn(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    elif model_str=='VGG16-bn': 
        model = VGG16_bn(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    elif model_str=='VGG19-bn': 
        model = VGG19_bn(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
        
    elif model_str=='ResNet20x2': 
        model = ResNet20x2(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    elif model_str=='ResNet32x2':
        model = ResNet32x2(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    
    elif model_str=='ResNet20': 
        model = ResNet20(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    elif model_str=='ResNet32': 
        model = ResNet32(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    elif model_str=='ResNet44': 
        model = ResNet44(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    elif model_str=='ResNet56': 
        model = ResNet56(num_classes=num_classes, dtype=model_dtype)
        model_info = {'batch_norm': True, 'dropout': False}
    
    return model, model_info


def initialized(key, input_shape, model, batch_stats=True, has_dropout=True):
    """Initialize model"""
    if has_dropout:
        variables = jax.jit(model.init, static_argnames=['train'])({'params': key}, jnp.ones(input_shape), train=False)
    else:
        variables = jax.jit(model.init)({'params': key}, jnp.ones(input_shape))
        
    if not batch_stats:
        return variables['params'], {}
    else:
        return variables['params'], variables['batch_stats']
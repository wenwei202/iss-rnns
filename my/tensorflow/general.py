from itertools import zip_longest

import itertools
import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np
import re

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def get_initializer(matrix):
    def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, var in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            assert g is not None, var.name
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def mask(val, mask, name=None):
    if name is None:
        name = 'mask'
    return tf.multiply(val, tf.cast(mask, 'float'), name=name)


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def add_wd(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    with tf.name_scope("weight_decay"):
        for var in variables:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="{}/wd".format(var.op.name))
            tf.add_to_collection('losses', weight_decay)

def _excluded_var_pattern():
    #return "(main/logits)|(main/p0/bi_attention)|(prepro/u1)"
    return "(thisisapatternwetrytoexcludenothing)"

def add_sparsity_regularization(wd, collection_name=None, scope=None):
    orig_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    variables = []
    for eachvar in orig_variables:
        if not re.match(_excluded_var_pattern(), eachvar.op.name):
            variables.append(eachvar)
    with tf.name_scope("sparsity_regular"):
        if len(variables):
            the_regularizer = tf.contrib.layers.l1_regularizer(scale=wd, scope=scope)
            reg_loss = tf.contrib.layers.apply_regularization(the_regularizer, variables)
            tf.add_to_collection('losses', reg_loss)
        # add to collections
        collection_name = collection_name or 'sparse_vars'
        for eachvar in variables:
            tf.add_to_collection(collection_name, eachvar)

def reduce_square_sum(var, start=0, end=0, axis=0):
    the_shape = var.get_shape().as_list()
    if len(the_shape) == 2:
        t = tf.square(var)
        t = tf.reduce_sum(t, axis=axis)
        assert(end>start and axis<2)
        t = tf.gather_nd(t,tf.range(start, end))
        return t
    else:
        raise NotImplementedError('variables with shapes != 2 is not implemented.')


def add_mixedlasso(groupwd, l1wd, collection_name=None, scope=None):
    orig_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    variables = []
    for eachvar in orig_variables:
        if not re.match(_excluded_var_pattern(), eachvar.op.name):
            variables.append(eachvar)
    with tf.name_scope("DimenGroupLasso"):
        collection_name = collection_name or 'sparse_vars'
        for eachvar in variables:
            the_shape = eachvar.get_shape().as_list()
            if len(the_shape)<=1: # l1 is group lasso when the group size is 1
                the_regularizer = tf.contrib.layers.l1_regularizer(scale=l1wd, scope=scope)
                reg = tf.contrib.layers.apply_regularization(the_regularizer, [eachvar])
            elif len(the_shape)==2:
                reg = 0.0
                for s, axis in zip(the_shape, range(len(the_shape))):
                    if s != np.prod(the_shape):
                        if s == 1:
                            the_regularizer = tf.contrib.layers.l1_regularizer(scale=l1wd, scope=scope)
                            reg = reg + tf.contrib.layers.apply_regularization(the_regularizer, [eachvar])
                        else:
                            t = tf.square(eachvar)
                            t = tf.reduce_sum(t, axis=axis) + tf.constant(1.0e-8)
                            t = tf.sqrt(t)
                            reg = reg + tf.reduce_sum(t) * groupwd
            else:
                raise NotImplementedError('variables with shapes > 2 is not implemented.')

            tf.add_to_collection('losses', reg)
            tf.add_to_collection(collection_name, eachvar)

def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
    args = [iter(iterable)] * n
    out = zip_longest(*args, fillvalue=fillvalue)
    out = list(out)
    if num_groups is not None:
        default = (fillvalue, ) * n
        assert isinstance(num_groups, int)
        out = list(each for each, _ in zip_longest(out, range(num_groups), fillvalue=default))
    if shorten:
        assert fillvalue is None
        out = (tuple(e for e in each if e is not None) for each in out)
    return out

def padded_reshape(tensor, shape, mode='CONSTANT', name=None):
    paddings = [[0, shape[i] - tf.shape(tensor)[i]] for i in range(len(shape))]
    return tf.pad(tensor, paddings, mode=mode, name=name)


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def zerout_gradients_for_zero_weights(grads_and_vars, zero_threshold=0.0, mode='element'):
  """ zerout gradients for weights with zero values, so as to freeze zero weights.
  (make sure the history gradients are zeros too, otherwise, zero weights can still be updated in adam etc)
  Args:
      grads_and_vars: Lists of (gradient, variable).
      mode: the mode to freeze weights.
        'element': freeze all zero weights
        'group': freeze rows/columns that are fully zeros
  """
  gradients, variables = zip(*grads_and_vars)
  zerout_gradients = []
  for gradient, variable in zip(gradients, variables):
    if gradient is None:
      zerout_gradients.append(None)
      continue

    if mode=='element':
      where_cond = tf.less_equal(tf.abs(variable), zero_threshold)
    elif mode=='group':
      raise NotImplementedError('Group wise freezing is not implemented yet.')
    else:
      raise ValueError('Unsupported mode == %s' % mode)

    zerout_gradient = tf.where(where_cond,
             tf.zeros_like(gradient),
             gradient)
    zerout_gradients.append(zerout_gradient)
  return list(zip(zerout_gradients, variables))
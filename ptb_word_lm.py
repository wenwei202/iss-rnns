# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import pylab
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from time import gmtime, strftime

import reader
import importlib
import os.path
import matplotlib.pyplot as plt

flags = tf.flags
zero_threshold = 0.0001

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large, sparselarge, validtestlarge.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("restore_path", None,
                    "Model input directory.")
flags.DEFINE_string("config_file", None,
                    "Parameter config file.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("display_weights", False,
                  "Display weight matrix.")
flags.DEFINE_string("regularizer", 'l1_regularizer',
                    "Regularizer type.")
flags.DEFINE_string("optimizer", 'gd',
                    "Optimizer of sgd: gd and adam.")

FLAGS = flags.FLAGS

def add_dimen_grouplasso(var, axis=0):
  with tf.name_scope("DimenGroupLasso"):
    t = tf.square(var)
    t = tf.reduce_sum(t, axis=axis) + tf.constant(1.0e-8)
    t = tf.sqrt(t)
    reg = tf.reduce_sum(t)
    return reg


def add_blockwise_grouplasso(t, block_row_size, block_col_size):
  raise NotImplementedError('Not debugged. And the implementation is very slow when block is small.')
  with tf.name_scope("BlockGroupLasso"):
    t = tf.expand_dims(tf.expand_dims(t,0),-1)
    blocks = tf.extract_image_patches(t,
                ksizes=[1, block_row_size, block_col_size, 1],
                strides=[1, block_row_size, block_col_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID')
    reg_sum = tf.constant(0.0)
    zero_blocks = 0.0
    total_blocks = 0.0
    blocks = tf.unstack(blocks) # list of 3-D tensors
    for b in blocks: # for each 3-D tensor
      for bb in tf.unstack(b): # for each 2-D tensor
        for block in tf.unstack(bb): # for each block
          blk_len = tf.sqrt(tf.reduce_sum(tf.square(block))) + tf.constant(1.0e-8)
          reg_sum = reg_sum + tf.cond(blk_len < zero_threshold,
                                      lambda: tf.constant(0.0),
                                      lambda: blk_len)

          # set them to zeros and calculate sparsity
          #block = tf.assign(block, tf.cond(blk_len < zero_threshold,
          #                                 lambda: tf.zeros_like(block),
          #                                 lambda: block))
          zero_blocks = zero_blocks + tf.cond( tf.equal(tf.reduce_sum(tf.square(block)), 0.0),
                                      lambda: tf.constant(1.0),
                                      lambda: tf.constant(0.0))
          total_blocks = total_blocks + 1.0
    return reg_sum, zero_blocks/total_blocks

def plot_tensor(t,title):
  if len(t.shape)==2:
    print(title)
    col_zero_idx = np.sum(np.abs(t), axis=0) == 0
    row_zero_idx = np.sum(np.abs(t), axis=1) == 0
    col_sparsity = (' column sparsity: %d/%d' % (sum(col_zero_idx), t.shape[1]) )
    row_sparsity = (' row sparsity: %d/%d' % (sum(row_zero_idx), t.shape[0]) )

    plt.figure()

    t = (t != 0)
    weight_scope = abs(t).max()
    plt.subplot(3, 1, 1)
    plt.imshow(t.reshape((t.shape[0], -1)),
               vmin=-weight_scope,
               vmax=weight_scope,
               cmap=plt.get_cmap('binary'),
               interpolation='none')
    plt.title(title)

    col_zero_map = np.tile(col_zero_idx, (t.shape[0], 1))
    row_zero_map = np.tile(row_zero_idx.reshape((t.shape[0], 1)), (1, t.shape[1]))
    zero_map = col_zero_map + row_zero_map
    zero_map_cp = zero_map.copy()
    plt.subplot(3,1,2)
    plt.imshow(zero_map_cp,cmap=plt.get_cmap('gray'),interpolation='none')
    plt.title(col_sparsity + row_sparsity)

    if 2*t.shape[0] == t.shape[1]:
      subsize = int(t.shape[0]/2)
      match_map = np.zeros(subsize,dtype=np.int)
      match_map = match_map + row_zero_idx[subsize:2 * subsize]
      for blk in range(0,4):
        match_map = match_map + col_zero_idx[blk*subsize : blk*subsize+subsize]
      match_idx = np.where(match_map == 5)[0]
      zero_map[subsize+match_idx,:] = False
      for blk in range(0, 4):
        zero_map[:,blk*subsize+match_idx] = False
      plt.subplot(3, 1, 3)
      plt.imshow(zero_map, cmap=plt.get_cmap('Reds'), interpolation='none')
      plt.title(' %d/%d matches' % (len(match_idx), sum(row_zero_idx[subsize:subsize*2])))
  else:
    print ('ignoring %s' % title)


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_, config_params = None):
    self._input = input_
    self.config_params = config_params

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(
    #     cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])

    # L1 regularization
    modname = importlib.import_module('tensorflow.contrib.layers')
    the_regularizer = getattr(modname, FLAGS.regularizer)(scale=config_params['weight_decay'], scope=FLAGS.regularizer)
    reg_loss = tf.contrib.layers.apply_regularization(the_regularizer, tf.trainable_variables()[1:])
    self._regularization = reg_loss

    sparsity = {}

    # Group Lasso regularization
    if config_params:
      glasso_params = config_params.get('grouplasso', None)
    else:
      glasso_params = None

    if glasso_params:
      for train_var in tf.trainable_variables():
        var_name = train_var.op.name
        glasso_param = glasso_params.get(var_name,None)
        if glasso_param:
          glasso_reg = add_dimen_grouplasso(train_var, axis=0)
          self._regularization = self._regularization + glasso_reg * glasso_params['global_decay'] * glasso_param['col_decay_multi']
          glasso_reg = add_dimen_grouplasso(train_var, axis=1)
          self._regularization = self._regularization + glasso_reg*glasso_params['global_decay']*glasso_param['row_decay_multi']

    if config_params['weight_decay'] > 0 or glasso_params:
      # sparsity statistcis
      for train_var in tf.trainable_variables():
        # zerout by small threshold to stablize the sparsity
        sp_name = train_var.op.name
        threshold = max(zero_threshold, 2*config_params['weight_decay'])
        where_cond = tf.less(tf.abs(train_var), threshold)
        train_var = tf.assign(train_var, tf.where(where_cond,
                                                  tf.zeros(tf.shape(train_var)),
                                                  train_var))
        # statistics
        s = tf.nn.zero_fraction(train_var)
        sparsity[sp_name + '_elt_sparsity'] = s
        if glasso_params and glasso_params.get(sp_name,None):
          s = tf.nn.zero_fraction(tf.reduce_sum(tf.square(train_var), axis=0))
          sparsity[sp_name + '_col_sparsity'] = s
          s = tf.nn.zero_fraction(tf.reduce_sum(tf.square(train_var), axis=1))
          sparsity[sp_name + '_row_sparsity'] = s
    self._sparsity = sparsity

    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost + self._regularization, tvars),
                                      config.max_grad_norm)

    if 'gd' == FLAGS.optimizer:
      optimizer = tf.train.GradientDescentOptimizer(self._lr)
    elif 'adam' == FLAGS.optimizer:
      optimizer = tf.train.AdamOptimizer(self._lr)
    else:
      raise ValueError("Wrong optimizer!")

    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def regularization(self):
    return self._regularization

  @property
  def sparsity(self):
    return self._sparsity

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  def __init__(self):
    self.init_scale = 0.1
    self.learning_rate = 1.0
    self.max_grad_norm = 5
    self.num_layers = 2
    self.num_steps = 20
    self.hidden_size = 200
    self.max_epoch = 4
    self.max_max_epoch = 13
    self.keep_prob = 1.0
    self.lr_decay = 0.5
    self.batch_size = 20
    self.vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  def __init__(self):
    self.init_scale = 0.05
    self.learning_rate = 1.0
    self.max_grad_norm = 5
    self.num_layers = 2
    self.num_steps = 35
    self.hidden_size = 650
    self.max_epoch = 6
    self.max_max_epoch = 39
    self.keep_prob = 0.5
    self.lr_decay = 0.8
    self.batch_size = 20
    self.vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  def __init__(self):
    self.init_scale = 0.04
    self.learning_rate = 1.0
    self.max_grad_norm = 10
    self.num_layers = 2
    self.num_steps = 35
    self.hidden_size = 1500
    self.max_epoch = 14
    self.max_max_epoch = 55
    self.keep_prob = 0.35
    self.lr_decay = 1 / 1.15
    self.batch_size = 20
    self.vocab_size = 10000

class SparseLargeConfig(object):
  """Sparse Large config."""
  def __init__(self):
    self.init_scale = 0.04
    self.learning_rate = 1.0
    self.max_grad_norm = 10
    self.num_layers = 2
    self.num_steps = 35
    self.hidden_size = 1500
    self.max_epoch = 14
    self.max_max_epoch = 55
    self.keep_prob = 0.60
    self.lr_decay = 0.1
    self.batch_size = 20
    self.vocab_size = 10000

class ValidTestLargeConfig(object):
  """Large config."""
  def __init__(self):
    self.init_scale = 0.04
    self.learning_rate = 0.0
    self.max_grad_norm = 10
    self.num_layers = 2
    self.num_steps = 35
    self.hidden_size = 1500
    self.max_epoch = 0
    self.max_max_epoch = 0
    self.keep_prob = 1.0
    self.lr_decay = 1.0
    self.batch_size = 20
    self.vocab_size = 10000

class TestConfig(object):
  """Tiny config, for testing."""
  def __init__(self):
    self.init_scale = 0.1
    self.learning_rate = 1.0
    self.max_grad_norm = 1
    self.num_layers = 1
    self.num_steps = 2
    self.hidden_size = 2
    self.max_epoch = 1
    self.max_max_epoch = 1
    self.keep_prob = 1.0
    self.lr_decay = 0.5
    self.batch_size = 20
    self.vocab_size = 10000

def fetch_sparsity(session, model, eval_op=None, verbose=False):
  outputs = {}

  fetches = {
      "sparsity": model.sparsity
  }

  vals = session.run(fetches)
  sparsity = vals["sparsity"]
  outputs['sparsity'] = sparsity
  return outputs


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  outputs = {}
  regularizations = 0.0
  sparsity = {}
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "regularization": model.regularization,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    regularizations += vals["regularization"]
    sparsity = session.run(model.sparsity)
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f  cost: %.4f  regularization: %.4f  total_cost: %.4f   speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size,
             np.exp(costs / iters),
             costs / iters,
             regularizations / iters,
             costs / iters + regularizations / iters,
             iters * model.input.batch_size / (time.time() - start_time)))

  outputs['perplexity'] = np.exp(costs / iters)
  outputs['cross_entropy'] = costs / iters
  outputs['regularization'] = regularizations / iters
  outputs['total_cost'] = costs / iters + regularizations / iters
  outputs['sparsity'] = sparsity
  return outputs


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "sparselarge":
    return SparseLargeConfig()
  elif FLAGS.model == 'validtestlarge':
    return ValidTestLargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def restore_trainables(sess, path):
  if path:
    assert tf.gfile.Exists(path)
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
      variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      restorer = tf.train.Saver(variables_to_restore)
      if os.path.isabs(ckpt.model_checkpoint_path):
        restorer.restore(sess, ckpt.model_checkpoint_path)
      else:
        restorer.restore(sess, os.path.join(path,
                                            ckpt.model_checkpoint_path))
      print('Pre-trained model restored from %s' % path)
    else:
      print('Restoring pre-trained model from %s failed!' % path)
      exit()

def write_scalar_summary(summary_writer, tag, value, step):
  value = summary_pb2.Summary.Value(tag=tag, simple_value=float(value))
  summary = summary_pb2.Summary(value=[value])
  summary_writer.add_summary(summary, step)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  if not FLAGS.config_file:
    raise ValueError("Must set --config_file to configuration file")
  else:
    with open(FLAGS.config_file, 'r') as fi:
      config_params = json.load(fi)

  # get logger
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger('ptb_rnn')
  logger.setLevel(logging.INFO)
  # saving path
  subfolder_name = strftime("%Y-%m-%d___%H-%M-%S", gmtime())
  config_params['save_path'] = os.path.join(config_params['save_path'], subfolder_name)
  if not os.path.exists(config_params['save_path']):
    os.mkdir(config_params['save_path'])
  else:
    raise IOError('%s exist!' % config_params['save_path'])

  log_file = os.path.join(config_params['save_path'], 'output.log')

  logger.addHandler(logging.FileHandler(log_file))
  logger.info('configurations in file:\n %s \n', config_params)
  logger.info('tf.FLAGS:\n %s \n', vars(FLAGS))

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  config.keep_prob = config_params.get('dropout_keep_prob',config.keep_prob)
  config.learning_rate = config_params.get('learning_rate', config.learning_rate)
  eval_config = get_config()
  eval_config.keep_prob = config_params.get('dropout_keep_prob',eval_config.keep_prob)
  eval_config.learning_rate = config_params.get('learning_rate', eval_config.learning_rate)
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  logger.info('network configurations: \n %s \n', vars(config))

  with tf.Graph().as_default():

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input, config_params=config_params)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input, config_params=config_params)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input, config_params = config_params)

    saver = tf.train.Saver(tf.global_variables())
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.log_device_placement = False
    with tf.Session(config=config_proto) as session:
      coord = tf.train.Coordinator()
      session.run(init)
      threads = tf.train.start_queue_runners(sess=session, coord=coord)
      if FLAGS.restore_path:
        restore_trainables(session, FLAGS.restore_path)
        if FLAGS.display_weights:
          outputs = fetch_sparsity(session, mtest)
          print("Sparsity: %s" % outputs['sparsity'])
          for train_var in tf.trainable_variables():
            plot_tensor(train_var.eval(), train_var.op.name)
          plt.show()

        outputs = run_epoch(session, mvalid)
        print("Restored model with Valid Perplexity: %.3f" % (outputs['perplexity']))

      summary_writer = tf.summary.FileWriter(
        config_params['save_path'],
        graph=tf.get_default_graph())

      for i in range(config.max_max_epoch):
        if 'gd' == FLAGS.optimizer:
          # lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          lr_decay = config.lr_decay ** ( i // (config.max_max_epoch//3) )
        elif 'adam' == FLAGS.optimizer:
          lr_decay = 1.0
        else:
          raise ValueError("Wrong optimizer!")
        m.assign_lr(session, config.learning_rate * lr_decay)
        write_scalar_summary(summary_writer, 'learning_rate', config.learning_rate * lr_decay, i+1)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        outputs = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f   regularization: %.4f " % (i + 1, outputs['perplexity'], outputs['regularization']))
        write_scalar_summary(summary_writer, 'TrainPerplexity', outputs['perplexity'], i + 1)
        write_scalar_summary(summary_writer, 'cross_entropy', outputs['cross_entropy'], i + 1)
        write_scalar_summary(summary_writer, 'regularization', outputs['regularization'], i + 1)
        write_scalar_summary(summary_writer, 'total_cost', outputs['total_cost'], i + 1)
        for key, value in outputs['sparsity'].items():
          write_scalar_summary(summary_writer, key, value, i + 1)

        checkpoint_path = os.path.join(config_params['save_path'], 'model.ckpt')
        saver.save(session, checkpoint_path, global_step=i + 1)

        outputs = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, outputs['perplexity']))
        write_scalar_summary(summary_writer, 'ValidPerplexity', outputs['perplexity'], i + 1)

      outputs = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % outputs['perplexity'])
      write_scalar_summary(summary_writer, 'TestPerplexity', outputs['perplexity'], 0)

      coord.request_stop()
      coord.join(threads)
  plt.show()

if __name__ == "__main__":
  tf.app.run()

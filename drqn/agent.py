import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

from .base import BaseModel
from .history import History
from .ops import linearParams, conv2dParams, affine, conv2dOut
from .replay_memory import ReplayMemory
from utils import get_time, save_pkl, load_pkl

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.weight_dir = 'weights'

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_drqn()

  def train(self):
    start_step = self.step_op.eval()
    start_time = time.time()

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []

    screen, reward, action, terminal = self.env.new_random_game()

    for _ in range(self.history_length):
      self.history.add(screen)

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      # 1. predict
      action = self.predict(self.history.get())
      # 2. act
      screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe
      self.observe(screen, reward, action, terminal)

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(action)
      total_reward += reward

      if self.step >= self.learn_start:
        if self.step % self.test_step == self.test_step - 1:
          avg_reward = total_reward / self.test_step
          avg_loss = self.total_loss / self.update_count
          avg_q = self.total_q / self.update_count

          try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
              % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

          if max_avg_ep_reward * 0.9 <= avg_ep_reward:
            self.step_assign_op.eval({self.step_input: self.step + 1})
            self.save_model(self.step + 1)

            max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            self.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of game': num_game,
                'episode.rewards': ep_rewards,
                'episode.actions': actions,
                'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
              }, self.step)

          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.q_action.eval({self._s_t: [s_t]})[0]

    return action

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

  def q_learning_mini_batch(self):
    if self.memory.count < self.history_length * self.sequence_length:
      return
    else:
      s_t, actions, rewards, s_t_plus_1, terminal = self.memory.sample_sequence()

    t = time.time()

    actions = actions[:, self.init_sequence_length:]
    rewards = rewards[:, self.init_sequence_length:]
    terminal = terminal[:, self.init_sequence_length:]

    q_t_plus_1 = self.target_q_values.eval({self._target_s_t: s_t_plus_1})

    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = np.max(q_t_plus_1, axis=2)
    target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + rewards

    _, q_values, loss, summary_str = self.sess.run([self.optim, self.q_values, self.loss, self.q_seq_summary], {
      self._target_q_values: target_q_t,
      self._actions: actions,
      self._seq_t: s_t,
      self.learning_rate_step: self.step,
    })

    self.writer.add_summary(summary_str, self.step)
    self.total_loss += loss
    self.total_q += tf.reduce_mean(tf.reduce_mean(q_values, 1), 0)
    self.update_count += 1

  def build_drqn(self):
    self.w = {}
    self.t_w = {}
    self.truncated_seq_length = self.sequence_length - self.init_sequence_length

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    with tf.variable_scope('shared_params'):
      self.w['l1_w'], self.w['l1_b'] = conv2dParams(self.history_length, 32, [8, 8], initializer, name='l1')
      self.w['l2_w'], self.w['l2_b'] = conv2dParams(32, 64, [4, 4], initializer, name='l2')
      self.w['l3_w'], self.w['l3_b'] = conv2dParams(64, 64, [3, 3], initializer, name='l3')

      # Define l4 params
      self.w['q_w'], self.w['q_b'] = linearParams(512, self.env.action_size, name='q')
      # Define lstm cell 
      self.lstm_cell = rnn_cell.BasicLSTMCell(512, state_is_tuple=True)

    # training network
    with tf.variable_scope('sequence_prediction'):
      if self.cnn_format == 'NHWC':
        self._seq_t = tf.placeholder('float32',
            [None, self.sequence_length, self.screen_height, self.screen_width, self.history_length], name='seq_t')
      else:
        self._seq_t = tf.placeholder('float32',
            [None, self.sequence_length, self.history_length, self.screen_height, self.screen_width], name='seq_t')

      # For training with a full sequence
      seq = []

      for i in range(self.sequence_length):
        l1 = conv2dOut(self._seq_t[:, i, :, :, :], self.w['l1_w'], self.w['l1_b'], [4, 4], self.cnn_format)
        l2 = conv2dOut(l1, self.w['l2_w'], self.w['l2_b'], [2, 2], self.cnn_format)
        l3 = conv2dOut(l2, self.w['l3_w'], self.w['l3_b'], [1, 1], self.cnn_format)

        shape = l3.get_shape().as_list()
        l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
        seq.append(l3_flat)

      if self.dueling:
        print 'Dueling networks not supported for DRQN (yet)!'
     
      # hist_sequence shape: (sequence_length, batch_size, l3_flattened) 
      hist_sequence = tf.pack(seq)
      # Reshaping to (sequence_length * batch_size, l3_flattened)
      hist_sequence = tf.reshape(hist_sequence, [-1, hist_sequence.get_shape().as_list()[2]]) 
      # Split to get a list of 'sequence_length' tensors of shape (batch_size, l3_flattened)
      hist_sequence = tf.split(0, self.sequence_length, hist_sequence)
      # list of length `sequence_length` of Tensors of size (?, 512)
      l4, state = tf.nn.rnn(self.lstm_cell, hist_sequence, dtype=tf.float32)  
      # throw out sequence elements less than init_sequence_length 
      l4 = l4[self.init_sequence_length:]

      q_values = []
      q_actions = []

      for i in range(self.truncated_seq_length):
        q = affine(l4[i], self.w['q_w'], self.w['q_b'])
        q_action = tf.argmax(q, dimension=1)
        q_values.append(q)
        q_actions.append(q_action)

      self.q_values = tf.pack(q_values, axis=1)
      self.q_actions = tf.pack(q_actions, axis=1)

      q_summary = []
      avg_q = tf.reduce_mean(self.q_values, 0)
      avg_q = tf.reduce_mean(avg_q, 0)
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
      self.q_seq_summary = tf.merge_summary(q_summary, 'q_seq_summary')

    # Shared parameters with sequence_prediction network, but accepts single image as input
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self._s_t = tf.placeholder('float32',
            [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
      else:
        self._s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_height, self.screen_width], name='s_t')

      self.l1 = conv2dOut(self._s_t, self.w['l1_w'], self.w['l1_b'], [4, 4], self.cnn_format)
      self.l2 = conv2dOut(l1, self.w['l2_w'], self.w['l2_b'], [2, 2], self.cnn_format)
      self.l3 = conv2dOut(l2, self.w['l3_w'], self.w['l3_b'], [1, 1], self.cnn_format)

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      # l3_flat shape: (batch_size, l3_flattened) 
      # list of length 1 of Tensors of size (?, 512)
      self.l4, state = tf.nn.rnn(self.lstm_cell, [self.l3_flat], dtype=tf.float32)

      self.q = affine(tf.squeeze(self.l4), self.w['q_w'], self.w['q_b'])
      self.q_action = tf.argmax(self.q, dimension=1)

      q_summary = []
      avg_q = tf.reduce_mean(self.q, 0)
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
      self.q_summary = tf.merge_summary(q_summary, 'q_summary')

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self._target_s_t = tf.placeholder('float32',
            [None, self.sequence_length, self.screen_height, self.screen_width, self.history_length], name='target_s_t')
      else:
        self._target_s_t = tf.placeholder('float32',
            [None, self.sequence_length, self.history_length, self.screen_height, self.screen_width], name='target_s_t')

      self.t_w['l1_w'], self.t_w['l1_b'] = conv2dParams(self.history_length, 32, [8, 8], initializer, name='target_l1')
      self.t_w['l2_w'], self.t_w['l2_b'] = conv2dParams(32, 64, [4, 4], initializer, name='target_l2')
      self.t_w['l3_w'], self.t_w['l3_b'] = conv2dParams(64, 64, [3, 3], initializer, name='target_l3')

      target_seq = []

      for i in range(self.sequence_length):
        self.target_l1 = conv2dOut(self._target_s_t[:, i, :, :, :], self.t_w['l1_w'], self.t_w['l1_b'], [4, 4], self.cnn_format)
        self.target_l2 = conv2dOut(self.target_l1, self.t_w['l2_w'], self.t_w['l2_b'], [2, 2], self.cnn_format)
        self.target_l3 = conv2dOut(self.target_l2, self.t_w['l3_w'], self.t_w['l3_b'], [1, 1], self.cnn_format)

        shape = self.target_l3.get_shape().as_list()
        self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
        target_seq.append(self.target_l3_flat)
   
      # hist_sequence shape: (sequence_length, batch_size, l3_flattened) 
      target_hist_sequence = tf.pack(target_seq)
      # Reshaping to (sequence_length * batch_size, l3_flattened)
      target_hist_sequence = tf.reshape(target_hist_sequence, [-1, target_hist_sequence.get_shape().as_list()[2]]) 
      # Split to get a list of 'sequence_length' tensors of shape (batch_size, l3_flattened)
      target_hist_sequence = tf.split(0, self.sequence_length, target_hist_sequence)
      # Define lstm cell 
      self.target_lstm_cell = rnn_cell.BasicLSTMCell(512, state_is_tuple=True)
      # list of length `sequence_length` of Tensors of size (?, 512)
      self.target_l4, target_state = tf.nn.rnn(self.target_lstm_cell, target_hist_sequence, dtype=tf.float32)
      
      # throw out sequence elements less than init_sequence_length 
      self.target_l4 = self.target_l4[self.init_sequence_length:]

      target_q_values = []
      target_q_actions = []

      self.t_w['q_w'], self.t_w['q_b'] = linearParams(512, self.env.action_size, name='t_q')

      for i in range(self.truncated_seq_length):
        target_q = affine(self.target_l4[i], self.t_w['q_w'], self.t_w['q_b'])
        target_q_action = tf.argmax(target_q, dimension=1)
        target_q_values.append(target_q)
        target_q_actions.append(target_q_action)

      self.target_q_values = tf.pack(target_q_values, axis=1)
      self.target_q_actions = tf.pack(target_q_actions, axis=1)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self._target_q_values = tf.placeholder('float32', [None, self.truncated_seq_length], name='target_q_values')
      self._actions = tf.placeholder('int64', [None, self.truncated_seq_length], name='actions')

      actions_one_hot = tf.one_hot(self._actions, self.env.action_size, 1.0, 0.0, name='actions_one_hot')
      q_acted = tf.reduce_sum(self.q_values * actions_one_hot, reduction_indices=2, name='q_values_acted')

      self.delta = self._target_q_values - q_acted
      self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

      self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    if not self.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print "="*30
      print " [%d] Best reward : %d" % (best_idx, best_reward)
      print "="*30

    if not self.display:
      self.env.env.monitor.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')

import numpy as np
import tensorflow as tf
import collections
import copy
import multiprocessing
import os
import re
import threading


class A3C(object):

  def __init__(self,
               env,
               model,
               max_rollout_length=20,
               discount_factor=0.99,
               advantage_lambda=0.98,
               value_weight=1.0,
               entropy_weight=0.01,
               optimizer=None,
               model_dir=None):
    self._env = env
    self.model = model
    self.max_rollout_length = max_rollout_length
    self.discount_factor = discount_factor
    self.advantage_lambda = advantage_lambda
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    if optimizer is None:
      self._optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    else:
      self._optimizer = optimizer
    self.states, self.V, self.Q = self.model.build(scope='global')
    self.rewards, self.advantages, self.actions, loss = self._add_loss(self.V, self.Q, scope='global')
    with self.model.graph.as_default():
      self.session = tf.Session(graph=self.model.graph)
      self.session.run(tf.global_variables_initializer())
    
  def _add_loss(self, V, Q, scope='default'):
    with self.model.graph.as_default():
      with tf.name_scope(scope):
        rewards = tf.placeholder(tf.float32, shape=(None,))
        advantages = tf.placeholder(tf.float32, shape=(None,))
        actions = tf.placeholder(tf.float32, shape=(None, self.model.n_actions))
        
        prob = Q + np.finfo(np.float32).eps
        log_prob = tf.log(prob)
        policy_loss = -tf.reduce_mean(advantages * tf.reduce_sum(actions * log_prob, axis=1))
        value_loss = tf.reduce_mean(tf.square(rewards - V))
        entropy = -tf.reduce_mean(tf.reduce_sum(prob * log_prob, axis=1))
        loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy
        return rewards, advantages, actions, loss

  def fit(self,
          total_steps,
          max_checkpoints_to_keep=5,
          checkpoint_interval=600,
          restore=False):
    with self.model.graph.as_default():
      step_count = [0]
      workers = []
      threads = []
      for i in range(multiprocessing.cpu_count()):
        workers.append(_Worker(self, i))
      self.session.run(tf.global_variables_initializer())
      if restore:
        self.restore()
      for worker in workers:
        thread = threading.Thread(
            name=worker.scope,
            target=lambda: worker.run(step_count, total_steps))
        threads.append(thread)
        thread.start()
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
      saver = tf.train.Saver(variables, max_to_keep=max_checkpoints_to_keep)
      checkpoint_index = 0
      while True:
        threads = [t for t in threads if t.isAlive()]
        if len(threads) > 0:
          threads[0].join(checkpoint_interval)
        checkpoint_index += 1
        saver.save(
            self.session, self._graph.save_file, global_step=checkpoint_index)
        if len(threads) == 0:
          break

  def predict(self, state):
    with self.model.graph.as_default():
      feed_dict = {self.states: self.model.process_states(state)}
      tensors = [self.Q, self.V]
      results = self.session.run(tensors, feed_dict=feed_dict)
      return results

  def select_action(self,
                    state,
                    deterministic=False):
    Q, V = self.predict(state)
    Q = np.exp(Q - np.max(Q))
    Q = Q / Q.sum(axis=1)
    if deterministic:
      return Q[0].argmax()
    else:
      return np.random.choice(np.arange(self.model.n_actions), p=Q[0])


class _Worker(object):
  """A Worker object is created for each training thread."""

  def __init__(self, a3c, index):
    self.a3c = a3c
    self.index = index
    self.scope = 'worker%d' % index
    self.env = copy.deepcopy(a3c._env)
    self.env.reset()
    self.states, self.V, self.Q = self.a3c.model.build(scope=self.scope)
    self.rewards, self.advantages, self.actions, self.loss = self.a3c._add_loss(self.V, self.Q, scope=self.scope)
    
    with self.a3c.model.graph.as_default():
      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope)
      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      'global')
      gradients = tf.gradients(self.loss, local_vars)
      grads_and_vars = list(zip(gradients, global_vars))
      self.train_op = self.a3c._optimizer.apply_gradients(grads_and_vars)
      
      self.update_local_variables = tf.group(
          * [tf.assign(v1, v2) for v1, v2 in zip(local_vars, global_vars)])
      self.global_step = self.a3c.model. get_global_step() #???

  def run(self, step_count, total_steps):
     with self.a3c.model.graph.as_default():
      while step_count < total_steps:
        self.a3c.model.session.run(self.update_local_variables)
        states, actions, rewards, values = self.create_rollout()
        self.process_rollout(states, actions, rewards, values, step_count)
        step_count += len(actions)

  def create_rollout(self):
    n_actions = self.a3c.model.n_actions
    session = self.a3c.model.session
    states = []
    actions = []
    rewards = []
    values = []

    # Generate the rollout.

    for i in range(self.a3c.max_rollout_length):
      if self.env.terminated:
        break
      state = self.env.state
      states.append(state)
      feed_dict = {self.states: self.model.process_states(state)}
      results = session.run([self.Q, self.V], feed_dict=feed_dict)
      probabilities, value = results
      action = np.random.choice(np.arange(n_actions), p=probabilities[0])
      actions.append(action)
      values.append(float(value))
      rewards.append(self.env.step(action))

    # Compute an estimate of the reward for the rest of the episode.

    if not self.env.terminated:
      feed_dict = {self.states: self.model.process_states(self.env.state)}
      final_value = self.a3c.discount_factor * float(session.run(self.V, feed_dict))
    else:
      final_value = 0.0
    values.append(final_value)
    if self.env.terminated:
      self.env.reset()
      
    return states, actions, np.array(
        rewards, dtype=np.float32), np.array(
            values, dtype=np.float32)

  def process_rollout(self, states, actions, rewards, values,
                      initial_rnn_states, step_count):
    """Train the network based on a rollout."""

    # Compute the discounted rewards and advantages.

    discounted_rewards = rewards.copy()
    discounted_rewards[-1] += values[-1]
    advantages = rewards - values[:-1] + self.a3c.discount_factor * np.array(
        values[1:])
    for j in range(len(rewards) - 1, 0, -1):
      discounted_rewards[j -
                         1] += self.a3c.discount_factor * discounted_rewards[j]
      advantages[
          j -
          1] += self.a3c.discount_factor * self.a3c.advantage_lambda * advantages[
              j]

    # Convert the actions to one-hot.

    n_actions = self.env.n_actions
    actions_matrix = []
    for action in actions:
      a = np.zeros(n_actions)
      a[action] = 1.0
      actions_matrix.append(a)

    # Rearrange the states into the proper set of arrays.

    if self.a3c._state_is_list:
      state_arrays = [[] for i in range(len(self.features))]
      for state in states:
        for j in range(len(state)):
          state_arrays[j].append(state[j])
    else:
      state_arrays = [states]

    # Build the feed dict and apply gradients.

    feed_dict = {}
    for placeholder, value in zip(self.graph.rnn_initial_states,
                                  initial_rnn_states):
      feed_dict[placeholder] = value
    for f, s in zip(self.features, state_arrays):
      feed_dict[f.out_tensor] = s
    feed_dict[self.rewards.out_tensor] = discounted_rewards
    feed_dict[self.actions.out_tensor] = actions_matrix
    feed_dict[self.advantages.out_tensor] = advantages
    feed_dict[self.global_step] = step_count
    self.a3c.session.run(self.train_op, feed_dict=feed_dict)
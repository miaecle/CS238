#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:27:33 2017

@author: zqwu
"""
from __future__ import division
import numpy as np
import tensorflow as tf
import os
from sklearn.externals import joblib 
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

class DuelNetwork(object):
  """ 
  Main Model, using a dueling network structure.
  """
  def __init__(self, len_states=10, n_actions=14, batch_size=32, lr=0.001):
    self.len_states = len_states
    self.n_actions = n_actions # Number of actions
    self.batch_size = batch_size
    self.lr = lr # learning rate
    self.initialize_featurizer()
    self.g = tf.Graph()
    with self.g.as_default():
      self._build_model()
      self._add_train_cost()
      self.sess = tf.Session(graph=self.g)
      self.sess.run(tf.global_variables_initializer())
  
  def save(self, model_path, global_step=1):
    """ Save model """
    with self.g.as_default():
      saver = tf.train.Saver()
      saver.save(self.sess, os.path.join(model_path, "model"), global_step)
    joblib.dump(self.scaler, os.path.join(model_path, "scaler.joblib"))
    joblib.dump(self.featurizer, os.path.join(model_path, "featurizer.joblib"))
      
  def restore(self, model_path, global_step=1):
    """ Restore model """
    with self.g.as_default():
      saver = tf.train.Saver()
      saver.restore(self.sess, os.path.join(model_path, "model-") + str(global_step))
    self.scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
    self.featurizer = joblib.load(os.path.join(model_path, "featurizer.joblib"))

  def initialize_featurizer(self):
    observation_examples = np.array([np.random.uniform(-np.pi, 
                                                       np.pi, 
                                                       (self.len_states,)) 
   for x in range(10000)])
    self.scaler = sklearn.preprocessing.StandardScaler()
    self.scaler.fit(observation_examples) # Normalization

    self.featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))])
    self.featurizer.fit(self.scaler.transform(observation_examples)) # RBF featurizer
  
  def process_states(self, s):
    # This function featurize states(batch_size * 2)  into inputs(batch_size * 400)
    assert len(s.shape) == 2
    assert s.shape[1] == self.len_states
    scaled = self.scaler.transform(s)
    featurized = self.featurizer.transform(scaled)
    return featurized

  def _build_model(self):
    # Inputs: batch_size * 400
    self.state = tf.placeholder(tf.float32, shape=(None, 400))
    # First hidden layer: batch_size * 10
    W1 = tf.Variable(np.random.normal(0, 0.02, (400, 10)), dtype=tf.float32)
    b1 = tf.Variable(np.zeros((10,))+0.01, dtype=tf.float32)
    hidden1 = tf.nn.tanh(tf.matmul(self.state, W1) + b1)
    # State values: batch_size * 1
    W2 = tf.Variable(np.random.normal(0, 0.02, (10, 1)), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,))+0.01, dtype=tf.float32)
    self.V = tf.matmul(hidden1, W2) + b2
    # Action values: batch_size * n_actions
    W3 = tf.Variable(np.random.normal(0, 0.02, (10, self.n_actions)), dtype=tf.float32)
    b3 = tf.Variable(np.zeros((self.n_actions,))+0.01, dtype=tf.float32)
    self.A = tf.matmul(hidden1, W3) + b3
    
    # State-action values: batch_size * n_actions
    self.Q_prediction = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
    
  def _add_train_cost(self):
    # Target Q values
    self.Q_label = tf.placeholder(tf.float32, shape=(None, self.n_actions))
    # Minimize L2 loss
    self.loss = tf.reduce_sum(tf.nn.l2_loss(self.Q_prediction - self.Q_label))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.train_op = self.optimizer.minimize(self.loss)
    
  def _get_Q(self, states):
    # Extract all actions values for states
    s = np.reshape(np.array(states, dtype=float), (-1, self.len_states))
    feed_dict = {
        self.state: self.process_states(s)
    }
    return self.sess.run(self.Q_prediction, feed_dict=feed_dict)
  
  def update_Q(self, states, target):
    # Update model variables
    s = np.reshape(np.array(states, dtype=float), (-1, self.len_states))
    feed_dict = {
        self.state: self.process_states(s),
        self.Q_label: target
    }
    self.sess.run(self.train_op, feed_dict=feed_dict)
  
  def ChooseAction(self, states, mode='epsilon', epsilon=0.9):
    # Using epsilon-greedy to choose actions
    Q_actions = self._get_Q(states)
    if mode =='epsilon':
      choice = np.random.rand(Q_actions.shape[0], 1) < epsilon
      best = np.expand_dims(np.argmax(Q_actions, axis=1), 1)
      rand = np.random.randint(0, Q_actions.shape[1], (Q_actions.shape[0], 1))
      a = choice*1*best + (1 - choice*1)*rand
    elif mode == 'eval':
      a = np.expand_dims(np.argmax(Q_actions, axis=1), 1)
    return a
  

  
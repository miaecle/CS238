#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:03:38 2017

@author: zqwu
"""

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
  def __init__(self, len_states=10, n_actions=14):
    self.len_states = len_states
    self.n_actions = n_actions # Number of actions
    self.initialize_featurizer()
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
  
  def save(self, model_path, global_step=1):
    joblib.dump(self.scaler, os.path.join(model_path, "scaler.joblib"))
    joblib.dump(self.featurizer, os.path.join(model_path, "featurizer.joblib"))
      
  def restore(self, model_path, global_step=1):
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
    s = np.reshape(s, (-1, self.len_states))
    scaled = self.scaler.transform(s)
    featurized = self.featurizer.transform(scaled)
    return featurized
    
  def build(self, scope="default"):
    with self.graph.as_default():
      with tf.name_scope(scope):
        # Inputs: batch_size * 400
        states = tf.placeholder(tf.float32, shape=(None, 400), name='states')
        # First hidden layer: batch_size * 10
        W1 = tf.Variable(np.random.normal(0, 0.02, (400, 10)), dtype=tf.float32)
        b1 = tf.Variable(np.zeros((10,))+0.01, dtype=tf.float32)
        hidden1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
        # State values: batch_size * 1
        W2 = tf.Variable(np.random.normal(0, 0.02, (10, 1)), dtype=tf.float32)
        b2 = tf.Variable(np.zeros((1,))+0.01, dtype=tf.float32)
        V = tf.add(tf.matmul(hidden1, W2), b2, name='V')
        # Action values: batch_size * n_actions
        W3 = tf.Variable(np.random.normal(0, 0.02, (10, self.n_actions)), dtype=tf.float32)
        b3 = tf.Variable(np.zeros((self.n_actions,))+0.01, dtype=tf.float32)
        A = tf.add(tf.matmul(hidden1, W3), b3, name='A')
        Q = tf.add(V, (A - tf.reduce_mean(A, axis=1, keep_dims=True)), name='Q')
    return states, V, Q
    
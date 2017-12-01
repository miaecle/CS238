#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:02:58 2017

@author: zqwu
"""

import os
os.chdir(os.environ['HOME'] + '/cs238/CS238')
from sawyer import Sawyer
from DuelNetwork import DuelNetwork
import numpy as np
import tensorflow as tf

current_dir = os.getcwd()
urdfFile = os.path.join(current_dir, "rethink/sawyer_description/urdf/sawyer_no_base.urdf")
agent = Sawyer(urdfFile)

model_path = os.path.join(current_dir, "DN")
model_current = DuelNetwork(lr=0.001)
model_old = DuelNetwork(lr=0.001)
model_current.save(model_path, 0)
model_old.restore(model_path, 0)
  
def train(model_current, model_old, memory, batch_size=32, gamma=0.95):
  # Training on memory
  memory_size = len(memory[1])
  inds = np.random.randint(0, memory_size, (batch_size,))
  
  # Load random batch_size samples from memory
  states = [memory[0][ind] for ind in inds]
  actions = np.array(memory[1])[inds]
  rewards = np.array(memory[2])[inds]
  states_ = [memory[3][ind] for ind in inds]

  # Calculate current Q values
  Q_target = model_current._get_Q(states)

  # Calculate target Q values
  Q_new = gamma * np.max(model_old._get_Q(states_), axis=1)
  Q_update = rewards + (1-(rewards > 0))*Q_new
  Q_target[np.arange(batch_size), actions] = Q_update
  
  # Update
  model_current.update_Q(states, Q_target)

# Reset memory
active_memory = [[], [], [], []]

n_episodes = 1000
for i_episode in range(n_episodes):
  s = agent.reset()
  
  done = False
  r_total = 0
  step_count = 0
  while not done:
    step_count += 1
    a = model_current.ChooseAction([s])
    flag = agent.move(a)
    s_ = agent.state
    
    dis = agent.distance(agent.getTargetPosition(), agent.getEFFPosition())
    r = -10*(1 - flag) - dis
    
    if (dis < 0.1):
        done = True
        r = r + 1000
        
    r_total = r_total + r
    
    # Record step into memory
    active_memory[0].append(s)
    active_memory[1].append(int(a))
    active_memory[2].append(r)
    active_memory[3].append(s_)
    s = s_
    
    # Update model_current
    train(model_current, model_old, active_memory, batch_size=128)
    
    # Keep the most recent memory
    active_memory = [active_memory[0][-10000:], 
                     active_memory[1][-10000:], 
                     active_memory[2][-10000:], 
                     active_memory[3][-10000:]]
    if step_count%100 == 0:
        model_current.save(model_path, i_episode)
        model_old.restore(model_path, i_episode)
    if step_count > 1000:
        break
    
  print("Episode: {}, reward: {}.".format(i_episode, r_total))
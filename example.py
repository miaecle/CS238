#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:52:10 2017

@author: zqwu
"""

import os
os.chdir(os.environ['HOME'] + '/cs238/CS238')
from a3c import A3C, _Worker
from model_example import DuelNetwork
from sawyer import Sawyer

current_dir = os.getcwd()
urdfFile = os.path.join(current_dir, "rethink/sawyer_description/urdf/sawyer_no_base.urdf")
env = Sawyer(urdfFile)
model = DuelNetwork()

alg = A3C(env, model)
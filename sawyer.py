import pybullet as p
import os
import numpy as np
import time

class Sawyer(object):
    def __init__(self, urdfFile, GUI=False, gravity=None, timeStep=None, animate=False):
        
        self.urdfFile = urdfFile
        self.eef_link_id = 6
        self.jids = [0, 1, 2, 3, 4, 5, 6]
        self.n_actions = 7
        self.GUI = GUI
        
        self.gravity = gravity
        if self.gravity is None:
            self.gravity = (0, 0, 0)
        assert len(self.gravity) == 3
        
        self.timeStep = timeStep
        if self.timeStep is None:
            self.timeStep = 0.01
            
        self.animate = animate
        self.connect()
        self.setup(self.urdfFile)
        #self.reset()
        self.terminated = False
        
    def connect(self):
        if self.GUI:
            self.physicsClientId = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            self.physicsClientId = p.connect(p.DIRECT)
    
    def disconnect(self):
        p.disconnect(physicsClientId=self.physicsClientId)

    def setup(self, urdfFile):    
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2], physicsClientId=self.physicsClientId)
        p.setTimeStep(self.timeStep, physicsClientId=self.physicsClientId)
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClientId)
        self.cubeId = p.loadURDF('cube_small.urdf', [0.9,0.2,0.1], physicsClientId=self.physicsClientId)
        self.sawyerId = p.loadURDF(urdfFile, useFixedBase=1,flags=2, physicsClientId=self.physicsClientId)

    def resetPos(self, val):
        assert len(val) == len(self.jids), "Number of inputs not matching"
        for i, jid in enumerate(self.jids):
            p.resetJointState(self.sawyerId, 
                              jid, 
                              targetValue=val[i], 
                              targetVelocity=0.,
                              physicsClientId=self.physicsClientId)

    def readQ(self):
        jointStates = p.getJointStates(self.sawyerId, self.jids, physicsClientId=self.physicsClientId)
        q = [jointState[0] for jointState in jointStates]
        return q

    def getTargetJointPosition(self, xyz_pos):
        target = p.calculateInverseKinematics(self.sawyerId, 
                                              self.eef_link_id, 
                                              xyz_pos,
                                              physicsClientId=self.physicsClientId)    
        return target

    def getTargetPosition(self):
        return p.getBasePositionAndOrientation(self.cubeId, 
                                               physicsClientId=self.physicsClientId)[0]

    def getEFFPosition(self):
        return p.getLinkState(self.sawyerId, self.eef_link_id, physicsClientId=self.physicsClientId)[0]

    def distance(self, a, b):
        assert len(a) == len(b)
        return np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
        
    def moveTo(self, joint_position, thresh=0.01):
        assert not self.terminated
        loopCount = 0
        err = thresh
        assert len(joint_position) == len(self.jids), "Number of inputs not matching"
    
        p.setJointMotorControlArray(self.sawyerId,
                                    jointIndices=self.jids,
                                    controlMode=p.POSITION_CONTROL,
                                    physicsClientId=self.physicsClientId,
                                    targetPositions=joint_position)    
        
        trace = []
        flag = True
        done = False
        while(err>=thresh):
            current_joints = self.readQ()
            trace.append(current_joints)
            trace = trace[-100:]
            err = self.distance(current_joints, joint_position)
            if self.animate:
                time.sleep(0.01)
            p.stepSimulation(physicsClientId=self.physicsClientId)
            loopCount += 1
            if (len(trace) == 100 and (self.distance(trace[0], trace[-1]) < 0.0001*thresh)) or loopCount>1e3:
                flag = False
                break
        if self.distance(self.getEFFPosition(), self.getTargetPosition()) < thresh:
            done = True
        return self.getEFFPosition(), done

    def move(self, action):
        assert not self.terminated
        movement = np.zeros((14,))
        movement[int(action)] = 1
        movement = np.reshape(movement, (7, 2))
        movement = np.squeeze(np.matmul(movement, np.array([[0.1], [-0.1]])), axis=-1)
        pos = self.readQ()
        pos_ = np.array(pos) + movement
        eef_pos, flag = self.moveTo(pos_)
        
        return flag
    
    @property
    def state(self):
        pos = np.array(self.readQ())
        dis = np.array(self.getEFFPosition()) - np.array(self.getTargetPosition())
        return np.concatenate([pos, dis])
    
    def step(self, action, state=None):
        assert not self.terminated
        if state is None:
            state = self.state
        self.resetPos(state[:7])
        done = self.move(action)
        reward = 100*done
        if done:
            self.terminated = True
        return reward
    
    def reset(self):
        self.resetPos([0.]*7)
        self.terminated = False
        return self.state
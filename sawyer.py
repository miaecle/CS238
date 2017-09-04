import pybullet as p
import os
import ctypes
from SawyerController.SawyerController import SawyerController as SC
import time
from time import sleep




#Global Variables
#numOfJoints = range(p.getNumJoints(sawyerId))
jids = [1,3,4,5,6,7,8]
#jids = [5,10,11,12,13,15,18]
sawyerId = None     #Call setup function to load this variable
physicsClientId = None     #Call connect function to load this variable
sc = SC('sawyer.urdf')

def connect():
    global physicsClientId
    physicsClientId = p.connect(p.GUI)#or p.DIRECT for non-graphical version

def disconnect():
    p.disconnect()

#gravity must have len of 3
def setup(gravity, timeStep, urdfFile):
    global sawyerId
    #TODO: Insert assert statement for len of gravity
    p.setGravity(gravity[0],gravity[1],gravity[2])
    p.setTimeStep(timeStep)
    sawyerId = p.loadURDF(urdfFile, useFixedBase = 1)
    for i in range (p.getNumJoints(sawyerId,physicsClientId)):
        print(i, p.getJointInfo(sawyerId,i,physicsClientId)[1])




def resetPos(val):
    for jid in jids:
        p.resetJointState(sawyerId, jid, targetValue=val, targetVelocity=0.,
                      physicsClientId=physicsClientId)
    sleep(5)
    #for _ in range(100):
        #p.stepSimulation()

def disableMotors():
    p.setJointMotorControlArray(sawyerId, jointIndices=jids,
                            controlMode=p.VELOCITY_CONTROL,
                            physicsClientId=physicsClientId,
                            forces=(0.,) * len(jids) )

#pos and orn must have len of 3
def setDesPosOrn(pos, orn):
    #TODO: Include assert statement for len of pos and orn
    pos = "{} {} {}".format(pos[0], pos[1], pos[2])
    orn = "{} {} {}".format(orn[0], orn[1], orn[2])


def readQdQ():
    jointStates = p.getJointStates(sawyerId, jids)
    while jointStates is None:
        jointStates = p.getJointStates(sawyerId, jids)

    q = [jointState[0] for jointState in jointStates]
    dq = [jointState[1] for jointState in jointStates]
    sum_dq = sum([abs(val) for val in dq])

    print('dq:::',dq)
    return [q, dq, sum_dq]



def readTorque():
    #Wait for torque value to be ready
    #start = time.time()
    torque_isReady = r.get(PY_TORQUE_READY)
    while torque_isReady != "Ready to Read":
        torque_isReady = r.get(PY_TORQUE_READY)
    #end = time.time()
    #print('seconds used getting torque:',end-start)

    #Read torque values
    tau = r.get(PY_JOINT_TORQUES_COMMANDED_KEY) # Get torque from PID
        #Tell Saw Controller that torque value has been read
    r.set(PY_TORQUE_READY, "Ready to Write")
        #Build float array
    tau_floats = [float(x) for x in tau.split()]
    return tau_floats

def sendTorque(torque):
    #Apply torques
    p.setJointMotorControlArray(sawyerId, jids,
                                controlMode=p.TORQUE_CONTROL,
                                physicsClientId=physicsClientId,
                                forces=torque)

def moveTo(pos,orn):
    loopCount = 0
    #p.setRealTimeSimulation(1,physicsClientId)
    while(1):
        start = time.time()

        # read Q and dQ values from pybullet simulator
        [q,dq,sum_dq] = readQdQ()
        
        loopCount += 1
        if sum_dq<0.03 and loopCount>10:
            print('done moving to desired position')
            break


        # Call SawyerController to calculate torque
        torque= sc.calcTorque(q, dq, pos, orn)
        

    
        # set pybullet simulator joints to apply torque
        sendTorque(torque)
    
        end = time.time()
        print('time of one step:', end - start )
        p.stepSimulation()

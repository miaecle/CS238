import os
os.chdir(os.environ['HOME'] + '/cs238/CS238')
import sawyer
from math import pi
import time

##########################
## Setup Sawyer simulator
##########################
#physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

sawyer.connect()
gravity = (0, 0, -9.8)
timeStep = 0.0001
current_dir = os.getcwd()
urdfFile = os.path.join(current_dir, "rethink/sawyer_description/urdf/sawyer_no_base.urdf")

sawyer.setup(gravity, timeStep, urdfFile)


############################
## Reseting initial position
############################
sawyer.resetPos([0]*7)
#time.sleep(1)


#######################
## input desired position and orientation
#######################
xyz_pos =  sawyer.getTargetPosition()
#pos = sawyer.getTargetJointPosition(xyz_pos)
pos = [-1]*7

###################
### TODO: ./sawyer sawyer.urdf
### Run sawyer.cpp
####################


########################
## START MOVING
########################
while(1):    
    sawyer.moveTo(pos, 0.001)
    if sawyer.checkCollision():
        print("collision!")
        break




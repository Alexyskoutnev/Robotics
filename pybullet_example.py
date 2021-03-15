import pybullet as p
import time
import pybullet_data
import math
import numpy as np
from Robot.RobotArm import RobotArm


#Pybullet Initialization
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("simpleplane.urdf")
FixedBase = True #if fixed no plane is imported
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("Robot/three_link_manipulator.urdf", startPos, startOrientation, useFixedBase = 1)

#get the infomation of each joint
number_of_joints = p.getNumJoints(boxId)
for joint_number in range(number_of_joints):
    info = p.getJointInfo(boxId, joint_number)
    print(info[0], ": ", info[1])


# #Personal Robot (basic_robot.urdf)
# #lengths
# l1 = .5
# l2 = .5
# l3 = .5
# length = [l1,l2,l3]
# #parameters
# theta1 = 0
# theta2 = 0
# theta3 = 0
# # Forward Kinematics Parameters
# thetas = np.array([theta1, theta2, theta3])
# T = np.array([[1, 0,  0,     .7],
#               [0, 1,  0,      .7],
#               [0, 0, 1,      .1],
#               [0, 0,  0,      1]])
# thetalist0 = np.array([.1, .1, .1])
# M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0,0,1, l1+l2+l3], [0,0,0,1]])
# S_list = np.array([[0,0,1,0,0,0], [0,1,0,0.5,0,0], [1,0,0,0,-1,0]]).T
# robot = RobotArm(M_arm= M, slist = S_list)
# T = np.array([[1, 0,  0,     .5],
#               [0, 1,  0,     .5],
#               [0, 0, 1,      .5],
#               [0, 0,  0,      1]])

#Three_Link_manipulator
M = np.array([[1, 0, 0, 1.5],
              [0, 1,  0, 0],
              [0, 0, 1, 0],
              [0, 0,  0, 1]])
S_list = np.array([[0,0,1,0,-1,0], [0,0,1,0,-2,0], [0,0,1,0,-3,0]]).T

#target
T = np.array([[1, 0,  0,    1.5],
               [0, 1,  0,      -2.7],
               [0, 0, 1,      0],
               [0, 0,  0,      1]])

#Initializing the robot with some parameters
three_link = RobotArm(M_arm = M, slist = S_list)
thetalist0 = [0,0,0]
x_pos = three_link.position_update(T, theta_0= thetalist0, learning_rate= .5)

# #trying to use velocity just to control the robot, (doesn't work yet)
# q = thetalist0
# current_pos = three_link.Forward_Kinmatics_Tranformation(q)
# Vs = np.dot(three_link.Adjoint(current_pos), three_link.se3ToVec(three_link.MatrixLog6(np.dot(three_link.TransInv(current_pos), T))))
# Vtarget = np.dot(three_link.Adjoint(T), three_link.se3ToVec(three_link.MatrixLog6(np.dot(three_link.TransInv(current_pos), T))))
# diff = np.subtract(Vtarget, Vs)
# J = three_link.Jacobian_S_Frame(q)
# for i in range (5000):
#     print(q)
#     q = q + np.dot(np.linalg.pinv(J), Vs)
#     p.setJointMotorControl2(boxId, 0, p.VELOCITY_CONTROL, targetVelocity=11)
#     p.setJointMotorControl2(boxId, 1, p.VELOCITY_CONTROL, targetVelocity=5)
#     p.setJointMotorControl2(boxId, 2, p.VELOCITY_CONTROL, targetVelocity=.5)
#     # current_pos = three_link.Forward_Kinmatics_Tranformation(q)
#     # Vs = np.dot(three_link.Adjoint(current_pos), three_link.se3ToVec(three_link.MatrixLog6(np.dot(three_link.TransInv(current_pos), T))))
#     # Vs = np.dot(three_link.Adjoint(current_pos), three_link.se3ToVec(three_link.MatrixLog6(np.dot(three_link.TransInv(current_pos), T))))
#     # diff = np.subtract(Vtarget, Vs)
#     # J = three_link.Jacobian_S_Frame(q)
#     # if (np.linalg.norm(diff) < .1):
#     #     q = np.zeros(0, 3)
#     # else:
#     #     pass
#     time.sleep(.1)


#Displaying the position of the robot using inverse kinematics
for i in range(1000):
        p.setJointMotorControl2(boxId, 0, p.POSITION_CONTROL, targetPosition=x_pos[0])
        p.setJointMotorControl2(boxId, 1, p.POSITION_CONTROL, targetPosition=x_pos[1])
        p.setJointMotorControl2(boxId, 2, p.POSITION_CONTROL, targetPosition=x_pos[2])
        p.stepSimulation()
        time.sleep(1. / 24.)



#Finding orientation and position of the end effector
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
position_end_effector = p.getLinkState(boxId, 3, computeForwardKinematics = 1)
print("=============End Effector Info==============")
print("World Position :", position_end_effector[0])
print("World Orientation :", position_end_effector[1])
p.disconnect()

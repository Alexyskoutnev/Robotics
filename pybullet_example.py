import pybullet as p
import time
import pybullet_data
import math
import numpy as np
from Robot.RobotArm import RobotArm



physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("simpleplane.urdf")
FixedBase = True #if fixed no plane is imported
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("three_link_manipulator.urdf",startPos, startOrientation, useFixedBase = 1)

number_of_joints = p.getNumJoints(boxId)
for joint_number in range(number_of_joints):
    info = p.getJointInfo(boxId, joint_number)
    print(info[0], ": ", info[1])

#
# #Personal Robot
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

three_link = RobotArm(M_arm = M, slist = S_list)
thetalist0 = [0,0,0]
#x_pos = three_link.IKinSpace(T, thetalist0)
x_pos = three_link.position_update(T, theta_0= thetalist0, learning_rate= .5)
print(x_pos)

for i in range (50000):
    pass
    # J = robot.Jacobain_S_Frame(S_list, thetalist0)
    # print(J)
    # X = np.array([0, -1, 1]).T
    #Q_dot = robot.Updated_Joint_Rate(T, thetalist0)
    #Q_dot = robot.IKinSpace(T, thetalist0, ev = .1, eomg= .1)
    #print(Q_dot, "q")
    #thetalist0 = .5*(Q_dot) + thetalist0
    # thetalist0 = thetalist0 + 0.001*robot.update(T, thetalist0)
    # print(thetalist0)
    # p.setJointMotorControl2(boxId, 0, p.VELOCITY_CONTROL, targetVelocity=thetalist0[0])
    # p.setJointMotorControl2(boxId, 1, p.VELOCITY_CONTROL, targetVelocity=thetalist0[1])
    # p.setJointMotorControl2(boxId, 2, p.VELOCITY_CONTROL, targetVelocity=thetalist0[2])
    # thetalist0 = robot.InverseKinematics_NRM()
    #p.stepSimulation()
    #time.sleep(.1)
for i in range(100):
        p.setJointMotorControl2(boxId, 0, p.POSITION_CONTROL, targetPosition=x_pos[0])
        p.setJointMotorControl2(boxId, 1, p.POSITION_CONTROL, targetPosition=x_pos[1])
        p.setJointMotorControl2(boxId, 2, p.POSITION_CONTROL, targetPosition=x_pos[2])
        p.stepSimulation()
        time.sleep(1. / 24.)

# angle = p.addUserDebugParameter('rotate', -0.5, 0.5, 0)
# throttley = p.addUserDebugParameter('y-direction', -5, 5, 0)
# throttlex = p.addUserDebugParameter('x-direction', -5, 5, 0)
#
# while True:
#     user_angle = p.readUserDebugParameter(angle)
#     user_throttley = p.readUserDebugParameter(throttley)
#     user_throttlex = p.readUserDebugParameter(throttlex)
#     p.setJointMotorControl2(boxId, 0, p.VELOCITY_CONTROL, targetVelocity=user_angle)
#     p.setJointMotorControl2(boxId, 1, p.VELOCITY_CONTROL, targetVelocity=user_throttley)
#     p.setJointMotorControl2(boxId, 2, p.VELOCITY_CONTROL, targetVelocity=user_throttlex)
#     p.stepSimulation()

# p.setJointMotorControl2(boxId, 0, p.POSITION_CONTROL, targetPosition = 0)
# p.setJointMotorControl2(boxId, 1, p.POSITION_CONTROL, targetPosition = np.pi/4)
# p.setJointMotorControl2(boxId, 2, p.POSITION_CONTROL, targetPosition = np.pi/4)



cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# joint_info_top1 = p.getJointState(boxId, 0)
# joint_info_top2 = p.getJointState(boxId, 1)
# joint_info_top3 = p.getJointState(boxId, 2)
position_end_effector = p.getLinkState(boxId, 3, computeForwardKinematics = 1)
# print(cubePos,cubeOrn)
# print(joint_info_top1)
# print(joint_info_top2)
# print(joint_info_top3)
print("=============End Effector Info==============")
print("World Position :", position_end_effector[0])
print("World Orientation :", position_end_effector[1])
p.disconnect()

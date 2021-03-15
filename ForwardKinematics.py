import numpy as np
import pybullet as p
import time
import math

# p.connect(p.GUI)
# robotId = p.loadURDF("basic_robot.urdf")
#
#
#
# axis = list()
# number_of_joints = p.getNumJoints(robotId)
# for joint_number in range(number_of_joints):
#     info = p.getJointInfo(robotId, joint_number)
#     print(info[0], ": ", info[1])
#     axis.append(info[13])
#     print(info)

def NearZero(z):
    """Determines whether a scalar is small enough to be treated as zero
    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise
    Example Input:
        z = -1e-7
    Output:
        True
    """
    return abs(z) < 1e-6

def VecToso3(omg):
    return np.array([[0, -omg[2], omg[1]], [omg[2], 0, -omg[0]], [-omg[0], omg[0], 0]])

def so3ToVec(so3mat):
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def VecTose3(V):
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]], np.zeros((1, 4))]

def se3ToVec(se3):
    return np.r_[[se3[2][1], se3[0][2], se3[1][0]], [se3[0][3], se3[1][3], se3[2][3]]]

def TransToRp(T):
    """

    :param T: Receives a T Matrix
    :return:  Returns Rotation and Position Matrix
    """
    return T[0:3,0:3], T[0:3,-1]

def MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix
    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R
    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def MatrixExp3(so3):
    """
    :param so3: receives a so3 matrix in exponential representation
    :return: returns a 3x3 matrix exponential
    """
    omgtheta = so3ToVec(so3)
    if np.linalg.norm(omgtheta) < 1e-6:
        return np.eye(3)
    else:
        theta = np.linalg.norm(omgtheta)
        omgmat = so3 / theta
        return np.eye(3) + np.sin(theta) * omgmat + ((1 - np.cos(theta)) * np.dot(omgmat, omgmat))

def TransInv(T):
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def MatrixExp6(se3):
    """
    :param se3: Receives a se3 matrix (4x4)
    :return: Computes a exponential matrix from the se3 matrix
    """
    se3mat = np.array(se3)
    omgtheta = so3ToVec(se3[0:3, 0:3])
    if np.linalg.norm(omgtheta) < 1e-6:
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = np.linalg.norm(omgtheta)
        omgmat = se3[0:3, 0:3] / theta
        return np.r_[np.c_[MatrixExp3(se3[0:3,0:3]), np.dot(((np.eye(3)*theta) + ((1 - np.cos(theta))*omgmat) + ((theta - np.sin(theta))*np.dot(omgmat, omgmat))), (se3[0:3,-1]/theta))], [[0,0,0,1]]]

def MatrixLog6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix
    :param R: A matrix in SE3
    :return: The matrix logarithm of R
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[0,          0,           0,           0]
                  [0,          0, -1.57079633,  2.35619449]
                  [0, 1.57079633,           0,  2.35619449]
                  [0,          0,           0,           0]])
    """
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                     [[0, 0, 0, 0]]]

def Adjoint(T):
    R = T[0:3,0:3]
    P = T[0:3,3]
    Pmat = VecToso3(P)
    return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(Pmat, R), R]]

def Forward_Kinmatics_Tranformation(M, S_list, theta):
    """
    :param M:  Relation Matrix between origin and end point
    :param S_list: The Spatial coordinates in matrix form
    :param theta: The angles that each joint is rotated by
    :return: Transformation matrix
    """
    T = np.array(M)
    for i in range(len(theta)-1, -1, -1):
        T = np.dot(MatrixExp6(VecTose3(np.array(S_list)[:,i] * theta[i])), T)
    return T

def Jacobain_S_Frame(S_list, theta):
    T = np.eye(4)
    J = np.array(S_list).copy()
    for i in range(1, len(theta)):
        T = np.dot(T, MatrixExp6(VecTose3(np.array(S_list[:,i - 1]))*theta[i - 1]))
        J[:,i] = np.dot(Adjoint(T), np.array(S_list[:,i]))
    return J

def SpatialTwist(Jacobian, theta_dot):
    return np.dot(Jacobian, theta_dot)

def InverseKinematic_NRM(S_list, M, T, theta_0, eomg, ev):
    """

    :param S_list: Screw Parameters Matrix
    :param M: Starting Position of Matrix
    :param T:
    :param theta_0: Initial guess
    :param eomg:
    :param ev:
    :return:
    """
    theta_update = theta_0.copy()
    i = 0
    max_i = 10
    T_sb = Forward_Kinmatics_Tranformation(M, S_list, theta_0)
    T_Inv = TransInv(T_sb)
    V_s = np.dot(Adjoint(T_sb), se3ToVec(MatrixLog6(np.dot(T_Inv, T))))
    print(V_s)
    err_1 = np.linalg.norm([V_s[3], V_s[4], V_s[5]])
    err_a = np.linalg.norm([V_s[0], V_s[1], V_s[2]])
    while ((err_1 > eomg and err_a > ev) or i > max_i):
        theta_update = theta_update + np.dot(np.linalg.pinv(Jacobain_S_Frame(S_list, theta_update)), V_s)
        T_sb = Forward_Kinmatics_Tranformation(M, S_list, theta_update)
        T_Inv = TransInv(T)
        V_s = np.dot(Adjoint(T), se3ToVec(MatrixLog6(np.dot(T_Inv, T))))
        err_1 = np.linalg.norm([V_s[3], V_s[4], V_s[5]])
        err_a = np.linalg.norm([V_s[0], V_s[1], V_s[2]])
    return theta_update

Slist = np.array([[0, 0,  1,  4, 0,    0],
                  [0, 0,  0,  0, 1,    0],
                  [0, 0, -1, -6, 0, -0.1]]).T
M = np.array([[-1, 0,  0, 0],
              [ 0, 1,  0, 6],
              [ 0, 0, -1, 2],
              [ 0, 0,  0, 1]])
T = np.array([[0, 1,  0,     -5],
              [1, 0,  0,      4],
              [0, 0, -1, 1.6858],
              [0, 0,  0,      1]])
thetalist0 = np.array([1.5, 2.5, .1])
eomg = 0.01
ev = 0.001

print(InverseKinematic_NRM(Slist, M, T, thetalist0, eomg, ev), "inverse Kinematics")



#lengths
l1 = .5
l2 = .5
l3 = .5
length = [l1,l2,l3]
#parameters
theta1 = 0
theta2 = np.pi/4
theta3 = np.pi/4
# Forward Kinematics Parameters
thetas = np.array([theta1, theta2, theta3])
M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0,0,1, l1+l2+l3], [0,0,0,1]])
# # M = np.array([[-1, 0, 0, .425+.392],
# #               [0, 0, 1, .109+.82],
# #               [0, 1, 0, .89-.95],
#               [0, 0, 0, 1]])
S_list = np.array([[0,0,1,0,0,0], [0,1,0,0,0,-.5], [-1,0,0,0,1,0]]).T
# # Blist = np.array([[0, 0, 1, 0, 0, 0],
# #                   [0, 1, 0, -.89, 0, 0],
# #                   [0, 1, 0, -.89, 0, .425],
# #                   [0,1,0, -.89, 0, .817],
# #                   [0,0,-1,-.109, .817, 0],
# #                   [0,1,0, .95-.89, 0, .425+.392]]).T


# M = np.array([[-1, 0,  0, 0],
#                       [ 0, 1,  0, 6],
#                       [ 0, 0, -1, 2],
#                       [ 0, 0,  0, 1]])
# Slist = np.array([[0, 0,  1,  4, 0,    0],
#                           [0, 0,  0,  0, 1,    0],
#                           [0, 0, -1, -6, 0, -0.1]]).T
# thetalist = np.array([np.pi / 2.0, 3, np.pi])

print("Location and orientation")
print(Forward_Kinmatics_Tranformation(M, S_list, thetas))

M = np.array([[-1, 0, 0, 0],
              [0, 1, 0, 6],
              [0, 0, -1, 2],
              [0, 0, 0, 1]])
Blist = np.array([[0, 0, -1, 2, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0.1]]).T

thetalist = np.array([np.pi / 2.0, 10, np.pi])


print(Forward_Kinmatics_Tranformation(M, Blist, thetalist), "right")


#Jacobian Parameters
thetadot_1 = 1
thetadot_2 = 1000
thetadot_3 = 1

theta_dot = np.array([theta1, theta2, theta3])
M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0,0,1, l1+l2+l3], [0,0,0,1]])
# # M = np.array([[-1, 0, 0, .425+.392],
# #               [0, 0, 1, .109+.82],
# #               [0, 1, 0, .89-.95],
#               [0, 0, 0, 1]])
S_list = np.array([[0,0,1,0,0,0], [0,1,0,-.5,0,0], [-1,0,0,0,1,0]]).T

Space_Jacobain = Jacobain_S_Frame(S_list, thetalist)


print(Jacobain_S_Frame(S_list, thetalist), "Jacobain")
print(SpatialTwist(Space_Jacobain, theta_dot), "Spatial Twist")

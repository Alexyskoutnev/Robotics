import numpy as np
import pybullet as p

class RobotArm:

    def __init__(self, slist, M_arm):
        self.M = np.array(M_arm)
        self.s = np.array(slist)
        self.N_joint = slist.shape[1]
        self.parMaxIteration = 100
        self.parSqrErrBound = .1
        self.GradientGain = .3

    def VecToso3(self, omg):
        """
        Converts a orientation vector to a rotation matrix
        :param omg: 3x1 orientation vector
        :return: 3x3 rotation matrix
        """
        return np.array([[0, -omg[2], omg[1]], [omg[2], 0, -omg[0]], [-omg[0], omg[0], 0]])

    def so3ToVec(self, so3mat):
        """
        Converts a rotation matrix to a orientation vector
        :param so3mat: 3x3 rotation matrix
        :return: 3x1 orientation vector
        """
        return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

    def VecTose3(self, V):
        """
        Converts a spatial vector into a Transformation Matrix
        :param V: 6x1 Spatial Vector
        :return: 4x4 Transformation Matrix
        """
        return np.r_[np.c_[self.VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]], np.zeros((1, 4))]

    def se3ToVec(self, se3):
        """
        Converts a Transformation Matrix into a spatial vector
        :param se3: 4x4 se matrix
        :return: 6x1 matrix
        """
        return np.r_[[se3[2][1], se3[0][2], se3[1][0]], [se3[0][3], se3[1][3], se3[2][3]]]

    def TransToRp(self, T):
        """
        Converts the Transformation Matrix into a position matrix and rotation matrix
        :param T: Receives a T Matrix
        :return:  Returns Rotation and Position Matrix
        """
        return T[0:3, 0:3], T[0:3, -1]

    def Adjoint(self, T):
        """
        Computes the Adjoint of a Matrix
        :param T: Transformation Matrix
        :return: Adjoint of Transformation Matrix
        """
        R = T[0:3, 0:3]
        P = T[0:3, 3]
        Pmat = self.VecToso3(P)
        return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(Pmat, R), R]]

    def MatrixLog3(self, R):
        """
        Computes the matrix logarithm of a rotation matrix
        :param R: A 3x3 rotation matrix
        :return: The matrix logarithm of R
        """
        acosinput = (np.trace(R) - 1) / 2.0
        if acosinput >= 1:
            return np.zeros((3, 3))
        elif acosinput <= -1:
            if not abs(1 + R[2][2]) < 1e-6:
                omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                      * np.array([R[0][2], R[1][2], 1 + R[2][2]])
            elif not abs(1 + R[1][1]) < 1e-6:
                omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                      * np.array([R[0][1], 1 + R[1][1], R[2][1]])
            else:
                omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                      * np.array([1 + R[0][0], R[1][0], R[2][0]])
            return self.VecToso3(np.pi * omg)
        else:
            theta = np.arccos(acosinput)
            return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

    def MatrixExp3(self, so3):
        """
        Computes the matrix exponential of a rotation matrix
        :param so3: receives a so3 matrix in  representation
        :return: returns a 3x3 matrix exponential
        """
        omgtheta = self.so3ToVec(so3)
        if np.linalg.norm(omgtheta) < 1e-6:
            return np.eye(3)
        else:
            theta = np.linalg.norm(omgtheta)
            omgmat = so3 / theta
            return np.eye(3) + np.sin(theta) * omgmat + ((1 - np.cos(theta)) * np.dot(omgmat, omgmat))

    def TransInv(self, T):
        """
        Computes the inverse of the Transformation Matrix
        :param T: 4x4 Transformation Matrix
        :return: 4x4 Inverse Transformation Matrix
        """
        R, p = self.TransToRp(T)
        Rt = np.array(R).T
        return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

    def MatrixExp6(self, se3):
        """
        Computes the Matrix Exponential of a Transformation Matrix
        :param se3: Receives a se3 matrix (4x4)
        :return: Computes a exponential matrix from the se3 matrix
        """
        se3mat = np.array(se3)
        omgtheta = self.so3ToVec(se3[0:3, 0:3])
        if np.linalg.norm(omgtheta) < 1e-6:
            return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
        else:
            theta = np.linalg.norm(omgtheta)
            omgmat = se3[0:3, 0:3] / theta
            return np.r_[np.c_[self.MatrixExp3(se3[0:3, 0:3]), np.dot(((np.eye(3) * theta) + (
                        (1 - np.cos(theta)) * omgmat) + ((theta - np.sin(theta)) * np.dot(omgmat, omgmat))),
                                                                 (se3[0:3, -1] / theta))], [[0, 0, 0, 1]]]

    def MatrixLog6(self, T):
        """Computes the matrix logarithm of a homogeneous transformation matrix
        :param R: A matrix in SE3
        :return: The matrix logarithm of R
        """
        R, p = self.TransToRp(T)
        omgmat = self.MatrixLog3(R)
        if np.array_equal(omgmat, np.zeros((3, 3))):
            return np.r_[np.c_[np.zeros((3, 3)),
                               [T[0][3], T[1][3], T[2][3]]],
                         [[0, 0, 0, 0]]]
        else:
            theta = np.arccos((np.trace(R) - 1) / 2.0)
            return np.r_[np.c_[omgmat,
                               np.dot(np.eye(3) - omgmat / 2.0 \
                                      + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2)
                                      * np.dot(omgmat, omgmat) / theta, [T[0][3], T[1][3], T[2][3]])],[[0, 0, 0, 0]]]

    def Forward_Kinmatics_Tranformation(self, theta):
        """
        Computes the Forward Kinematics of the end effector with respect to the base frame
        :param M:  Relation Matrix between origin and end point
        :param S_list: The Spatial coordinates in matrix form
        :param theta: The angles that each joint is rotated by
        :return: Transformation matrix
        """
        T = np.array(self.M)
        for i in range(len(theta) - 1, -1, -1):
            T = np.dot(self.MatrixExp6(self.VecTose3(np.array(self.s)[:, i] * theta[i])), T)
        return T

    def Jacobian_S_Frame(self, theta):
        """
        Computes the Jacobian with respect to the base frame
        :param theta: joint theta values (1xnumberofjoints)
        :return: 6x6 Jacobian
        """
        T = np.eye(4)
        J = np.array(self.s).copy()
        for i in range(1, len(theta)):
            T = np.dot(T, self.MatrixExp6(self.VecTose3(np.array(self.s[:, i - 1])) * theta[i - 1]))
            J[:, i] = np.dot(self.Adjoint(T), np.array(self.s[:, i]))
        return J

    def InverseKinematics_NRM(self, T, theta_0, eomg, ev):
        i = 0
        max_i = 20
        T_sb = self.Forward_Kinmatics_Tranformation(theta_0)
        T_Inv = self.TransInv(T_sb)
        Vs = np.dot(self.Adjoint(T_sb), self.se3ToVec(self.MatrixLog6(np.dot(T_Inv, T))))
        thetalist = np.array(theta_0.copy())
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
        while err and i < max_i:
            thetalist = thetalist + np.dot(np.linalg.pinv(self.Jacobian_S_Frame(thetalist)), Vs)
            T_sb = self.Forward_Kinmatics_Tranformation(thetalist)
            i += 1
            V_s = np.dot(self.Adjoint(T_sb), self.se3ToVec(self.MatrixLog6(np.dot(self.TransInv(T_sb), T))))
            err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
            print(np.linalg.norm([Vs[0], Vs[1], Vs[2]]) )
            print(np.linalg.norm([Vs[3], Vs[4], Vs[5]]))
        return thetalist

    def position_update(self, T, theta_0, learning_rate):
        """
        Using Gradient Descent to find the joint angle given @T
        :param T: Target Transformation Matrix
        :param theta_0: Initial theta guess
        :param learning_rate: Learning rate
        :return: Joint angle values (1xnumberofjoints)
        """
        i = 0
        #current position of end effector
        T_sb = self.Forward_Kinmatics_Tranformation(theta_0)
        #starting q value
        q = theta_0
        #end effector current position in x-y-z
        p_effector = self.se3ToVec(T_sb)[3:6]
        #goal position in x-y-z
        p_goal = self.se3ToVec(T)[3:6]
        #distance between goal and end effector
        #delta_p_err = abs(self.se3ToVec(T_sb)[3:6] - self.se3ToVec(T)[3:6])
        #maximum number of steps in while loop
        max_iteration = 25
        PosCur = self.se3ToVec(self.Forward_Kinmatics_Tranformation(theta_0))
        PosRef = self.se3ToVec(T)
        vecPosErr = self.calPosErr(PosCur[3:6], PosRef[3:6])
        #While end effect is not within .1 distance of the goal
        print(np.linalg.norm(vecPosErr), "error")
        while(np.linalg.norm(vecPosErr) > self.parSqrErrBound and self.parMaxIteration > i):
            #Computation of linear velocity Jacobian
            J = self.Jacobian_S_Frame(q)[3:6]
            #inverse of velocity Jacobian
            J_inv = np.linalg.pinv(J)
            #Gradient
            delta_p = learning_rate*vecPosErr
            #Using Jacobain to find joint angle gradient
            delta_q = np.matmul(J_inv, delta_p)
            #change in joint angle
            q = q + delta_q
            #currenkt position of end effector in terms of translation matrix now
            T_sb = self.Forward_Kinmatics_Tranformation(q)
            #current position in x-y-z
            p_effector = self.se3ToVec(T_sb)[3:6]
            #distance between end effector and goal
            vecPosErr = self.calPosErr(PosCur[3:6], PosRef[3:6])
            i += 1
        return q

    def calPosErr(self, PosCurVec, PosRefVec):
        err_x = PosCurVec[0] - PosRefVec[0]
        err_y = PosCurVec[1] - PosRefVec[1]
        err_z = PosCurVec[2] - PosRefVec[2]
        return np.array([err_x, err_y, err_z])

    def calAngErr(self, PosCurVec, PosRefVec):
        err_theta_x = PosCurVec[0] - PosRefVec[0]
        err_theta_y = PosCurVec[1] - PosRefVec[1]
        err_theta_z = PosCurVec[2] - PosRefVec[2]
        return np.array([err_theta_x, err_theta_y, err_theta_z])

    def solIK(self, JointCur, PosRef):
        vecJointCur = np.array(JointCur)
        for idx in range(0, self.parMaxIteration):
            J = self.Jacobian_S_Frame(vecJointCur)
            T_sb = self.Forward_Kinmatics_Tranformation(vecJointCur)
            PosCur = self.se3ToVec(self.Forward_Kinmatics_Tranformation(vecJointCur))
            vecPosErr = np.concatenate((self.calAngErr(PosCur[0:3], PosRef[0:3]), self.calPosErr(PosCur[3:6], PosRef[3:6])))
            if np.sum(vecPosErr[3:6] ** 2) < self.parSqrErrBound:
                return vecJointCur.tolist()
            else:
                #vecGradient = np.gradient(V_PosCur) * vecPosErr
                vecGradient = self.GradientGain * vecPosErr
                try:
                    vecJointDif = np.dot(np.linalg.pinv(J), vecGradient)
                except:
                    vecJointDif = np.zeros(7)
                vecJointCur = vecJointCur + vecJointDif
        return vecJointCur



    # def solIK1(self, raJointCur, raPosRef):
    #     vecJointCur = np.array(raJointCur)
    #     for idx in range(0, self.parMaxItration):
    #         matJac, raPosCur = self.calJac(vecJointCur.tolist())
    #         vecPosErr = np.array(calPosErr(raPosCur[0:3], raPosRef[0:3]) + calAngErr(raPosCur[3:6], raPosRef[3:6]))
    #         if np.sum(vecPosErr[0:3] ** 2) + 1000 * np.sum(vecPosErr[3:6] ** 2) < self.parSqrErrBound:
    #             return vecJointCur.tolist()
    #         else:
    #             vecGradient = self.parGradientGain * vecPosErr
    #             try:
    #                 vecJointDif = np.dot(np.linalg.pinv(matJac), vecGradient)
    #             except:
    #                 vecJointDif = np.zeros(7)
    #             vecJointCur = vecJointCur + vecJointDif
    #     return False

    def solRP(self, raJointCur, raVelEE):
        matJac, raPosCur = self.calJac(raJointCur)
        vecVelEE = np.array(raVelEE)
        try:
            vecJointDif = np.dot(np.linalg.pinv(matJac), vecVelEE)
        except:
            vecJointDif = np.zeros(7)
        return vecJointDif

    def Inverse_Kinematics(self, T, thetalist0):
            """
            Computes the robot's Inverse Kinematics using Newton Raphson Method to find the predicted joint value at a given position
            :param T: Target Transformation Matrix
            :param thetalist0: The initial theta guess
            :return: Predicted theta @ Target Transformation Matrix
            """
            #Initial Theta
            thetalist = np.array(thetalist0).copy()
            i = 0
            maxiterations = 20
            Tsb = self.Forward_Kinmatics_Tranformation(thetalist)
            Vs = np.dot(self.Adjoint(Tsb), self.se3ToVec(self.MatrixLog6(np.dot(self.TransInv(Tsb), T))))
            delta_p = abs(self.se3ToVec(Tsb) - self.se3ToVec(T))
            err_d = np.linalg.norm(delta_p)
            while i < maxiterations and err_d > .01:
                thetalist = thetalist + np.dot(np.linalg.pinv(self.Jacobian_S_Frame(thetalist)), Vs)
                i = i + 1
                Tsb = self.Forward_Kinmatics_Tranformation(thetalist)
                Vs = np.dot(self.Adjoint(Tsb), self.se3ToVec(self.MatrixLog6(np.dot(self.TransInv(Tsb), T))))
                delta_p = abs(self.se3ToVec(Tsb) - self.se3ToVec(T))
                err_d = np.linalg.norm(delta_p)
            return thetalist

    # def runCtlrP2P(self):
    #
    #     global flagChangedArm
    #
    #     if flagChangedArm:
    #
    #         flagChangedArm = False
    #         raJointCur = self.model.getCurrentArmJoints(robot.POSITION)
    #         raJointGoal = self.ik.solIK(raJointCur, raCmdArm)
    #
    #         if raJointGoal:
    #             self.trjJointRefP2P = self.ik.genTrj(raJointCur, raJointGoal, valTimeStep=self.timeStep,
    #                                                  lenTrj=self.goalP2P)
    #             self.countP2P = 0
    #
    #     if not self.countP2P == self.goalP2P - 1:
    #         self.countP2P += 1
    #
    #     self.model.setTargetArmJoints(robot.POSITION, self.trjJointRefP2P[self.countP2P])
    #
    #     return True

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
thetalist0 = np.array([1.5, 2.5, 3])
eomg = 0.01
ev = 0.001
robot = RobotArm(M_arm = M, slist = Slist)
print(robot.position_update(T, theta_0= thetalist0, learning_rate= .1), "wrong")

# M = np.array([[1, 0, 0, 1.5],
#               [0, 1,  0, 0],
#               [0, 0, 1, 0],
#               [0, 0,  0, 1]])
# S_list = np.array([[0,0,1,0,-1,0], [0,0,1,0,-2,0], [0,0,1,0,-3,0]]).T
# robot = RobotArm(M_arm = M, slist = S_list)
# Current_Joint_Par = [0,1,.5]
# Target_Position = [1,1, 1]
# Joint_vec = robot.solIK(Current_Joint_Par, Target_Position)
# print(Joint_vec)

#print(np.concatenate((np.array([1, 1, 1]), np.array(Target_Position))))
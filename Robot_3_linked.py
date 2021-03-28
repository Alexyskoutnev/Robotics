import numpy as np


class Robot:

    def __init__(self, s_list = None, t = None):
        self.s = np.array(s_list)
        self.t = np.array(t)
        self.N_joints = self.s.shape[0]
        self.Gradient_Gain = .15
        self.MaxIter = 25

    def Rotation_Matrix(self, axis = None, theta = None):
        """
        Creates a rotation matrix
        :param axis: 1x3 matrix axis coordinates of rotation
        :param theta: Theta amount of rotation
        :return: Rotation Matrix
        """
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_theta_minus_1 = 1 - np.cos(theta)
        ax = axis[0]
        ay = axis[1]
        az = axis[2]

        #row 1
        r00 = ax * ax * c_theta_minus_1 + c_theta
        r01 = ax * ay * c_theta_minus_1 - az * s_theta
        r02 = ax * az * c_theta_minus_1 + ay * s_theta

        # row 2
        r10 = ax * ay * c_theta_minus_1 + az * s_theta
        r11 = ay * ay * c_theta_minus_1 + c_theta
        r12 = ay * az * c_theta_minus_1 - ax * s_theta

        # row 3
        r20 = ax * az * c_theta_minus_1 - ay * s_theta
        r21 = ay * az * c_theta_minus_1 + ax * s_theta
        r22 = az * az * c_theta_minus_1 + c_theta

        return np.r_[np.c_[r00, r01, r02], np.c_[r10, r11, r12], np.c_[r20, r21, r22]]

    def Tranformation_Matrix(self, V_sb, axis, theta):
        """
        Creates a Tranformation Matrix from axis and angle parameter
        :return: Transformation Matrix
        """
        pos_vec = np.array(([V_sb[0]], [V_sb[1]], [V_sb[2]]))
        rot_mat = self.Rotation_Matrix(axis, theta)
        top_mat = np.c_[rot_mat, pos_vec]
        lower_mat = np.array(([[0,0,0,1]]))
        Trans_mat = np.r_[top_mat, lower_mat]
        return Trans_mat

    def solFk(self, theta):
        """
        Returns the position vector of the end effector of the given theta
        :param theta: Joint angles
        :return: Position of End Effector w.r.t base fram
        """
        T_sb = np.array([[0], [0], [0], [1]])
        for idx in range(len(theta) - 1, -1, -1):
            T_sb = np.dot(self.Tranformation_Matrix(self.t[idx], self.s[idx], theta[idx]), T_sb)
        return np.ndarray.flatten(T_sb[0:3])

    def solFk_index(self, theta):
        """
        Returns the position vector of the end effector of the given theta
        :param theta: Joint angles
        :return: Position of End Effector w.r.t base fram
        """
        T_sb = np.eye(4)
        theta = theta.copy()
        for idx in range(0, len(theta)):
            T_sb = np.dot(T_sb, self.Tranformation_Matrix(self.t[idx], self.s[idx], theta[idx]))
            Trans = self.Tranformation_Matrix(self.t[idx], self.s[idx], theta[idx])
        PosCur = np.dot(T_sb, np.array([[0], [0], [0], [1]]))
        return np.ndarray.flatten(PosCur[0:3])

    def Jacobian(self, theta):
        """
        Calculates the linear Jacobian of the system
        :param theta: Joint Angles
        :return: 3xnumberofjoint Jacobian
        """
        PosCur = None
        PosEff = self.solFk(theta)
        for i in range(0 , self.N_joints-1):
            PosCur = self.solFk_index(theta[:i+1])
            delta = np.array(PosEff - PosCur)
            k = np.array([self.s[i][0], self.s[i][1], self.s[i][2]])
            current_column = np.cross(k, delta)
            if (i == 0):
                J = np.array([[current_column[0]], [current_column[1]],[current_column[2]]])
            else:
                J = np.concatenate((J, np.array([[current_column[0]], [current_column[1]],[current_column[2]]])), axis = 1)
        J = np.concatenate((J, np.array([[0],[0],[0]])), axis=1)
        return J

    def solIK(self, theta_0, PosRef):
        """
        Computes the joints angles @ desired (x,y,z) position
        :param theta_0: Initial Guess of Joint Angles
        :param PosRef: Desired Goal Position (x, y, z)
        :return: Joint Angles
        """
        q = theta_0.copy()
        PosRef = PosRef
        PosCur = self.solFk(q)
        delta_p = PosRef - PosCur
        theta_max_step = 0.3
        for idx in range(0, self.MaxIter):
            if (np.linalg.norm(delta_p) < .01):
                return q
            else:
                v_gradient = delta_p * self.Gradient_Gain / np.linalg.norm(delta_p)
                try:
                    J = self.Jacobian(q)
                    vecJointDif = np.dot(np.linalg.pinv(J), v_gradient)
                    q = q + np.clip(vecJointDif, -1 * theta_max_step, theta_max_step)
                except:
                    vecJointDif = np.ones(3)
                q = q + vecJointDif
                PosCur = self.solFk(q)
                delta_p = PosRef - PosCur
        return q


# s_list = np.array([[0, 0, 1], [0, 0, 1], [0,0,1]])
# t = np.array([[0,0,0], [1.5,0,0], [1.5, 0, 0]])
# robot = Robot(s_list, t)
# #print(robot.N_joints)
# #print(robot.Rotation_Matrix([1,0,0], 20))
# #print(robot.Tranformation_Matrix([1,2,3], [1,0,0], 20))
# t_sb_1 = robot.Tranformation_Matrix([1,2,3], [1,0,0], 20)
# #print(robot.solFk([1.57,1.57,0]))
# print(robot.Jacobian([1.57,1.57,0]))
# #def Jacobain
# print(robot.solIK([0.1,0.1,0], [0,2.9,0]), "invers")

import numpy as np
import math
from SAC.common.util import *
from numpy import pi

class A1Config():
    def __init__(self):
        self.fileName = "/a1_description/urdf/a1.urdf"

        self.ground_contact_link = [
            "FL_lower_leg_2_foot_joint",
            "FR_lower_leg_2_foot_joint",
            "RL_lower_leg_2_foot_joint",
            "RR_lower_leg_2_foot_joint",
        ]

        deg2rad = pi/180.0

        self.q_bound_default = dict([
            ("FR_hip_motor_2_chassis_joint", [-0.8028, 0.8028]),  # FR_HipX_joint
            ("FR_upper_leg_2_hip_motor_joint", [-4.1885, 1.0471]),  # FR_HipY_joint
            ("FR_lower_leg_2_upper_leg_joint", [0.9162, 2.6963]),  # FR_Knee_joint
            ("FL_hip_motor_2_chassis_joint", [-0.8028, 0.8028]),  # FL_HipX_joint
            ("FL_upper_leg_2_hip_motor_joint", [-4.1885, 1.0471]),  # FL_HipY_joint
            ("FL_lower_leg_2_upper_leg_joint", [0.9162, 2.6963]),  # FL_Knee_joint
            ("RR_hip_motor_2_chassis_joint", [-0.8028, 0.8028]),  # HR_HipX_joint
            ("RR_upper_leg_2_hip_motor_joint", [-4.1885, 1.0471]),  # HR_HipY_joint
            ("RR_lower_leg_2_upper_leg_joint", [0.9162, 2.6963]),  # HR_Knee_joint
            ("RL_hip_motor_2_chassis_joint", [-0.8028, 0.8028]),  # HL_HipX_joint
            ("RL_upper_leg_2_hip_motor_joint", [-4.1885, 1.0471]),  # HL_HipY_joint
            ("RL_lower_leg_2_upper_leg_joint", [0.9162, 2.6963]),  # HL_Knee_joint
        ])

        self.q_nom_default = dict([
            ("FR_hip_motor_2_chassis_joint", 0),
            ("FR_upper_leg_2_hip_motor_joint", -0.785),
            ("FR_lower_leg_2_upper_leg_joint", 1.57),
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", -0.785),
            ("FL_lower_leg_2_upper_leg_joint", 1.57),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", -0.785),
            ("RR_lower_leg_2_upper_leg_joint", 1.57),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", -0.785),
            ("RL_lower_leg_2_upper_leg_joint", 1.57),
        ])

        # max torque
        self.u_max_default = dict([
            ("FR_hip_motor_2_chassis_joint", 33.5),
            ("FR_upper_leg_2_hip_motor_joint", 33.5),
            ("FR_lower_leg_2_upper_leg_joint", 33.5),
            ("FL_hip_motor_2_chassis_joint", 33.5),
            ("FL_upper_leg_2_hip_motor_joint", 33.5),
            ("FL_lower_leg_2_upper_leg_joint", 33.5),
            ("RR_hip_motor_2_chassis_joint", 33.5),
            ("RR_upper_leg_2_hip_motor_joint", 33.5),
            ("RR_lower_leg_2_upper_leg_joint", 33.5),
            ("RL_hip_motor_2_chassis_joint", 33.5),
            ("RL_upper_leg_2_hip_motor_joint", 33.5),
            ("RL_lower_leg_2_upper_leg_joint", 33.5),
        ])

        self.v_max_default = dict([
            ("FR_hip_motor_2_chassis_joint", 12),
            ("FR_upper_leg_2_hip_motor_joint", 12),
            ("FR_lower_leg_2_upper_leg_joint", 12),
            ("FL_hip_motor_2_chassis_joint", 12),
            ("FL_upper_leg_2_hip_motor_joint", 12),
            ("FL_lower_leg_2_upper_leg_joint", 12),
            ("RR_hip_motor_2_chassis_joint", 12),
            ("RR_upper_leg_2_hip_motor_joint", 12),
            ("RR_lower_leg_2_upper_leg_joint", 12),
            ("RL_hip_motor_2_chassis_joint", 12),
            ("RL_upper_leg_2_hip_motor_joint", 12),
            ("RL_lower_leg_2_upper_leg_joint", 12),
        ])

        self.Kp_default = dict([
            ("FR_hip_motor_2_chassis_joint", 100),
            ("FR_upper_leg_2_hip_motor_joint", 100),
            ("FR_lower_leg_2_upper_leg_joint", 100),
            ("FL_hip_motor_2_chassis_joint", 100),
            ("FL_upper_leg_2_hip_motor_joint", 100),
            ("FL_lower_leg_2_upper_leg_joint", 100),
            ("RR_hip_motor_2_chassis_joint", 100),
            ("RR_upper_leg_2_hip_motor_joint", 100),
            ("RR_lower_leg_2_upper_leg_joint", 100),
            ("RL_hip_motor_2_chassis_joint", 100),
            ("RL_upper_leg_2_hip_motor_joint", 100),
            ("RL_lower_leg_2_upper_leg_joint", 100),
        ])
        self.Kd_default = dict([
            ("FR_hip_motor_2_chassis_joint", 5),
            ("FR_upper_leg_2_hip_motor_joint", 5),
            ("FR_lower_leg_2_upper_leg_joint", 5),
            ("FL_hip_motor_2_chassis_joint", 5),
            ("FL_upper_leg_2_hip_motor_joint", 5),
            ("FL_lower_leg_2_upper_leg_joint", 5),
            ("RR_hip_motor_2_chassis_joint", 5),
            ("RR_upper_leg_2_hip_motor_joint", 5),
            ("RR_lower_leg_2_upper_leg_joint", 5),
            ("RL_hip_motor_2_chassis_joint", 5),
            ("RL_upper_leg_2_hip_motor_joint", 5),
            ("RL_lower_leg_2_upper_leg_joint", 5),
        ])

        self.controlled_joints = [
            "FL_hip_motor_2_chassis_joint",
            "FL_upper_leg_2_hip_motor_joint",
            "FL_lower_leg_2_upper_leg_joint",
            "FR_hip_motor_2_chassis_joint",
            "FR_upper_leg_2_hip_motor_joint",
            "FR_lower_leg_2_upper_leg_joint",
            "RL_hip_motor_2_chassis_joint",
            "RL_upper_leg_2_hip_motor_joint",
            "RL_lower_leg_2_upper_leg_joint",
            "RR_hip_motor_2_chassis_joint",
            "RR_upper_leg_2_hip_motor_joint",
            "RR_lower_leg_2_upper_leg_joint",
            ]

        self.controllable_joints = [
            "FL_hip_motor_2_chassis_joint",
            "FL_upper_leg_2_hip_motor_joint",
            "FL_lower_leg_2_upper_leg_joint",
            "FR_hip_motor_2_chassis_joint",
            "FR_upper_leg_2_hip_motor_joint",
            "FR_lower_leg_2_upper_leg_joint",
            "RL_hip_motor_2_chassis_joint",
            "RL_upper_leg_2_hip_motor_joint",
            "RL_lower_leg_2_upper_leg_joint",
            "RR_hip_motor_2_chassis_joint",
            "RR_upper_leg_2_hip_motor_joint",
            "RR_lower_leg_2_upper_leg_joint",
        ]

        self.base_pos_nom_default = [0, 0, 0.3]
        self.base_orn_nom_default = [0, 0, 0, 1]
        self.base_euler_offset = []

        self.actionNumber = len(self.controlled_joints)
        self.action_bound = np.zeros((2,self.actionNumber))
        for i in range(self.actionNumber):
            joint_name = self.controlled_joints[i]
            self.action_bound[0][i] = self.q_bound_default[joint_name][0]  # lower bound
            self.action_bound[1][i] = self.q_bound_default[joint_name][1]  # upper bound


    #keypose initialization for fall recovery
        self.key_pose = []
        # nominal
        base_pos_nom = [0, 0, 0.3]
        base_orn_nom = euler_to_quat(0, 0, 0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),
            ("FR_upper_leg_2_hip_motor_joint", -0.785),
            ("FR_lower_leg_2_upper_leg_joint", 1.57),
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", -0.785),
            ("FL_lower_leg_2_upper_leg_joint", 1.57),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", -0.785),
            ("RR_lower_leg_2_upper_leg_joint", 1.57),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", -0.785),
            ("RL_lower_leg_2_upper_leg_joint", 1.57),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        # crouch
        base_pos_nom = [0, 0, 0.2]
        base_orn_nom = euler_to_quat(0, 0, 0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),
            ("FR_upper_leg_2_hip_motor_joint", -1.66),
            ("FR_lower_leg_2_upper_leg_joint", 2.6),
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", -1.66),
            ("FL_lower_leg_2_upper_leg_joint", 2.6),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", -1.66),
            ("RR_lower_leg_2_upper_leg_joint", 2.6),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", -1.66),
            ("RL_lower_leg_2_upper_leg_joint", 2.6),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        #back
        base_pos_nom = [0, 0, 0.07]
        base_orn_nom = euler_to_quat(3.14, 0, 0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),
            ("FR_upper_leg_2_hip_motor_joint", -0.74),
            ("FR_lower_leg_2_upper_leg_joint", 1.69),
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", -0.74),
            ("FL_lower_leg_2_upper_leg_joint", 1.69),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", -0.74),
            ("RR_lower_leg_2_upper_leg_joint", 1.69),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", -0.74),
            ("RL_lower_leg_2_upper_leg_joint", 1.69),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        #right
        base_pos_nom = [0, 0, 0.16]
        base_orn_nom = euler_to_quat(1.57, 0, 0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", -0.39),
            ("FR_upper_leg_2_hip_motor_joint", -0.6),
            ("FR_lower_leg_2_upper_leg_joint", 0.9162),
            ("FL_hip_motor_2_chassis_joint", 0.39),
            ("FL_upper_leg_2_hip_motor_joint", -0.3),
            ("FL_lower_leg_2_upper_leg_joint", 0.9162),
            ("RR_hip_motor_2_chassis_joint", -0.39),
            ("RR_upper_leg_2_hip_motor_joint", 0),
            ("RR_lower_leg_2_upper_leg_joint", 0.9162),
            ("RL_hip_motor_2_chassis_joint", 0.39),
            ("RL_upper_leg_2_hip_motor_joint", -0.3),
            ("RL_lower_leg_2_upper_leg_joint", 0.9162),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        #left
        base_pos_nom = [0, 0, 0.16]
        base_orn_nom = euler_to_quat(-1.57, 0, 0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", -0.39),
            ("FR_upper_leg_2_hip_motor_joint", -0.3),
            ("FR_lower_leg_2_upper_leg_joint", 0.9162),
            ("FL_hip_motor_2_chassis_joint", 0.39),
            ("FL_upper_leg_2_hip_motor_joint", -0.6),
            ("FL_lower_leg_2_upper_leg_joint", 0.9162),
            ("RR_hip_motor_2_chassis_joint", -0.39),
            ("RR_upper_leg_2_hip_motor_joint", -0.3),
            ("RR_lower_leg_2_upper_leg_joint", 0.9162),
            ("RL_hip_motor_2_chassis_joint", 0.39),
            ("RL_upper_leg_2_hip_motor_joint", 0),
            ("RL_lower_leg_2_upper_leg_joint", 0.9162),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
# parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0, parentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import math

from utils.filter_array import FilterClass
from utils.JointInterpolateArray import JointTrajectoryInterpolate

from a1_description.a1_gallop_configuration import *
# from obstacle_script.loadObstacles import *

class A1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __del__(self):
        p.disconnect()

    def __init__(self,
                 max_time=16,  # in seconds
                 initial_gap_time=0.05,  # in seconds
                 isEnableSelfCollision=True,
                 renders=True,
                 control_freq=25.0,
                 motor_command_freq=500.0,
                 physics_freq=1000.0, #motor model runs at the physics simulation frequency
                 control_type='custom_PD',
                 logFileName=None,
                 controlled_joints_list=None,
                 Kp = None,
                 Kd = None,
                 action_bound = None, #dictionary of bounds
                 filter_torque=False,
                 interpolation=True,
                 filter_action=False,
                 imitation_weight=0.5,
                 task_weight=0.5,
                 dsr_gait_period=0.5,
                 obstacle=None,
                 ):

        # action settings
        self.filter_action = filter_action
        self.filter_torque = filter_torque
        self.interpolation = interpolation

        self.control_type = control_type
        self.isEnableSelfCollision = isEnableSelfCollision
        self.jointLowerLimit = []
        self.jointUpperLimit = []

        self._p = p
        self._seed()
        self._envStepCounter = 0
        self._renders = renders
        if logFileName is None:
            self._logFileName = os.path.dirname(os.path.realpath(__file__))
        else:
            self._logFileName = logFileName

        if self._renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        #Robot basic configuration
        self.robotConfig = A1Config()

        if action_bound is None:
            self.q_bound_default = self.robotConfig.q_bound_default
        if Kp is None:
            self.Kp = self.robotConfig.Kp_default
        if Kd is None:
            self.Kd = self.robotConfig.Kd_default

        if controlled_joints_list is None:
            self.controlled_joints = self.robotConfig.controlled_joints
        else:
            self.controlled_joints = controlled_joints_list

        # nominal joint configuration
        self.u_max = self.robotConfig.u_max_default
        self.v_max = self.robotConfig.v_max_default
        self.q_nom = self.robotConfig.q_nom_default

        self.q_nom_default = self.robotConfig.q_nom_default

        self.controllable_joints = self.robotConfig.controllable_joints
        self.uncontrolled_joints = [a for a in self.controllable_joints if a not in self.controlled_joints]

        self.nu = len(self.controlled_joints)
        self.r = -1
        self.control_freq = control_freq
        self.motor_command_freq = motor_command_freq
        self.physics_freq = physics_freq
        self._control_loop_skip = int(motor_command_freq/control_freq)
        self._motor_command_loop_skip = int(physics_freq/motor_command_freq)
        self._dt_physics = (1./ self.physics_freq)
        self._dt_motor_command = (1. / self.motor_command_freq)
        self._dt_control = (1./self.control_freq)
        self._dt = self._dt_physics # PD control loop timestep
        self._dt_filter = self._dt_motor_command #filter time step
        self._dt_interpolate = self._dt_control # time period for interpolation
        self.g = 9.81

        self.max_steps = max_time * self.motor_command_freq # PD control loop timestep
        self.initial_gap_steps = initial_gap_time * self.motor_command_freq # Simulation reset timestep in PD control loop

        self.joint_ref_interpolation = list()

        self.jointIdx = dict()
        self.jointNameIdx = dict()

        self.base_pos_nom = np.array(self.robotConfig.base_pos_nom_default)
        self.base_orn_nom = np.array(self.robotConfig.base_orn_nom_default)

        self.base_pos = np.array([0.0,0.0,0.0])

        self.stateNumber = 6+2+3+len(self.controlled_joints)
        self.actionNumber = len(self.controlled_joints)

        observationDim = self.stateNumber
        self._observationDim = observationDim
        self._actionDim = len(self.controlled_joints)

        self.action = np.zeros(self._actionDim)
        self.prev_action = np.zeros(self._actionDim)
        self.control_torque = np.zeros(self._actionDim)

        self.prev_base_pos = np.array([0,0,0])
        self.prev_base_quat = np.array([0,0,0,0])#quat
        self.prev_base_pos_vel = np.array([0,0,0])
        self.prev_base_orn_vel = np.array([0,0,0])
        self.prev_base_pos_vel_yaw = np.array([0,0,0])#past base linear velocity in adjusted yaw frame

        print("observationDim", self._observationDim, "actionDim", self._actionDim)
        self.viewer = None

        #setup joint action bound array
        self.action_bound = np.zeros((2,self.actionNumber))
        for i in range(self.actionNumber):
            joint_name = self.controlled_joints[i]
            self.action_bound[0][i] = self.q_bound_default[joint_name][0]  # lower bound
            self.action_bound[1][i] = self.q_bound_default[joint_name][1]  # upper bound

        self.episode_count = 0

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        #A complete gait cycles is 0.5s
        self.phase_counter = 0
        self.target_gait_period = dsr_gait_period
        self.period = int(self.target_gait_period*self.control_freq) # number of steps for a complete gait cycle

        self.imitation_weight = imitation_weight
        self.task_weight = task_weight

        # Setup Simulation
        self.obstacle = obstacle
        self._setupSimulation()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self, base_pos_nom = None, base_orn_nom = None, fixed_base = False, q_nom = None, g = 9.81, base_vel = None, phase_counter=None):
        self.episode_count+=1

        if phase_counter is None:
            self.phase_counter = np.random.uniform(0, self.period) % self.period  # randomly initialize phase
        else:
            self.phase_counter = phase_counter
        pose = self.referenceStateInitialization()

        if base_pos_nom is None:
            base_pos_nom = pose[0]
        if base_orn_nom is None:
            base_orn_nom = pose[1]
        if q_nom is None:
            q_nom = pose[2]

        self.g = g

        self._setupSimulation(base_pos_nom, base_orn_nom, fixed_base, q_nom, obstacle=self.obstacle)
        self._envStepCounter = 0

        if base_vel is None:
            base_vel = [np.random.uniform(-0.5,2),0,0]
        p.resetBaseVelocity(self.r, linearVelocity=base_vel)

        self._observation = self.getExtendedObservation()

        return np.array(self._observation)

    def getExtendedObservation(self):
        self._observation = self.getFilteredObservation()  # filtered observation
        return self._observation

    def _step(self, action, pelvis_push_force=[0,0,0], frame = ['World', 'base'], force_pos = [0,0,0]):
        index = -1
        if frame[0] == 'Local':
            frame_flag = p.LINK_FRAME
            force_pos = [0,0,0]
        elif frame[0] == 'World':
            frame_flag = p.WORLD_FRAME
            if frame[1] == 'base':
                index = -1
                force_pos, _ = p.getBasePositionAndOrientation(self.r)
            else:
                index = self.jointIdx[frame[1]]
                link_state = p.getLinkState(self.r, index)
                force_pos = link_state[0]
        else:
            frame_flag = p.LINK_FRAME
            force_pos = [0, 0, 0]

        #record state observations for reward computation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        self.prev_base_pos = np.array(base_pos)
        self.prev_base_quat = np.array(base_quat) #quaternion
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.r)
        self.prev_base_pos_vel = np.array(base_pos_vel)
        self.prev_base_orn_vel = np.array(base_orn_vel)

        self.action = action
        self.action = np.clip(self.action, self.action_bound[0], self.action_bound[1]) #clip action
        for n in range(self._actionDim):
            key  = self.controlled_joints[n]
            max = self.q_bound_default[key][1]
            min = self.q_bound_default[key][0]
            self.action[n] = np.clip(self.action[n], min, max) #clip joint range

        self.joint_ref_interpolation.cubic_interpolation_setup(self.prev_action,.0,self.action,.0,self._dt_interpolate)

        for i in range(self._control_loop_skip):
            torque_dict = dict()
            action_interpolate_list = []

            #array operation
            action_interpolate_list = self.joint_ref_interpolation.interpolate(self._dt_motor_command)

            if self.interpolation == True:
                action_target = action_interpolate_list
            else:
                action_target = self.action

            if self.filter_action == True:
                filtered_action = self.action_filter_method.applyFilter(action_target)
                action_target = filtered_action

            if self.control_type == 'custom_PD':
                for j in range(self._motor_command_loop_skip):

                    for n in range(self._actionDim):
                        name = self.controlled_joints[n]
                        joint_state = p.getJointState(self.r, self.jointIdx[name])
                        pos = joint_state[0]
                        vel = joint_state[1]

                        P_u = self.Kp[name] * (action_target[n] - pos)
                        D_u = self.Kd[name] * (0-vel)
                        control_torque = P_u+D_u

                        control_torque = np.clip(control_torque, -self.u_max[name], self.u_max[name])
                        self.control_torque[n] = control_torque #needed for reward calculation

                        torque = control_torque
                        torque_dict.update({name: torque})
                    self.setTorqueControlwithVelocityConstrain(torque_dict)

                    p.applyExternalForce(self.r, index, forceObj=pelvis_push_force, posObj=force_pos, flags=frame_flag)
                    p.stepSimulation()

            else:
                for j in range(self._motor_command_loop_skip):
                    for n in range(self._actionDim):
                        name = self.controlled_joints[n]
                        joint_state = p.getJointState(self.r, self.jointIdx[name])
                        pos = joint_state[0]
                        vel = joint_state[1]
                        action_interpolate = action_interpolate_list[n]
                        P_u = self.Kp[name] * (action_interpolate - pos)
                        D_u = self.Kd[name] * (0-vel)
                        control_torque = P_u+D_u

                        control_torque = np.clip(control_torque, -self.u_max[name], self.u_max[name])
                        self.control_torque[n] = control_torque #needed for reward calculation

                        torque = control_torque
                        torque_dict.update({name: torque})

                    self.setTorqueControlwithVelocityConstrain(torque_dict)

                    p.applyExternalForce(self.r, index, forceObj=pelvis_push_force, posObj=force_pos, flags=frame_flag)
                    p.stepSimulation()

            self.performFiltering()

        self._observation = self.getExtendedObservation()  # filtered

        self._envStepCounter += 1
        #counter for phase
        self.phase_counter +=1
        self.phase_counter = self.phase_counter%self.period
        reward, reward_term = self._reward()

        done = self._termination()
        # done = False #no termination

        # recording history data for future use
        #update history at the end
        self.prev_action = np.array(self.action) #use previous bounded action

        return np.array(self._observation), reward, done, reward_term

    def _render(self, mode='human', close=False, distance=3, yaw=0, pitch=-30, roll=0, track_pos = None):
        width = 1920
        height = 1080
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        base_orn = p.getEulerFromQuaternion(base_quat)
        if track_pos is None:
            track_pos = base_pos

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=track_pos,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2, #z axis 0,1,2
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width)/height,
            nearVal=0.1,
            farVal=100.0,
        )

        (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:,:,:3]
        return rgb_array

    def _termination(self):
        return self.checkFall()

    def _startLoggingVideo(self):
        self.logId = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                         fileName=self._logFileName + '/video.mp4')

    def _stopLoggingVideo(self):
        p.stopStateLogging(self.logId)

    def _reward(self):
        task_reward, task_reward_term = self.task_reward()
        imitation_reward, imitation_reward_term = self.imitation_reward()

        reward_term = dict()
        reward_term.update(task_reward_term)
        reward_term.update(imitation_reward_term)

        reward_term.update({'task_reward': task_reward})
        reward_term.update({'imitation_reward': imitation_reward})

        reward = self.task_weight*task_reward + self.imitation_weight*imitation_reward
        return reward, reward_term

    def task_reward(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        base_orn = p.getEulerFromQuaternion(base_quat)

        prev_base_orn = p.getEulerFromQuaternion(self.prev_base_quat)

        #Instantaneous final velocity
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.r)
        final_base_pos_vel = np.array(base_pos_vel)
        final_base_orn_vel = np.array(base_orn_vel)

        #Properties used for reward calculation
        #using start base orientation as base coordinate
        base_orn = np.array(prev_base_orn)
        base_quat = np.array(self.prev_base_quat)

        #using final instantaneous velocity during time period
        base_pos_vel = np.array(final_base_pos_vel)
        base_orn_vel = np.array(final_base_orn_vel)

        Rz = rotZ(base_orn[2])
        Rz_i = np.linalg.inv(Rz)
        R = quat_to_rot(base_quat)
        R_i = np.linalg.inv(R)

        base_orn_vel.resize(1, 3)
        base_orn_vel_yaw = np.transpose(Rz_i @ base_orn_vel.transpose())  # base angular velocity in adjusted yaw frame
        base_pos_vel.resize(1, 3)
        base_pos_vel_yaw = np.transpose(Rz_i @ base_pos_vel.transpose())  # base velocity in adjusted yaw frame

        alpha = 1e-2

        base_x_vel_tar = 1.7
        base_y_vel_tar = 0.0
        base_xy_vel_tar = np.array([1.7, 0.0])
        base_z_vel_tar = 0.0

        # No penalization when x_vel_tar<COM_vel_yaw to tolerate higher velocity
        base_x_vel_err = base_x_vel_tar - base_pos_vel_yaw[0][0]
        # base_x_vel_err = base_pos_vel_yaw[0][0]
        base_x_vel_err = np.maximum(base_x_vel_err ,0.0)
        base_xy_vel = np.array([base_pos_vel_yaw[0][0], base_pos_vel_yaw[0][1]])
        base_xy_vel_err = np.linalg.norm(base_xy_vel_tar-base_xy_vel)
        base_y_vel_err = base_y_vel_tar - base_pos_vel_yaw[0][1]
        base_z_vel_err = base_z_vel_tar - base_pos_vel_yaw[0][2]

        base_xy_vel_reward = math.exp(math.log(alpha)*(base_xy_vel_err/1.0)**2)
        base_x_vel_reward = base_x_vel_err
        base_y_vel_reward = math.exp(math.log(alpha)*(base_y_vel_err/1.0)**2)
        base_z_vel_reward = math.exp(math.log(alpha)*(base_z_vel_err/1.0)**2)

        #swing foot clearance
        foot_z = 0.1
        #end effector position
        fl = self._p.getLinkState(self.r, self.jointIdx[self.robotConfig.ground_contact_link[0]], computeLinkVelocity=1)
        fl_vel_xy = np.linalg.norm([fl[1][0], fl[1][1]])
        fl_pos_z = fl[0][2]
        self.fl_z = fl_pos_z
        # fl_err = (fl_pos_z - foot_z)/0.1*fl_vel_xy#0.1
        fl_err = np.maximum(foot_z - fl_pos_z, 0.0)/0.1*fl_vel_xy
        # do not penalize when foot is higher than target height
        fl_reward = math.exp(math.log(alpha) * fl_err**2)

        fr = self._p.getLinkState(self.r, self.jointIdx[self.robotConfig.ground_contact_link[1]], computeLinkVelocity=1)
        fr_vel_xy = np.linalg.norm([fr[1][0], fr[1][1]])
        fr_pos_z = fr[0][2]
        self.fr_z = fr_pos_z
        # fr_err = (fr_pos_z - foot_z)/0.1*fr_vel_xy
        fr_err = np.maximum(foot_z - fr_pos_z, 0.0)/0.1*fr_vel_xy
        # do not penalize when foot is higher than target height
        fr_reward = math.exp(math.log(alpha) * fr_err**2)

        rl = self._p.getLinkState(self.r, self.jointIdx[self.robotConfig.ground_contact_link[2]], computeLinkVelocity=1)
        rl_vel_xy = np.linalg.norm([rl[1][0], rl[1][1]])
        rl_pos_z = rl[0][2]
        self.rl_z = rl_pos_z
        # rl_err = (rl_pos_z - foot_z)/0.1*rl_vel_xy
        rl_err = np.maximum(foot_z - rl_pos_z, 0.0)/0.1*rl_vel_xy
        # do not penalize when foot is higher than target height
        rl_reward = math.exp(math.log(alpha) * rl_err**2)

        rr = self._p.getLinkState(self.r, self.jointIdx[self.robotConfig.ground_contact_link[3]], computeLinkVelocity=1)
        rr_vel_xy = np.linalg.norm([rr[1][0], rr[1][1]])
        rr_pos_z = rr[0][2]
        self.rr_z = rr_pos_z
        # rr_err = (rr_pos_z - foot_z)/0.1*rr_vel_xy
        rr_err = np.maximum(foot_z - rr_pos_z,0.0)/0.1*rr_vel_xy
        # do not penalize when foot is higher than target height
        rr_reward = math.exp(math.log(alpha) * rr_err**2)

        # average foot placement reward
        fl_base = np.array(fl[0]) - base_pos
        fl_base.resize(1, 3)
        fl_base = np.transpose(Rz_i @ fl_base.transpose())

        fr_base = np.array(fr[0]) - base_pos
        fr_base.resize(1, 3)
        fr_base = np.transpose(Rz_i @ fr_base.transpose())

        rl_base = np.array(rl[0]) - base_pos
        rl_base.resize(1, 3)
        rl_base = np.transpose(Rz_i @ rl_base.transpose())

        rr_base = np.array(rr[0]) - base_pos
        rr_base.resize(1, 3)
        rr_base = np.transpose(Rz_i @ rr_base.transpose())

        base_x_pos_tar = (fl[0][0]+fr[0][0]+rl[0][0]+rr[0][0])/4.0
        base_y_pos_tar = (fl[0][1]+fr[0][1]+rl[0][1]+rr[0][1])/4.0
        base_xy_pos_tar = np.array([base_x_pos_tar, base_y_pos_tar])
        base_z_pos_tar = 0.35

        base_xy_pos = np.array([base_pos[0], base_pos[1]])
        base_xy_pos_err = np.linalg.norm(base_xy_pos-base_xy_pos_tar)
        base_xy_pos_reward = math.exp(math.log(alpha)*(base_xy_pos_err/0.3)**2)
        base_z_pos_err = base_z_pos_tar-base_pos[2]
        base_z_pos_err = np.maximum(base_z_pos_err, 0.0) #Do not penalize when robot is higher than base height
        base_z_pos_reward = math.exp(math.log(alpha)*(base_z_pos_err/0.3)**2)

        # angular yaw velocity error (0 desired)
        yaw_vel_err = 0-base_orn_vel[0][2] #yaw velocity error in world frame
        yaw_vel_reward = math.exp(math.log(alpha)*(yaw_vel_err/0.785)**2)

        #base
        #gravity vector in world frame [0,0,-1]
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        gravity = np.array([0,0,-1])
        gravity_quat = p.getQuaternionFromEuler([0,0,0])
        invBasePos, invBaseQuat = p.invertTransform([0,0,0], base_quat)
        #gravity vector in base frame
        gravityPosInBase, gravityQuatInBase = p.multiplyTransforms(invBasePos, invBaseQuat, gravity, gravity_quat)

        base_gravity_vector = np.array(gravityPosInBase)
        gravity_vector_tar = np.array([0,0,-1])
        base_gravity_vector_err = np.linalg.norm(gravity_vector_tar-base_gravity_vector)

        # gravity vector error
        gravity_vector_tar = np.array([0,0,-1])
        base_gravity_vector_err = np.linalg.norm(gravity_vector_tar-base_gravity_vector)
        self.base_gravity_vector = np.array(base_gravity_vector)
        self.base_gravity_vector_err = np.array(base_gravity_vector_err)

        base_gravity_vector_reward = math.exp(math.log(alpha)*(base_gravity_vector_err/1.4)**2)

        # Joint torque and velocity reward
        joint_vel_err = 0
        for key in self.controlled_joints:
            joint_state = p.getJointState(self.r, self.jointIdx[key])

            joint_vel_err += (joint_state[1] / self.v_max[key]) ** 2
        joint_vel_err = joint_vel_err/len(self.controlled_joints)

        joint_torque_err = 0
        for i in range(self._actionDim):
            key = self.controlled_joints[i]

            joint_torque_err += (self.control_torque[i] / self.u_max[key]) ** 2
        joint_torque_err = joint_torque_err/len(self.controlled_joints)

        joint_vel_reward = math.exp(math.log(alpha)*joint_vel_err)
        joint_torque_reward = math.exp(math.log(alpha)*joint_torque_err)

        reward = (
                      8.*base_x_vel_reward \
                    + 1.*base_y_vel_reward \
                    + 1.*base_z_vel_reward \
                    + 2. * base_xy_pos_reward \
                    + 4.*base_z_pos_reward \
                    + 4.*yaw_vel_reward \
                    + 4.*base_gravity_vector_reward \
                    + 1.*joint_vel_reward \
                    + 1.*joint_torque_reward \
                    + 0.5 * fl_reward \
                    + 0.5 * fr_reward \
                    + 0.5 * rl_reward \
                    + 0.5 * rr_reward \
                     ) \
                * 10. / (8.0+1.0+1.0+2.0+4.0+4.0+4.0+1.0+1.0+0.5+0.5+0.5+0.5)

        foot_contact_term = 0
        fall_term = 0
        self_contact_term = 0
        success_term = 0
        if (self.checkGroundContact()): # any feet has contact
            foot_contact_term += 1.

        if not self.checkFall():
            fall_term += 1

        if not self.checkSelfContact():
            self_contact_term += 1.0

        reward += fall_term

        reward_term = []
        reward_term = dict([
            ("yaw_vel_reward", yaw_vel_reward),
            ("base_gravity_vector_reward", base_gravity_vector_reward),
            ("base_x_vel_reward", base_x_vel_reward),
            ("base_y_vel_reward", base_y_vel_reward),
            ("base_z_vel_reward", base_z_vel_reward),
            ("base_xy_pos_reward", base_xy_pos_reward),
            ("base_z_pos_reward", base_z_pos_reward),
            ("foot_contact_term", foot_contact_term),
            ("fall_term", fall_term),
            ("self_contact_term", self_contact_term),
            ("joint_vel_reward", joint_vel_reward),
            ("joint_torque_reward", joint_torque_reward),
            ("fl_reward", fl_reward),
            ("fr_reward", fr_reward),
            ("rl_reward", rl_reward),
            ("rr_reward", rr_reward),
        ])

        return reward, reward_term

    def imitation_reward(self):
        joint_pos_reward = 0
        joint_vel_reward = 0
        total_weight = 0
        alpha = 1e-2

        contact_term = 0

        fl_ground_contact = len(self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[0]], -1)) > 0
        fr_ground_contact = len(self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[1]], -1)) > 0
        rl_ground_contact = len(self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[2]], -1)) > 0
        rr_ground_contact = len(self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[3]], -1)) > 0

        gap = 0.05
        gait_phase = float(self.phase_counter)/float(self.period)

        if gait_phase >= 0 and gait_phase < 0.1:
            if (fl_ground_contact == False and fr_ground_contact == False and rl_ground_contact == True and rr_ground_contact == False):
                contact_term = 2
            else:
                contact_term = 0
        elif gait_phase >= 0.1 and gait_phase < 0.5:
            if (fl_ground_contact == False and fr_ground_contact == False and rl_ground_contact == False and rr_ground_contact == True):
                contact_term = 2
            else:
                contact_term = 0
        elif gait_phase >= 0.5 and gait_phase < 0.6:
            if (fl_ground_contact == True and fr_ground_contact == False and rl_ground_contact == False and rr_ground_contact == False):
                contact_term = 2
            else:
                contact_term = 0
        elif gait_phase >= 0.6 and gait_phase <= 1.0:
            if (fl_ground_contact == False and fr_ground_contact == True and rl_ground_contact == False and rr_ground_contact == False):
                contact_term = 2
            else:
                contact_term = 0
        else:
            contact_term = 0

        reward = contact_term*5

        reward_term = []
        reward_term = dict([
            ("imitation_contact_term", contact_term),
        ])
        return reward, reward_term

    def checkFall(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)

        fall = False

        link = []
        for key,value in self.jointIdx.items():
            link.append(key)

        for name in self.robotConfig.ground_contact_link:
            link.remove(name)

        for key in link:
            if (len(p.getContactPoints(self.r, self.plane, self.jointIdx[key], -1)) > 0):
                # print(key)
                fall = True
                break
        #base contact
        if (len(p.getContactPoints(self.r, self.plane, -1, -1)) > 0):
            # print(key)
            fall = True

        #orientation error
        if self.base_gravity_vector_err>=1.414:
            fall=True

        return fall

    def checkSuccess(self):

        return True

    def checkSelfContact(self):
        link = []
        for key,value in self.jointIdx.items():
            link.append(key)

        collision_list = list()
        collision_count = 0
        temp_link = list(link)
        for linkA in link:
            #check base contact
            contact = len(p.getContactPoints(self.r, self.r, -1, self.jointIdx[linkA])) > 0
            if contact:
                collision_count += 1
                collision_list.append(linkA + ';' + 'base')

            temp_link.remove(linkA)

            for linkB in temp_link:
                contact = len(p.getContactPoints(self.r, self.r, self.jointIdx[linkA], self.jointIdx[linkB])) > 0
                if contact:
                    collision_count += 1
                    collision_list.append(linkA + ';' + linkB)
        # print(collision_list)
        # print(collision_count)
        return collision_count, collision_list

    def resetJointStates(self, base_pos_nom=None, base_orn_nom=None, q_nom=None):
        self.q_nom = self.q_nom_default
        if base_pos_nom is None:
            base_pos_nom = self.base_pos_nom
        if base_orn_nom is None:
            base_orn_nom = self.base_orn_nom
        if q_nom is None:
            self.q_nom = self.q_nom_default#self.q_nom
        else:
        # replace nominal joint angle with target joint angle
            temp=dict(self.q_nom)
            for key, value in q_nom.items():
                temp[key] = value
            q_nom = dict(temp)
            self.q_nom = dict(q_nom)

        for jointName in self.q_nom:
            p.resetJointState(self.r,
                                self.jointIdx[jointName],
                                targetValue=self.q_nom[jointName],
                                targetVelocity=0)

        p.resetBasePositionAndOrientation(self.r, base_pos_nom, base_orn_nom)
        p.resetBaseVelocity(self.r, [0, 0, 0], [0, 0, 0])

    def startRendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    def stopRendering(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def _setupFilter(self):
        #filter using array
        self.state_filter_method = FilterClass(self.stateNumber)
        self.state_filter_method.butterworth(self._dt_filter, 10, 1)  # sample period, cutoff frequency, order

        filter_order = 1
        self.action_filter_method = FilterClass(self._actionDim)
        self.action_filter_method.butterworth(1./self.motor_command_freq, 4, filter_order)  # sample period, cutoff frequency, order

    def _setupCamera(self, cameraDistance=1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=np.array([0, 0, 0.7])):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER,0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=cameraDistance, cameraYaw=cameraYaw, cameraPitch=cameraPitch, cameraTargetPosition=cameraTargetPosition)

    def _setupSimulation(self, base_pos_nom=None, base_orn_nom=None, fixed_base=False, q_nom=None, obstacle=None):
        if base_pos_nom is None:
            base_pos_nom = self.base_pos_nom
        if base_orn_nom is None:
            base_orn_nom = self.base_orn_nom

        self._setupFilter()

        p.resetSimulation()

        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -self.g)
        p.setTimeStep(self._dt)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        currentdir= os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        plane_urdf = self.dir_path + "/plane/plane.urdf"

        self.plane = p.loadURDF(plane_urdf, basePosition=[0, 0, 0], baseOrientation=[0,0,0,1], useFixedBase=True)
        if obstacle is not None:
            self.addTerrain(obstacle_type=obstacle)
        else:
            self.obstacleID = list()

        if self.isEnableSelfCollision == True:
            robot_urdf = self.dir_path + self.robotConfig.fileName
            flags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_INERTIA_FROM_FILE
        else:
            robot_urdf = self.dir_path + self.robotConfig.fileName
            flags = p.URDF_USE_INERTIA_FROM_FILE

        self.r = p.loadURDF(fileName=robot_urdf,
                            basePosition=base_pos_nom,
                            baseOrientation=base_orn_nom,
                            flags=flags,
                            useFixedBase=fixed_base,
                            )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Set up joint mapping
        # set up joint mapping for all existing joints, regardless of revolute, prismatic or fixed
        for jointNo in range(p.getNumJoints(self.r)):
            info = p.getJointInfo(self.r, jointNo)
            joint_name = info[1].decode("utf-8")
            self.jointIdx.update({joint_name: info[0]})
            # print(joint_name, info[0])
            self.jointNameIdx.update({info[0]: joint_name})
        self.nq = len(self.controllable_joints)

        self.setCollisionFilter()

        self.joint_ref_interpolation = []#clear
        for joint in self.controllable_joints:
            info = p.getJointInfo(self.r, self.jointIdx[joint])
            self.jointLowerLimit.append(info[8])
            self.jointUpperLimit.append(info[9])

            name = joint

        #setup joint trajectory interpolation array operation
        self.joint_ref_interpolation = JointTrajectoryInterpolate(self.actionNumber)


        self.resetJointStates(base_pos_nom, base_orn_nom, q_nom)
        self.setZeroOrderHoldNominalPose()

        p.stepSimulation()
        self.initializeFiltering()

        for _ in range(int(self.initial_gap_steps)):  #control loop time steps
            for _ in range(int(self._motor_command_loop_skip)):
                p.stepSimulation()

            self.performFiltering()

    def getObservation(self):
        x_observation = np.zeros((self.stateNumber,)) #create a new numpy array instance

        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        self.base_pos = np.array(base_pos)
        base_orn = p.getEulerFromQuaternion(base_quat)
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.r) # base velocity in cartesian world coordinates

        # overwrite base_pos with the calculated base COM position
        # base_pos is tuple, tuple is immutable
        # change tuple to array
        base_pos = np.array(base_pos)

        Rz = rotZ(base_orn[2])
        Rz_i = np.linalg.inv(Rz)
        self.Rz_i = Rz_i
        R = quat_to_rot(base_quat)
        R_i = np.linalg.inv(R)

        #phase
        phase = self.phase_counter/self.period
        #decouple phase into circular sin, cos for continous transition
        phase_sin = np.sin(2*np.pi*phase)
        phase_cos = np.cos(2*np.pi*phase)
        x_observation[9] = phase_sin
        x_observation[10] = phase_cos

        #gravity vector in world frame [0,0,-1]
        gravity = np.array([0,0,-1])
        gravity_quat = p.getQuaternionFromEuler([0,0,0])
        invBasePos, invBaseQuat = p.invertTransform([0,0,0], base_quat)
        #gravity vector in base frame
        gravityPosInBase, gravityQuatInBase = p.multiplyTransforms(invBasePos, invBaseQuat, gravity, gravity_quat)

        x_observation[3] = gravityPosInBase[0]
        x_observation[4] = gravityPosInBase[1]
        x_observation[5] = gravityPosInBase[2]

        # base linear velocity
        base_pos_vel = np.array(base_pos_vel)
        base_pos_vel.resize(1, 3)
        base_pos_vel_yaw = np.transpose(Rz_i @ base_pos_vel.transpose())  # base velocity in adjusted yaw frame
        x_observation[0] = base_pos_vel_yaw[0][0]
        x_observation[1] = base_pos_vel_yaw[0][1]
        x_observation[2] = base_pos_vel_yaw[0][2]

        self.prev_base_pos_vel_yaw = base_pos_vel_yaw

        #base angular velocity
        base_orn_vel = np.array(base_orn_vel)
        base_orn_vel.resize(1,3)
        base_orn_vel_base = np.transpose(R_i @ base_orn_vel.transpose())

        x_observation[6] = base_orn_vel_base[0][0]
        x_observation[7] = base_orn_vel_base[0][1]
        x_observation[8] = base_orn_vel_base[0][2]

        index = 11

        for i in range(self._actionDim):
            name = self.controlled_joints[i]
            joint_state = p.getJointState(self.r, self.jointIdx[name])
            x_observation[index] = joint_state[0]
            index+=1 #counter

        return np.array(x_observation)

    def getEEF(self):
        #end effector position
        self.fl_lower = self._p.getLinkState(self.r, self.jointIdx[self.robotConfig.ground_contact_link[0]], computeLinkVelocity=0)
        self.fl_lower = np.array(self.fl_lower[0]) - self.base_pos
        self.fl_lower.resize(1, 3)
        self.fl_lower = np.transpose(self.Rz_i @ self.fl_lower.transpose())

        self.fr_lower = self._p.getLinkState(self.r, self.jointIdx[self.robotConfig.ground_contact_link[1]], computeLinkVelocity=0)
        self.fr_lower = np.array(self.fr_lower[0]) - self.base_pos
        self.fr_lower.resize(1, 3)
        self.fr_lower = np.transpose(self.Rz_i @ self.fr_lower.transpose())

        self.rl_lower = self._p.getLinkState(self.r, self.jointIdx[self.robotConfig.ground_contact_link[2]], computeLinkVelocity=0)
        self.rl_lower = np.array(self.rl_lower[0]) - self.base_pos
        self.rl_lower.resize(1, 3)
        self.rl_lower = np.transpose(self.Rz_i @ self.rl_lower.transpose())

        self.rr_lower = self._p.getLinkState(self.r, self.jointIdx[self.robotConfig.ground_contact_link[3]], computeLinkVelocity=0)
        self.rr_lower = np.array(self.rr_lower[0]) - self.base_pos
        self.rr_lower.resize(1, 3)
        self.rr_lower = np.transpose(self.Rz_i @ self.rr_lower.transpose())

        self.fl_force, self.fr_force, self.rl_force, self.rr_force = self.calGroundContactForce()

        self.fl_force_smooth = self.smoothContactForce(self.fl_force)
        self.fr_force_smooth = self.smoothContactForce(self.fr_force)
        self.rl_force_smooth = self.smoothContactForce(self.rl_force)
        self.rr_force_smooth = self.smoothContactForce(self.rr_force)
        return self.fl_lower, self.fr_lower, self.rl_lower, self.rr_lower, \
            self.fl_force, self.fr_force, self.rl_force, self.rr_force,\
            self.fl_force_smooth, self.fr_force_smooth, self.rl_force_smooth, self.rr_force_smooth

    def smoothContactForce(self, force, slope=2.0, threshold=2.0):
        force_norm = np.linalg.norm(force)
        return 1./(1.+math.exp(-slope*(force_norm-threshold)))

    def getJointAngleDict(self):
        joint_angle = dict()
        for key in self.controlled_joints:
            index = self.jointIdx[key]
            joint_state = p.getJointState(self.r, index)
            angle = joint_state[0]
            joint_angle.update({key:angle})

        return joint_angle

    def getJointAngle(self):
        joint_angle = []
        for key in self.controlled_joints:
            index = self.jointIdx[key]
            joint_state = p.getJointState(self.r, index)
            angle = joint_state[0]
            joint_angle.append(angle)

        return np.array(joint_angle)

    def getJointVelDict(self):
        joint_vel = dict()
        for key in self.controlled_joints:
            index = self.jointIdx[key]
            joint_state = p.getJointState(self.r, index)
            vel = joint_state[1]
            joint_vel.update({key:vel})

        return joint_vel

    def getJointVel(self):
        joint_vel = []
        for key in self.controlled_joints:
            index = self.jointIdx[key]
            joint_state = p.getJointState(self.r, index)
            vel = joint_state[1]
            joint_vel.append(vel)

        return np.array(joint_vel)

    def getJointTorqueDict(self):
        joint_torque = dict()
        for key in self.controlled_joints:
            index = self.jointIdx[key]
            joint_state = p.getJointState(self.r, index)
            torque = joint_state[3]
            joint_torque.update({key:torque})

        return joint_torque

    def getJointTorque(self):
        joint_torque = []
        for key in self.controlled_joints:
            index = self.jointIdx[key]
            joint_state = p.getJointState(self.r, index)
            torque = joint_state[3]
            joint_torque.append(torque)

        return np.array(joint_torque)

    def getBaseInfo(self):
        pos, quat = p.getBasePositionAndOrientation(self.r)
        euler = p.getEulerFromQuaternion(quat)
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.r)
        return pos, quat, euler, base_pos_vel, base_orn_vel

    def getFilteredObservation(self):
        #filter using array
        observation_filtered = self.state_filter_method.y[0]
        #replace filtered state with unfiltered state for user provided data
        observation_filtered[9:11] = self.unfiltered_phase
        return observation_filtered

    def initializeFiltering(self):
        observation = self.getObservation()

        #filter using array
        self.state_filter_method.initializeFilter(observation)

    def performFiltering(self):
        observation = self.getObservation()

        #filter using array
        self.state_filter_method.applyFilter(observation)
        self.unfiltered_phase = np.array([observation[9], observation[10]])

    def checkGroundContact(self):
        # At least 2 feet are in contact with the ground
        fl_ground_contact = len(self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[0]], -1)) > 0
        fr_ground_contact = len(self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[1]], -1)) > 0
        rl_ground_contact = len(self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[2]], -1)) > 0
        rr_ground_contact = len(self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[3]], -1)) > 0

        return 1.0 if (fl_ground_contact + fr_ground_contact + rl_ground_contact + rr_ground_contact) >= 2.0 else 0.0

    def checkTerrainCollision(self):
        list = []
        for key in self.jointIdx:
            points_ground = p.getContactPoints(self.r, self.plane, self.jointIdx[key], -1)
            list.extend(points_ground)
            for i in range(len(self.obstacleID)):
                points_terrain = p.getContactPoints(self.r, self.obstacleID[i], self.jointIdx[key], -1)
                list.extend(points_terrain)
        return list

    def setZeroOrderHoldNominalPose(self):
        for jointName in self.controllable_joints:
            p.setJointMotorControl2(self.r,
                                    self.jointIdx[jointName],
                                    p.POSITION_CONTROL,
                                    targetPosition=self.q_nom[jointName],
                                    force=self.u_max[jointName],
                                    )

    def setTorqueControlwithVelocityConstrain(self, torque_dict): #set control
        for jointName, torque in torque_dict.items():
            p.setJointMotorControl2(
                self.r, self.jointIdx[jointName], targetVelocity=np.sign(torque) * self.v_max[jointName],
                force=np.abs(torque),
                controlMode=p.VELOCITY_CONTROL,
            )

    def setTorqueControl(self, torque_dict):
        for jointName, torque in torque_dict.items():
            torque = np.clip(torque, -self.u_max[jointName], self.u_max[jointName])
            #initializing motor for torque control
            #Each motor has a default velocity control mode to keep the velocity zero. You can switch this off
            p.setJointMotorControl2(
                self.r, self.jointIdx[jointName], targetVelocity=np.sign(torque) * self.v_max[jointName],
                force=0,
                controlMode=p.VELOCITY_CONTROL,
            )
            p.setJointMotorControl2(
                self.r, self.jointIdx[jointName],
                force=torque,
                controlMode=p.TORQUE_CONTROL,
            )

    # Initialze state
    def referenceStateInitialization(self):
        idx = self.episode_count % len(self.robotConfig.key_pose)
        pose = self.robotConfig.key_pose[idx]

        # # random initialization
        # clip_bounds = np.array([
        #     [-0.5236, -1.0471, 0.9162, -0.5236, -1.0471, 0.9162, -0.5236, -1.0471, 0.9162, -0.5236, -1.0471, 0.9162],
        #     [0.5236, 1.0471, 2.09, 0.5236, 1.0471, 2.09, 0.5236, 1.0471, 2.09, 0.5236, 1.0471, 2.09]
        # ])
        # base_pos_nom = [0, 0, 0.35]
        # base_orn_nom = euler_to_quat(np.random.uniform(-3.14, 3.14), np.random.uniform(-3.14, 3.14), 0)
        # q_nom = dict([
        #     ("FR_hip_motor_2_chassis_joint", np.random.uniform(clip_bounds[0][0], clip_bounds[1][0])),
        #     ("FR_upper_leg_2_hip_motor_joint", np.random.uniform(clip_bounds[0][1], clip_bounds[1][1])),
        #     ("FR_lower_leg_2_upper_leg_joint", np.random.uniform(clip_bounds[0][2], clip_bounds[1][2])),
        #     ("FL_hip_motor_2_chassis_joint", np.random.uniform(clip_bounds[0][0], clip_bounds[1][0])),
        #     ("FL_upper_leg_2_hip_motor_joint", np.random.uniform(clip_bounds[0][1], clip_bounds[1][1])),
        #     ("FL_lower_leg_2_upper_leg_joint", np.random.uniform(clip_bounds[0][2], clip_bounds[1][2])),
        #     ("RR_hip_motor_2_chassis_joint", np.random.uniform(clip_bounds[0][0], clip_bounds[1][0])),
        #     ("RR_upper_leg_2_hip_motor_joint", np.random.uniform(clip_bounds[0][1], clip_bounds[1][1])),
        #     ("RR_lower_leg_2_upper_leg_joint", np.random.uniform(clip_bounds[0][2], clip_bounds[1][2])),
        #     ("RL_hip_motor_2_chassis_joint", np.random.uniform(clip_bounds[0][0], clip_bounds[1][0])),
        #     ("RL_upper_leg_2_hip_motor_joint", np.random.uniform(clip_bounds[0][1], clip_bounds[1][1])),
        #     ("RL_lower_leg_2_upper_leg_joint", np.random.uniform(clip_bounds[0][2], clip_bounds[1][2])),
        # ])
        # pose=[base_pos_nom, base_orn_nom, q_nom]

        return pose

    def calGroundContactForce(self):
        footGroundContact = []
        ankleRollContact = self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[0]], -1)
        footGroundContact.extend(ankleRollContact)
        fl_contact_info = footGroundContact

        footGroundContact = []
        ankleRollContact = self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[1]], -1)
        footGroundContact.extend(ankleRollContact)
        fr_contact_info = footGroundContact

        footGroundContact = []
        ankleRollContact = self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[2]], -1)
        footGroundContact.extend(ankleRollContact)
        rl_contact_info = footGroundContact

        footGroundContact = []
        ankleRollContact = self._p.getContactPoints(self.r, self.plane, self.jointIdx[self.robotConfig.ground_contact_link[3]], -1)
        footGroundContact.extend(ankleRollContact)
        rr_contact_info = footGroundContact

        fl_contact_force = self.calContactForce(fl_contact_info)
        fr_contact_force = self.calContactForce(fr_contact_info)
        rl_contact_force = self.calContactForce(rl_contact_info)
        rr_contact_force = self.calContactForce(rr_contact_info)

        return fl_contact_force, fr_contact_force, rl_contact_force, rr_contact_force

    def calContactForce(self, contact_info):

        if len(contact_info)<1: # no contact
            F = np.array([0,0,0])
            return F

        F = np.array([0, 0, 0])  # force among the x,y,z axis of world frame
        for i in range(len(contact_info)):
            contactNormal = np.array(contact_info[i][7])  # contact normal of foot pointing towards plane
            contactNormalForce = np.array(contact_info[i][9])
            # print(contactNormalForce)
            F_contact = np.array(contactNormal)*contactNormalForce
            # print(contactNormal)
            F = F + F_contact


        return F

    def setCollisionFilter(self):
        enableCollision = 0
        # non collision pairs with base
        nonCollisionPairs1 = ['FR_upper_leg_2_hip_motor_joint', 'FL_upper_leg_2_hip_motor_joint',
                              'RR_upper_leg_2_hip_motor_joint', 'RL_upper_leg_2_hip_motor_joint']
        for joint0 in nonCollisionPairs1:
            idx0 = self.jointIdx[joint0]
            p.setCollisionFilterPair(self.r, self.r, idx0, -1, enableCollision)

        base_idx = self.jointIdx['floating_base']
        for joint0 in nonCollisionPairs1:
            idx0 = self.jointIdx[joint0]
            p.setCollisionFilterPair(self.r, self.r, idx0, base_idx, enableCollision)

        nonCollisionHip = ['FR_hip_fixed', 'FL_hip_fixed', 'RR_hip_fixed', 'RL_hip_fixed']
        for joint0, joint1 in zip(nonCollisionHip, nonCollisionPairs1):
            idx0 = self.jointIdx[joint0]
            idx1 = self.jointIdx[joint1]
            p.setCollisionFilterPair(self.r, self.r, idx0, idx1, enableCollision)

    def setFootCollisionFilter(self):
        enableCollision = 0
        # non collision pairs with base
        nonCollisionFoot = ["FL_lower_leg_2_foot_joint", "FR_lower_leg_2_foot_joint",
                            "RL_lower_leg_2_foot_joint", "RR_lower_leg_2_foot_joint", ]
        for foot in nonCollisionFoot:
            idx = self.jointIdx[foot]
            p.setCollisionFilterPair(self.r, self.plane, idx, -1, enableCollision)

    def getFrame(self):
        frame = dict()
        link_list = [
            "trunk",
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

        link_new_list = [
            "base_link",
            "FL_HipX_joint",
            "FL_HipY_joint",
            "FL_Knee_joint",
            "FR_HipX_joint",
            "FR_HipY_joint",
            "FR_Knee_joint",
            "HL_HipX_joint",
            "HL_HipY_joint",
            "HL_Knee_joint",
            "HR_HipX_joint",
            "HR_HipY_joint",
            "HR_Knee_joint",
        ]

        for i in range(len(link_list)):
            linkName = link_list[i]
            if linkName != 'trunk':
                state = self._p.getLinkState(self.r, self.jointIdx[linkName])
                pos_orn = state[4:6]
                frame.update({link_new_list[i]: pos_orn})
            else:
                positionA, orientationA = self._p.getBasePositionAndOrientation(self.r)
                # _, _, _, positionB, orientationB, _, _, _, _, _ = self._p.getDynamicsInfo(self.r, -1)
                state = self._p.getDynamicsInfo(self.r, -1)
                positionB, orientationB = state[3], state[4]
                # Need to flip position and orientation to account for the inertia (where COM is) frame to root link frame.
                orientationB = np.array(orientationB)
                orientationB[0] *= -1
                orientationB[1] *= -1
                orientationB[2] *= -1
                pos_orn = self._p.multiplyTransforms(positionA, orientationA, -np.array(positionB), orientationB)
                #dict with new key name
                frame.update({link_new_list[i]: pos_orn})
        return frame


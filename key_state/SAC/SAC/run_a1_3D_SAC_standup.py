#add search directory
from collections import deque, defaultdict
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
os.sys.path.insert(0, parentdir)
print(parentdir)

import gc
import pickle
import moviepy as mpy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from SAC.SAC.configuration_trot import *
from SAC.agent.agent_SAC import *
from SAC.common.logger import logger
from a1_standup_env_key_state import A1
from a1_description.a1_standup_configuration import *
from SAC.common.util import *
import argparse
gc.enable()

class Run():
    def __init__(self, config, dir_path, args):
        self.dir_path = dir_path
        self.config = config
        self.config.load_configuration(dir_path)
        self.config.print_configuration()

        self.motor_control_freq = self.config.conf['LLC-frequency']
        self.physics_freq = self.config.conf['Physics-frequency']
        self.network_freq = self.config.conf['HLC-frequency']

        self.reward_scale = config.conf['reward-scale']

        self.max_time_per_train_episode = 5#self.config.conf['max-train-time']
        self.max_step_per_train_episode = int(self.max_time_per_train_episode*self.network_freq)
        self.max_time_per_test_episode = 5#self.config.conf['max-test-time']
        self.max_step_per_test_episode = int(self.max_time_per_test_episode*self.network_freq)
        self.train_external_force_disturbance = False

        self.env = A1(renders=False,
                           max_time=self.max_time_per_train_episode,
                            initial_gap_time=0.05,
                           isEnableSelfCollision=True,
                           motor_command_freq=self.motor_control_freq,
                           control_freq=self.network_freq,
                           physics_freq=self.physics_freq,
                           control_type='custom_PD',
                            filter_torque=self.config.conf['filter-torque'],
                            interpolation=self.config.conf['interpolation'],
                            filter_action=self.config.conf['filter-action'],
                           )

        self.clip_bounds = np.array([
            [-0.8028, -2.7923, 0.9162, -0.8028, -2.7923, 0.9162, -0.8028, -2.7923, 0.9162, -0.8028, -2.7923, 0.9162],
            [0.8028, 1.0471, 2.6963, 0.8028, 1.0471, 2.6963, 0.8028, 1.0471, 2.6963, 0.8028, 1.0471, 2.6963]
        ])
        config.conf['state-dim'] = self.env.stateNumber
        self.agent = Agent(self.env, self.config)

        actor = self.agent.agent.load_actor_variable(dir_path+'/best_network')
        self.agent.agent.transfer_actor_variable(actor)

        self.logging = logger(dir_path)

        self.episode_count = 0
        self.step_count = 0
        self.train_iter_count = 0

        self.best_reward = 0
        self.best_episode = 0
        self.best_train_iter = 0

        self.force = [0,0,0]
        self.force_chest = [0, 0, 0]
        self.force_pelvis = [0, 0, 0]

        self.image_list = []

        self.recorded_trajectory = defaultdict(list)

        self.args = args
        self.saliency = args.saliency

    def test(self):

        total_reward = 0
        for i in range(1):#self.config.conf['test-num']):
            q_arr = []
            base_pos_arr = []
            base_orn_arr = []

            # 1:crouch, 2:back, 3:right, 4:left
            key_pose_index = self.args.pose
            base_pos_nom = self.env.robotConfig.key_pose[key_pose_index][0]
            base_orn_nom = self.env.robotConfig.key_pose[key_pose_index][1]
            q_nom = self.env.robotConfig.key_pose[key_pose_index][2]
            if key_pose_index == 1:
                self.path_prefix = "/plane/crouch"
            if key_pose_index == 2:
                self.path_prefix = "/plane/back"
            if key_pose_index == 3:
                self.path_prefix = "/plane/right"
            if key_pose_index == 4:
                self.path_prefix = "/plane/left"
            if not os.path.exists(self.dir_path + self.path_prefix):
                os.makedirs(self.dir_path + self.path_prefix)

            state = self.env._reset(fixed_base=False, base_pos_nom=base_pos_nom, base_orn_nom=base_orn_nom, q_nom=q_nom)
            # state = self.env._reset(base_pos_nom=[0,0,0.9], base_orn_nom=[0,0,0,1], fixed_base=True)
            self.env.startRendering()
            self.env._startLoggingVideo()

            next_state = state

            for step in range(self.max_step_per_test_episode):
                state = np.array(next_state)
                state = np.squeeze(state)

                if step <1*self.network_freq:
                    action = []
                    for key in self.env.controlled_joints:
                        action.append(q_nom[key])
                    action = np.array(action)
                    control_action = action
                    if self.saliency==True:
                        saliency_integrated = self.agent.agent.integrated_gradient_saliency(state)
                else:
                    action = self.agent.action(state)

                    action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)

                    control_action = np.array(action)
                    control_action = rescale(control_action, self.config.conf['actor-output-bounds'], self.config.conf['action-bounds'])

                    control_action = np.clip(control_action, self.clip_bounds[0], self.clip_bounds[1])
                    if self.saliency==True:
                        saliency_integrated = self.agent.agent.integrated_gradient_saliency(state)

                v = self.agent.agent.V(state)
                q = self.agent.agent.Q(state, action)

                self.logging.filename2 = (self.dir_path + self.path_prefix + '/run_log' + '.mat')
                self.logging.add_run('value', v)
                self.logging.add_run('Q', q)

                next_state, reward, done, reward_term = self.env._step(control_action)
                reward = self.reward_scale * reward

                rgb=self.env._render(distance=1.5, pitch=0, yaw=52)
                self.image_list.append(rgb)

                total_reward += reward

                base_pos, base_quat, base_euler, base_vel, _ = self.env.getBaseInfo()
                self.logging.add_run('base_vel', base_vel)
                q = self.env.getJointAngle()
                q_dot = self.env.getJointVel()
                q_t = self.env.getJointTorque()
                q_arr.append(q)
                base_pos_arr.append(base_pos)
                base_orn_arr.append(base_quat)

                obs = self.env.getObservation()
                gravity_vec = obs[0:3]
                self.logging.add_run('gravity_vec', gravity_vec)

                self.logging.add_run('action', action)
                self.logging.add_run('target_joint_angle', control_action)
                self.logging.add_run('joint_angle', q)
                self.logging.add_run('joint_vel', q_dot)
                self.logging.add_run('joint_torque', q_t)
                self.logging.add_run('base_pos', base_pos)

                self.logging.add_run('base_quat', base_quat)
                self.logging.add_run('base_euler', base_euler)
                self.logging.add_run('reward', reward)
                self.logging.add_run('state', np.squeeze(state))
                if self.saliency == True:
                    self.logging.add_run('saliency_integrated', np.squeeze(saliency_integrated))

                for key, value in reward_term.items():
                    self.logging.add_run(key, value)

                frame = self.env.getFrame()
                for key, value in frame.items():
                    self.recorded_trajectory[key].append(value)

            self.env._stopLoggingVideo()
            self.env.stopRendering()
            self.logging.save_run()

        ave_reward = total_reward/self.config.conf['test-num']

        clip = ImageSequenceClip(self.image_list, fps=self.network_freq)
        clip.write_videofile(self.dir_path+self.path_prefix+'/test.mp4', fps=self.network_freq, audio=False)


        trajectories_dic = dict()
        for key, value in self.recorded_trajectory.items():
            # print(key, value)
            pos = []
            orn = []
            for i in range(len(value)):
                pos.append(value[i][0])
                orn.append(value[i][1])

            trajectory = [pos, orn]
            trajectories_dic.update({key: trajectory})

        # file = open(self.dir_path + self.path_prefix + '/robot_dict.txt', "wb")
        # pickle.dump(trajectories_dic, file)
        # file.close()

def main():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    os.sys.path.insert(0, parentdir)
    config = Configuration()
    print(parentdir)

    parser = argparse.ArgumentParser()
    folder = "2021_04_13_02.15.08"
    parser.add_argument("-f", "--folder", type=str, default=folder)
    parser.add_argument("-p", "--pose", type=int, default=2)
    # if -s is specified, assign the value True to args.saliency, otherwise False
    parser.add_argument("-s", "--saliency", action="store_true")
    args = parser.parse_args()
    dir_path = parentdir + '/SAC/SAC/SAC/record/a1/standup/' + args.folder
    test = Run(config, dir_path, args)
    test.test()

if __name__ == '__main__':
    main()

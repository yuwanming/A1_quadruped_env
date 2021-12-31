import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
os.sys.path.insert(0, parentdir)
print(parentdir)

import copy
import gc
import time
from datetime import datetime

from SAC.SAC.configuration_trot import *
from SAC.agent.agent_SAC import *
from SAC.common.logger import logger

from a1_trot_phase_env_key_state import A1
from SAC.common.util import *
from a1_description.a1_configuration import *


import pybullet_envs
import gym
gc.enable()

class Train():
    def __init__(self, config):
        self.config = config

        self.motor_control_freq = self.config.conf['LLC-frequency']
        self.physics_freq = self.config.conf['Physics-frequency']
        self.network_freq = self.config.conf['HLC-frequency']

        self.reward_scale = config.conf['reward-scale']

        self.max_time_per_train_episode = self.config.conf['max-train-time']
        self.max_step_per_train_episode = int(self.max_time_per_train_episode*self.network_freq)
        self.max_time_per_test_episode = self.config.conf['max-test-time']
        self.max_step_per_test_episode = int(self.max_time_per_test_episode*self.network_freq)

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
            [-0.087, -2.7923, 0.9162, -0.087, -2.7923, 0.9162, -0.087, -2.7923, 0.9162, -0.087, -2.7923, 0.9162],
            [0.087, 1.0471, 2.6963, 0.087, 1.0471, 2.6963, 0.087, 1.0471, 2.6963, 0.087, 1.0471, 2.6963]
        ])
        self.config.conf['state-dim'] = self.env.stateNumber
        self.config.conf['action-bounds'] = self.env.action_bound
        self.agent = Agent(self.env, self.config)

        self.episode_count = 0
        self.step_count = 0
        self.train_iter_count = 0

        self.best_reward = 0
        self.best_episode = 0
        self.best_train_iter = 0


        # create new network
        dir_path = 'SAC/record/a1/trot/' + datetime.now().strftime('%Y_%m_%d_%H.%M.%S')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.logging = logger(dir_path)
        self.config.save_configuration(dir_path)
        self.config.record_configuration(dir_path)
        self.config.print_configuration()
        self.agent.agent.load_network(dir_path) #initialize saver
        self.dir_path = dir_path

        self.n_step_gamma = np.ones(self.config.conf['n-step'])
        for i in range(self.config.conf['n-step']):
            self.n_step_gamma[i] = self.config.conf['gamma']**i
        self.paths = []
        self.buffer = ReplayBuffer(self.config.conf['replay-buffer-size'])
        self.robotConfig = A1Config()


    def get_single_path(self):
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        task_rewards = []
        imitation_rewards = []

        self.episode_count+=1

        # state = self.env._reset(fixed_base=False)
        state = self.env._reset(fixed_base=False, base_vel=[0,0,0]) # train with no initialization velocity
        for step in range(self.max_step_per_train_episode):
            self.step_count+=1

            state = np.squeeze(state)

            action = self.agent.action_noise(state)

            action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)
            action = np.clip(action, self.config.conf['actor-output-bounds'][0], self.config.conf['actor-output-bounds'][1])

            control_action = np.array(action)
            control_action = rescale(control_action, self.config.conf['actor-output-bounds'], self.config.conf['action-bounds'])
            control_action = np.clip(control_action, self.clip_bounds[0], self.clip_bounds[1])

            next_state, reward, terminal, _ = self.env._step(control_action)
            reward = self.reward_scale*reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminal)
            next_states.append(next_state)

            if step <= self.config.conf['n-step']:
                terminal = False

            if terminal: #only terminate
                break

            state = np.array(next_state)

        path = dict(states=np.array(states), actions=np.array(actions), rewards=np.array(rewards), dones=np.array(dones),
                    next_states=np.array(next_states), task_rewards=task_rewards, imitation_rewards=imitation_rewards)

        self.agent.noise.reset()

        return path

    def get_paths(self, num_of_paths=None, prefix='', verbose=True):
        if num_of_paths is None:
            num_of_paths = self.config.conf['max-path-num']
        paths = []
        t = time.time()
        if verbose:
            print(prefix + 'Gathering Samples')
        step_count = 0

        path_count = 0
        while(1):

            #explored path
            path = self.get_single_path()

            states = path['states']
            actions = path['actions']
            next_states = path['next_states']
            rewards = path['rewards']
            dones = path['dones']

            length = len(dones)
            #calculate cumulative n step rewards
            for i in range(length-(self.config.conf['n-step']-1)):
                s = states[i][:]
                a = actions[i][:]
                r = rewards[i:i+self.config.conf['n-step']] #n step rewards

                cum_r = np.sum(r*self.n_step_gamma) #cumulative sum over n steps
                s_next = next_states[i+self.config.conf['n-step']-1][:]
                d = dones[i+self.config.conf['n-step']-1]

                s = np.squeeze(s)
                a = np.squeeze(a)
                cum_r = np.squeeze(cum_r)
                s_next = np.squeeze(s_next)
                d = np.squeeze(d)

                self.agent.replay_buffer.add(s, a, cum_r, s_next, d)

            paths.append(path)
            step_count += len(dones)
            path_count +=1
            num_of_paths = path_count
            if step_count>=self.config.conf['max-path-step']:
                break

        if verbose:
            print('%i paths sampled. %i steps sampled. %i total paths sampled. Total time used: %f.' % (num_of_paths, step_count, self.episode_count, time.time() - t))
        return paths

    def train_paths(self):

        self.paths = self.get_paths()
        return

    def train(self):
        self.paths = self.get_paths()
        print(len(self.agent.replay_buffer))
        if len(self.agent.replay_buffer) > self.config.conf['replay-start-size']:
            self.train_iter_count+=1
            self.train_SAC()

            self.test()

        return

    def train_SAC(self, prefix = '', verbose = True):
        t = time.time()
        Q1_loss, Q2_loss, policy_loss, temp_loss, expected_entropy, entropy, std, temp = self.agent.train_actor_SAC()

        self.logging.add_train('Q1_loss', Q1_loss)
        self.logging.add_train('Q2_loss', Q2_loss)
        self.logging.add_train('policy_loss', policy_loss)
        self.logging.add_train('temp_loss', temp_loss)
        self.logging.add_train('expected_entropy', expected_entropy)
        self.logging.add_train('entropy', entropy)
        self.logging.add_train('std', std)
        self.logging.add_run('temp', temp)

        self.logging.save_train()

        if verbose:
            print(prefix + 'Training network. Total time used: %f.' % (time.time() - t))


        return

    def test(self):
        total_reward = 0
        total_task_reward = 0
        total_imitation_reward = 0
        for i in range(self.config.conf['test-num']):#
            if i == 0:
                pose = self.robotConfig.key_pose[4]
                base_pos_nom = pose[0]
                base_orn_nom = pose[1]
                q_nom = pose[2]
                base_vel = [0,0,0]
            elif i == 1:
                pose = self.robotConfig.key_pose[5]
                base_pos_nom = pose[0]
                base_orn_nom = pose[1]
                q_nom = pose[2]
                base_vel = [0,0,0]
            elif i == 2:
                pose = self.robotConfig.key_pose[6]
                base_pos_nom = pose[0]
                base_orn_nom = pose[1]
                q_nom = pose[2]
                base_vel = [0,0,0]
            elif i == 3:
                pose = self.robotConfig.key_pose[7]
                base_pos_nom = pose[0]
                base_orn_nom = pose[1]
                q_nom = pose[2]
                base_vel = [0,0,0]
            else:
                pose = self.robotConfig.key_pose[0]
                base_pos_nom = pose[0]
                base_orn_nom = pose[1]
                q_nom = pose[2]
                base_vel = [0,0,0]

            state = self.env._reset(fixed_base=False, base_vel=base_vel, q_nom=q_nom, base_pos_nom=base_pos_nom, base_orn_nom=base_orn_nom)
            # state = self.env._reset(fixed_base=False, base_vel=base_vel)

            for step in range(self.max_step_per_test_episode):
                state = np.squeeze(state)

                action = self.agent.action(state)

                action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)
                action = np.clip(action, self.config.conf['actor-output-bounds'][0], self.config.conf['actor-output-bounds'][1])

                control_action = np.array(action)
                control_action = rescale(control_action, self.config.conf['actor-output-bounds'], self.config.conf['action-bounds'])
                control_action = np.clip(control_action, self.clip_bounds[0], self.clip_bounds[1])

                next_state, reward, terminal, _ = self.env._step(control_action)
                reward = self.reward_scale * reward
                state = np.array(next_state)
                total_reward += reward
                #

                if terminal:
                    break
        ave_reward = total_reward/self.config.conf['test-num']

        self.agent.save_weight(self.step_count, self.dir_path+'/latest_network')
        self.agent.agent.save_actor_variable(self.step_count, self.dir_path + '/latest_network')
        self.agent.agent.save_critic_variable(self.step_count, self.dir_path + '/latest_network')
        if self.best_reward<ave_reward:
            self.best_episode = self.episode_count
            self.best_train_iter = self.train_iter_count
            self.best_reward=ave_reward
            self.agent.save_weight(self.step_count, self.dir_path+'/best_network')
            self.agent.agent.save_actor_variable(self.step_count, self.dir_path+'/best_network')
            self.agent.agent.save_critic_variable(self.step_count, self.dir_path+'/best_network')

        episode_rewards = np.array([np.sum(path['rewards']) for path in self.paths])

        print('iter:' + str(self.train_iter_count) + ' episode:' + str(self.episode_count) + ' step:' + str(self.step_count)
              + ' Deterministic policy return:' + str(ave_reward) + ' Average return:' + str(np.mean(episode_rewards)))
        print('best train_iter', self.best_train_iter, 'best reward', self.best_reward)
        self.logging.add_train('episode', self.episode_count)
        self.logging.add_train('step', self.step_count)
        self.logging.add_train('ave_reward', ave_reward)

        self.logging.add_train('average_return', np.mean(episode_rewards))

        self.logging.save_train()


def main():
    config = Configuration()
    train = Train(config)

    while 1:
        train.train()
        if train.episode_count>config.conf['max-episode-num']:
            break
        if train.step_count>config.conf['max-step-num']:
            break
    return

if __name__ == '__main__':
    main()

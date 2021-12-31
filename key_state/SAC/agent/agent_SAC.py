from SAC.agent.SAC import SAC
from SAC.common.util import *
from SAC.common.prioritized_replay_memory import *
from SAC.common.random_process import *

class Agent:
    """docstring for D4PG"""

    def __init__(self, env, config):

        self.config = config
        self.state_dim = config.conf['state-dim']
        self.action_dim = config.conf['action-dim']

        self.agent = SAC(config, self.state_dim, self.action_dim)

        self.replay_buffer = ReplayBuffer(self.config.conf['replay-buffer-size'])

        self.noise = OrnsteinUhlenbeckProcess(dimension=self.action_dim, num_steps=1, theta=0.15, mu=0.0, sigma=0.2, dt=1.0)

        # Randomly initialize actor network and critic network
        # with both their target networks

        self.batch_size = config.conf['batch-size']
        self.gamma = config.conf['gamma']

        self.action_bounds = config.conf['action-bounds']
        self.actor_output_bounds = config.conf['actor-output-bounds']

        self.minibatch_states=[]
        self.minibatch_actions=[]
        self.minibatch_next_states=[]
        self.minibatch_rewards=[]
        self.minibatch_dones=[]

    def train_actor_SAC(self):
        iter = self.config.conf['max-path-step']*self.config.conf['replay-ratio']
        Q1_loss = 0
        Q2_loss = 0
        policy_loss = 0
        temp_loss = 0

        logstd = 0
        entropy = 0
        logpi = 0
        ave_reward = 0
        print(self.agent.check_network_update())
        for i in range(int(iter)):
            self.minibatch_states, self.minibatch_actions, self.minibatch_rewards, self.minibatch_next_states, self.minibatch_dones =\
                self.replay_buffer.sample(self.config.conf['batch-size'])

            target_V = self.agent.target_V(self.minibatch_next_states)
            # print(target_V)
            discount_return = np.vstack(self.minibatch_rewards) + self.gamma*np.vstack(1.0-self.minibatch_dones)*np.vstack(target_V)
            # print(discount_return)
            Q1_loss += self.agent.train_Q(self.minibatch_states, self.minibatch_actions, discount_return, 'Q1')
            Q2_loss += self.agent.train_Q(self.minibatch_states, self.minibatch_actions, discount_return, 'Q2')

            policy_loss += self.agent.train_actor(self.minibatch_states)
            temp_loss += self.agent.train_temperature(self.minibatch_states)

            entropy += self.agent.entropy(self.minibatch_states)
            logstd += self.agent.logstd(self.minibatch_states)
            logpi += self.agent.logpi(self.minibatch_states)

            ave_reward+=self.minibatch_rewards
            self.agent.soft_update_critic()

            # print(self.agent.check_network_update())
            # self.agent.soft_update_actor()
            # print(self.agent.check_network_update())
            # self.agent.soft_update_critic()
        # print(self.agent.check_network_update())
        Q1_loss = Q1_loss/iter
        Q2_loss = Q2_loss/iter
        policy_loss = policy_loss/iter
        temp_loss = temp_loss/iter
        ave_reward = np.sum(self.minibatch_rewards)/iter
        expected_entropy = -np.mean(logpi, axis=0)/iter
        entropy = np.mean(entropy, axis=0)/iter
        std = np.exp(np.mean(logstd, axis=0)/iter)
        temp = self.agent.temperature()

        print('Q1_loss:', Q1_loss)
        print('Q2_loss:', Q2_loss)
        print('policy_loss:', policy_loss)
        print('temp_loss:', temp_loss)
        print('ave_reward', ave_reward)
        print('- logpi, expected entropy', expected_entropy)
        print('entropy:', entropy)
        print('std:', std)
        print('temperature', temp)

        # self.agent.hard_copy_critic()
        print(self.agent.check_network_update())
        return Q1_loss, Q2_loss, policy_loss, temp_loss, -np.mean(logpi, axis=0)/iter, np.mean(entropy, axis=0)/iter, np.exp(np.mean(logstd, axis=0)/iter), temp

    def train_actor(self):

        return

    def train_critic(self, states, actions, next_states, rewards, dones):

        return #Final Value Network Loss

    def action_noise(self, state):
        action = self.agent.action(state)
        action = np.clip(action, self.actor_output_bounds[0], self.actor_output_bounds[1])

        return action

    def action(self, state):
        action = self.agent.action_mu(state)
        action = np.clip(action, self.actor_output_bounds[0], self.actor_output_bounds[1])

        return action

    def actions_noise(self, state):
        action = self.agent.action(state)
        action = np.clip(action, self.actor_output_bounds[0], self.actor_output_bounds[1])

        return action

    def actions(self, state):
        action = self.agent.action_mu(state)
        action = np.clip(action, self.actor_output_bounds[0], self.actor_output_bounds[1])

        return action

    def reset(self):
        #clear the buffer
        self.buffer = []
        self.noise.reset()

    def save_weight(self, time_step, dir_path):
        print("Now we save model")
        self.agent.save_network(time_step, dir_path)
        self.agent.save_actor_variable(time_step, dir_path)
        self.agent.save_critic_variable(time_step, dir_path)


    def load_weight(self, dir_path):
        print("Now we load the weight")
        actor = self.agent.load_actor_variable(dir_path)
        critic = self.agent.load_critic_variable(dir_path)
        self.agent.transfer_actor_variable(actor)
        self.agent.transfer_critic_variable(critic)

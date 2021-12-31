import tensorflow.contrib as tc
import tensorflow as tf

from SAC.common.grad_inverter import *
from SAC.common.util import *
from SAC.network.actor import *
from SAC.network.critic import *
from SAC.network.net import *
import os

class SAC:
    def __init__(self, config, state_dim, action_dim):

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        self.name = 'SAC'
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = self.config.conf['tau']
        self.critic_lr = self.config.conf['critic-lr']
        self.critic_weight_decay = self.config.conf['critic-weight-decay']
        self.actor_lr = self.config.conf['actor-lr']
        self.actor_weight_decay = self.config.conf['actor-weight-decay']

        self.action_bounds = config.conf['action-bounds']
        self.actor_output_bounds = config.conf['actor-output-bounds']
        self.logstd_bounds = config.conf['actor-output-bounds']
        self.grad_invert = grad_inverter(self.actor_output_bounds, self.sess)
        self.translate_action = translate_action(self.action_bounds, self.actor_output_bounds, self.sess)

        self.batch_size = self.config.conf['batch-size']

        self.squash =self.config.conf['squash-action']

        #temperature
        TEMP_MIN = 0.001
        TEMP_MAX = 5.0
        self.log_alpha = tf.get_variable(initializer=0.0, dtype=tf.float32, trainable=True, name='log_temp') #scalar
        self.alpha = tf.exp(self.log_alpha)
        self.alpha = tf.clip_by_value(self.alpha, TEMP_MIN, TEMP_MAX)

        self.state_input = {}
        self.action_input = {}
        with tf.name_scope('inputs'):
            # define placeholder for inputs to network
            self.state_input['actor'] = tf.placeholder("float", [None, self.state_dim])
            self.action_input['actor'] = tf.placeholder("float", [None, self.action_dim])
            self.state_input['critic'] = tf.placeholder("float", [None, self.state_dim])
            self.action_input['critic'] = tf.placeholder("float", [None, self.action_dim])

        #actor param
        self.actor_input = self.state_input['actor']
        self.actor_input_dim=self.state_dim
        self.actor_output_dim=self.action_dim
        self.actor_layer_dim=self.config.conf['actor-layer-size']
        self.actor_bias=[True]
        self.actor_init_weight=[0.1,0.1,0.01,0.01]
        self.actor_init_bias=[None,None,None,None]
        self.actor_activation=self.config.conf['actor-activation-fn']

        #Q_param
        self.critic_input = tf.concat([self.state_input['critic'], self.action_input['critic']], axis=-1)
        self.critic_input_dim=self.state_dim+self.action_dim
        self.critic_output_dim=1
        self.critic_layer_dim=self.config.conf['critic-layer-size']
        self.critic_bias=[True]
        self.critic_init_weight=np.sqrt(2.0/np.concatenate([[self.critic_input_dim], self.critic_layer_dim],axis=0))
        self.critic_init_bias=[None,None,None,None]
        self.critic_activation=self.config.conf['critic-activation-fn']
        with tf.variable_scope("agent"):

            with tf.variable_scope('actor_network'):
                _, self.joint = tf.split(self.actor_input, [self.state_dim-self.action_dim, self.action_dim], -1)
                self.joint_raw = self.joint
                self.joint = self.translate_action.translate_op(self.joint)
                self.joint = tf.clip_by_value(self.joint, -0.999, 0.999)
                self.joint_tanh = self.joint

                self.actor_network = Fcnn(sess=self.sess, input_dim=self.state_dim, output_dim=self.action_dim*2,
                                            layer_dim=self.actor_layer_dim, config=self.config, if_bias=self.actor_bias,
                                            activation=self.actor_activation, init_weight=self.actor_init_weight,
                                            init_bias=self.actor_init_bias, input_tf=self.actor_input,
                                            name='actor_network', trainable=True, reusable=False)
                self.actor = DeterministicActor(net=self.actor_network, sess=self.sess, config=self.config)
                mu, logstd = tf.split(self.actor.action_mean, [self.action_dim, self.action_dim], -1)
                LOG_SIG_CAP_MAX = 2
                LOG_SIG_CAP_MIN = -10
                logstd = tf.clip_by_value(logstd, LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
                self.actor_logstd = logstd
                self.actor_dist = tc.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logstd))
                self.actor_action = self.actor_dist.sample()
                self.actor_entropy = self.actor_dist.entropy()
                print(self.actor_entropy)
                self.actor_entropy = tf.reshape(self.actor_entropy, [-1, 1]) #inferring the length of the first dimension
                print(self.actor_entropy)
                if self.squash: #squashing output with tanh
                    self.actor_raw_action = self.actor_action
                    self.actor_action = tf.nn.tanh(self.actor_raw_action)
                    self.actor_logpi = self.actor_dist.log_prob(self.actor_raw_action) - tf.reduce_sum(tf.log(1 - tf.nn.tanh(self.actor_raw_action) ** 2 + 1e-6), axis=1)
                    self.actor_mu = tf.nn.tanh(mu)
                else:
                    self.actor_logpi = self.actor_dist.log_prob(self.actor_action)
                    self.actor_mu = mu

                self.actor_mu_translate = self.translate_action.reverse_translate_op(self.actor_mu)

                print(self.actor_logpi)

                self.actor_logpi = tf.reshape(self.actor_logpi, [-1, 1]) #inferring the length of the first dimension
                print(self.actor_logpi)
                self.actor_vars = self.actor.var_list
                self.actor_trainable_vars = self.actor.trainable_var_list

        self.critic_actor_input = tf.concat([self.state_input['critic'], self.actor_action], axis=-1)

        with tf.variable_scope("agent"):
            with tf.variable_scope('Q1_network', reuse=False):
                self.Q1_network = Fcnn(sess=self.sess, input_dim=self.critic_input_dim, output_dim=self.critic_output_dim,
                                            layer_dim=self.critic_layer_dim, config=self.config, if_bias=self.critic_bias,
                                            activation=self.critic_activation, init_weight=self.critic_init_weight,
                                            init_bias=self.critic_init_bias, input_tf=self.critic_input,
                                            name='Q1_network', trainable=True, reusable=False)
                self.Q1 = Critic(net=self.Q1_network, sess=self.sess, config=self.config)
                self.Q1_output = self.Q1.output
                self.Q1_vars = self.Q1.var_list
                self.Q1_trainable_vars = self.Q1.trainable_var_list

            with tf.variable_scope('Q1_network', reuse=True):
                self.Q1_actor_network = Fcnn(sess=self.sess, input_dim=self.critic_input_dim, output_dim=self.critic_output_dim,
                                            layer_dim=self.critic_layer_dim, config=self.config, if_bias=self.critic_bias,
                                            activation=self.critic_activation, init_weight=self.critic_init_weight,
                                            init_bias=self.critic_init_bias, input_tf=self.critic_actor_input,
                                            name='Q1_network', trainable=True, reusable=False)
                self.Q1_actor = Critic(net=self.Q1_actor_network, sess=self.sess, config=self.config)
                self.Q1_actor_output = self.Q1_actor.output
                self.Q1_actor_vars = self.Q1_actor.var_list
                self.Q1_actor_trainable_vars = self.Q1_actor.trainable_var_list

            with tf.variable_scope('target_Q1_network',reuse=False):
                self.target_Q1_network = Fcnn(sess=self.sess, input_dim=self.critic_input_dim, output_dim=self.critic_output_dim,
                                            layer_dim=self.critic_layer_dim, config=self.config, if_bias=self.critic_bias,
                                            activation=self.critic_activation, init_weight=self.critic_init_weight,
                                            init_bias=self.critic_init_bias, input_tf=self.critic_input,
                                            name='target_Q1_network', trainable=False, reusable=False)
                self.target_Q1 = Critic(net=self.target_Q1_network, sess=self.sess, config=self.config)
                self.target_Q1_output = self.target_Q1.output
                self.target_Q1_vars = self.target_Q1.var_list
                self.target_Q1_trainable_vars = self.target_Q1.trainable_var_list

            with tf.variable_scope('target_Q1_network',reuse=True):
                self.target_Q1_actor_network = Fcnn(sess=self.sess, input_dim=self.critic_input_dim, output_dim=self.critic_output_dim,
                                            layer_dim=self.critic_layer_dim, config=self.config, if_bias=self.critic_bias,
                                            activation=self.critic_activation, init_weight=self.critic_init_weight,
                                            init_bias=self.critic_init_bias, input_tf=self.critic_actor_input,
                                            name='target_Q1_network', trainable=False, reusable=False)
                self.target_Q1_actor = Critic(net=self.target_Q1_actor_network, sess=self.sess, config=self.config)
                self.target_Q1_actor_output = self.target_Q1_actor.output
                self.target_Q1_actor_vars = self.target_Q1_actor.var_list
                self.target_Q1_actor_trainable_vars = self.target_Q1_actor.trainable_var_list

            with tf.variable_scope('Q2_network',reuse=False):
                self.Q2_network = Fcnn(sess=self.sess, input_dim=self.critic_input_dim, output_dim=self.critic_output_dim,
                                            layer_dim=self.critic_layer_dim, config=self.config, if_bias=self.critic_bias,
                                            activation=self.critic_activation, init_weight=self.critic_init_weight,
                                            init_bias=self.critic_init_bias, input_tf=self.critic_input,
                                            name='Q2_network', trainable=True, reusable=False)
                self.Q2 = Critic(net=self.Q2_network, sess=self.sess, config=self.config)
                self.Q2_output = self.Q2.output
                self.Q2_vars = self.Q2.var_list
                self.Q2_trainable_vars = self.Q2.trainable_var_list

            with tf.variable_scope('Q2_network',reuse=True):
                self.Q2_actor_network = Fcnn(sess=self.sess, input_dim=self.critic_input_dim, output_dim=self.critic_output_dim,
                                            layer_dim=self.critic_layer_dim, config=self.config, if_bias=self.critic_bias,
                                            activation=self.critic_activation, init_weight=self.critic_init_weight,
                                            init_bias=self.critic_init_bias, input_tf=self.critic_actor_input,
                                            name='Q2_network', trainable=True, reusable=False)

                self.Q2_actor = Critic(net=self.Q2_actor_network, sess=self.sess, config=self.config)
                self.Q2_actor_output = self.Q2_actor.output
                self.Q2_actor_vars = self.Q2_actor.var_list
                self.Q2_actor_trainable_vars = self.Q2_actor.trainable_var_list

            with tf.variable_scope('target_Q2_network',reuse=False):
                self.target_Q2_network = Fcnn(sess=self.sess, input_dim=self.critic_input_dim, output_dim=self.critic_output_dim,
                                            layer_dim=self.critic_layer_dim, config=self.config, if_bias=self.critic_bias,
                                            activation=self.critic_activation, init_weight=self.critic_init_weight,
                                            init_bias=self.critic_init_bias, input_tf=self.critic_input,
                                            name='target_Q2_network', trainable=False, reusable=False)
                self.target_Q2 = Critic(net=self.target_Q2_network, sess=self.sess, config=self.config)
                self.target_Q2_output = self.target_Q2.output
                self.target_Q2_vars = self.target_Q2.var_list
                self.target_Q2_trainable_vars = self.target_Q2.trainable_var_list

            with tf.variable_scope('target_Q2_network',reuse=True):
                self.target_Q2_actor_network = Fcnn(sess=self.sess, input_dim=self.critic_input_dim, output_dim=self.critic_output_dim,
                                            layer_dim=self.critic_layer_dim, config=self.config, if_bias=self.critic_bias,
                                            activation=self.critic_activation, init_weight=self.critic_init_weight,
                                            init_bias=self.critic_init_bias, input_tf=self.critic_actor_input,
                                            name='target_Q2_network', trainable=False, reusable=False)
                self.target_Q2_actor = Critic(net=self.target_Q2_actor_network, sess=self.sess, config=self.config)
                self.target_Q2_actor_output = self.target_Q2_actor.output
                self.target_Q2_actor_vars = self.target_Q2_actor.var_list
                self.target_Q2_actor_trainable_vars = self.target_Q2_actor.trainable_var_list

            with tf.variable_scope('V_network', reuse=False):
                print(tf.minimum(self.Q1_actor_output, self.Q2_actor_output))
                self.V_output = tf.minimum(self.Q1_actor_output, self.Q2_actor_output) - self.alpha*self.actor_logpi
                print(self.V_output)
                self.target_V_output = tf.minimum(self.target_Q1_actor_output, self.target_Q2_actor_output) - self.alpha*self.actor_logpi
                print(self.target_V_output)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        for var in self.actor_trainable_vars:
            print(var.name)
        for var in self.actor_vars:
            print(var.name)
        for var in self.Q1_trainable_vars:
            print(var.name)
        for var in self.Q1_vars:
            print(var.name)
        for var in self.target_Q1_trainable_vars:
            print(var.name)
        for var in self.target_Q1_vars:
            print(var.name)
        for var in self.Q1_actor_trainable_vars:
            print(var.name)
        for var in self.Q1_actor_vars:
            print(var.name)
        for var in self.target_Q1_actor_trainable_vars:
            print(var.name)
        for var in self.target_Q1_actor_vars:
            print(var.name)

        with tf.name_scope('actor_train_op'):
            self.setup_actor_training_method()
        with tf.name_scope('critic_train_op'):
            self.setup_Q_training_method()
        with tf.name_scope('network_update_op'):
            self.setup_network_update()
            self.setup_temperature_training_method()

        writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def get_network_update(self, vars, target_vars, tau):  # copy from var into target_var
        soft_update = []
        hard_copy = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            hard_copy.append(tf.assign(target_var, var))
            soft_update.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
        assert len(hard_copy) == len(vars)
        assert len(soft_update) == len(vars)
        return tf.group(*hard_copy), tf.group(*soft_update)


    def setup_network_update(self):
        self.hard_copy = {}
        self.soft_update = {}
        #print(self.actor_network['vars'])
        self.hard_copy['Q1'], self.soft_update['Q1'] = \
            self.get_network_update(self.Q1_vars, self.target_Q1_vars, self.tau)

        self.hard_copy['Q2'], self.soft_update['Q2'] = \
            self.get_network_update(self.Q2_vars, self.target_Q2_vars, self.tau)
        #used to check whether the old network is correctly updated

        self.Q1_var_diff = tf.reduce_sum(flatten_var(self.Q1_vars)-flatten_var(self.target_Q1_vars))
        self.Q2_var_diff = tf.reduce_sum(flatten_var(self.Q2_vars)-flatten_var(self.target_Q2_vars))

        return

    def setup_Q_training_method(self):
        self.target_q_input = tf.placeholder("float", [None, self.critic_output_dim])
        self.Q1_loss = 0.5*tf.reduce_mean(tf.square(self.Q1_output-self.target_q_input))
        self.Q2_loss = 0.5*tf.reduce_mean(tf.square(self.Q2_output-self.target_q_input))
        self.Q1_optimizer = tc.opt.AdamWOptimizer(weight_decay=self.critic_weight_decay, learning_rate=self.critic_lr).minimize(self.Q1_loss)
        self.Q2_optimizer = tc.opt.AdamWOptimizer(weight_decay=self.critic_weight_decay, learning_rate=self.critic_lr).minimize(self.Q2_loss)


    def setup_actor_training_method(self):

        self.saliency=list()#(action_dim,1,?,state_dim)
        self.saliency_abs=list()#(action_dim,1,?,state_dim)
        for i in range(self.action_dim):
            saliency = tf.gradients(ys=self.actor_mu[:, i], xs=self.state_input['actor'])#(?,state_dim)
            self.saliency.append(saliency)
            self.saliency_abs.append(tf.abs(saliency))
        # print(self.saliency[0])#(?,state_dim)

        #difference between actor output and measured joint angle
        self.smooth_loss = tf.reduce_mean(tf.square(self.joint_raw-self.actor_mu_translate))

        """
        Soft Actor-Critic Algorithms and Applications
        https://bair.berkeley.edu/blog/2018/12/14/sac/
        Minimum of the two Q values is used durinf policy update
        """
        self.policy_loss = tf.reduce_mean(self.alpha*self.actor_logpi - tf.minimum(self.Q1_actor_output, self.Q2_actor_output))\
                           + self.config.conf['loss-output-smooth-coeff']*self.smooth_loss

        """
        Addressing Function Approximation Error in Actor-Critic Methods
        Twin delayed DDPG
        One of the Q value is used during policy update
        """

        self.policy_optimizer = \
            tc.opt.AdamWOptimizer(weight_decay=self.actor_weight_decay, learning_rate=self.actor_lr).minimize(loss=self.policy_loss, var_list=self.actor_trainable_vars)
        return

    def setup_temperature_training_method(self):
        ENTROPY_TARGET = -self.action_dim
        self.temperature_loss = - tf.reduce_mean(self.log_alpha * tf.stop_gradient(self.actor_logpi + ENTROPY_TARGET))
        self.temp_optimizer = tf.train.AdamOptimizer(self.actor_lr).minimize(loss = self.temperature_loss, var_list=[self.log_alpha])


    def train_actor(self, state_batch):
        feed_dict = {self.state_input['critic']: state_batch,
                     self.state_input['actor']:state_batch,
                         }
        _, loss = self.sess.run([self.policy_optimizer, self.policy_loss], feed_dict=feed_dict)

        return loss

    def train_Q(self, state_batch, action_batch, y_batch, name='Q1'):
        if name=='Q1':

            loss,  _ = self.sess.run([self.Q1_loss, self.Q1_optimizer], feed_dict={
                self.target_q_input: y_batch,
                self.state_input['critic']: state_batch,
                self.action_input['critic']: action_batch,
            })
        elif name=='Q2':
            loss,  _ = self.sess.run([self.Q2_loss, self.Q2_optimizer], feed_dict={
                self.target_q_input: y_batch,
                self.state_input['critic']: state_batch,
                self.action_input['critic']: action_batch,
            })
        else:
            loss,  _ = self.sess.run([self.Q1_loss, self.Q1_optimizer], feed_dict={
                self.target_q_input: y_batch,
                self.state_input['critic']: state_batch,
                self.action_input['critic']: action_batch,
            })
        return loss

    def train_temperature(self, state_batch):
        loss,  _ = self.sess.run([self.temperature_loss, self.temp_optimizer], feed_dict={
            self.state_input['actor']: state_batch,
        })

        return loss
    def check_network_update(self):
        diff1, diff2 = self.sess.run([self.Q1_var_diff, self.Q2_var_diff])
        print('old and new network parameter diff:', diff1, ', ',diff2)
        return

    def soft_update_critic(self):
        self.sess.run(self.soft_update['Q1'])
        self.sess.run(self.soft_update['Q2'])
        return

    def hard_copy_critic(self):
        self.sess.run(self.hard_copy['Q1'])
        self.sess.run(self.hard_copy['Q2'])
        return

    def gradient_saliency(self,state,action_dim):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]
            return self.sess.run(self.saliency[action_dim], feed_dict={
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.saliency[action_dim], feed_dict={
                self.state_input['actor']: state_batch,
            })

    def integrated_gradient(self, state, action_dim, x_baseline=None, x_steps=25):
        if x_baseline is None:
            x_baseline = np.zeros_like(state)

        assert x_baseline.shape == state.shape

        x_diff = state - x_baseline

        total_gradients = np.zeros_like(state)

        for alpha in np.linspace(0, 1, x_steps):
            x_step = x_baseline + alpha * x_diff

            total_gradients += np.squeeze(self.gradient_saliency(x_step, action_dim))
        return total_gradients * x_diff / x_steps

    def integrated_gradient_saliency(self, state):
        self.integrated=list()
        for i in range(self.action_dim):
            integrated = self.integrated_gradient(state=state, action_dim=i)
            self.integrated.append(np.abs(integrated))
        int_grad = np.sum(np.squeeze(self.integrated), axis=0)
        return int_grad
        # print(int_grad.shape)

    def action(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]
            return self.sess.run(self.actor_action, feed_dict={
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.actor_action, feed_dict={
                self.state_input['actor']: state_batch,
            })

    def action_mu(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]
            return self.sess.run(self.actor_mu, feed_dict={
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.actor_mu, feed_dict={
                self.state_input['actor']: state_batch,
            })

    def temperature(self):
        return self.sess.run(self.alpha)

    def V(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]  # add new axis
            return self.sess.run(self.V_output, feed_dict={
                self.state_input['critic']: state,
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.V_output, feed_dict={
                self.state_input['critic']: state_batch,
                self.state_input['actor']: state_batch,
            })

    def target_V(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]  # add new axis
            return self.sess.run(self.target_V_output, feed_dict={
                self.state_input['critic']: state,
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.target_V_output, feed_dict={
                self.state_input['critic']: state_batch,
                self.state_input['actor']: state_batch,
            })

    def Q(self, state, action, name='Q1'):
        if name=='Q1':
            if state.ndim<2: # no batch
                state = state[np.newaxis,:] # add new axis
                action = action[np.newaxis, :]
                return self.sess.run(self.Q1_output, feed_dict={
                    self.state_input['critic']: state,
                    self.action_input['critic']:action,
                    self.action_input['critic']: action,
                })[0]
            else:
                state_batch = state
                action_batch = action
                return self.sess.run(self.Q1_output, feed_dict={
                    self.state_input['critic']: state_batch,
                    self.action_input['actor']: action_batch,
                    self.action_input['critic']: action_batch,
                })
        elif name=='Q2':
            if state.ndim<2: # no batch
                state = state[np.newaxis,:] # add new axis
                action = action[np.newaxis, :]
                return self.sess.run(self.Q2_output, feed_dict={
                    self.state_input['critic']: state,
                    self.action_input['critic']:action,
                    self.action_input['critic']: action,
                })[0]
            else:
                state_batch = state
                action_batch = action
                return self.sess.run(self.Q2_output, feed_dict={
                    self.state_input['critic']: state_batch,
                    self.action_input['actor']: action_batch,
                    self.action_input['critic']: action_batch,
                })
        else:
            if state.ndim<2: # no batch
                state = state[np.newaxis,:] # add new axis
                action = action[np.newaxis, :]
                return self.sess.run(self.Q1_output, feed_dict={
                    self.state_input['critic']: state,
                    self.action_input['critic']:action,
                    self.action_input['critic']: action,
                })[0]
            else:
                state_batch = state
                action_batch = action
                return self.sess.run(self.Q1_output, feed_dict={
                    self.state_input['critic']: state_batch,
                    self.action_input['actor']: action_batch,
                    self.action_input['critic']: action_batch,
                })

    def target_Q(self, state, action, name='Q1'):
        if name =='Q1':
            if state.ndim<2: # no batch
                state = state[np.newaxis,:] # add new axis
                action = action[np.newaxis, :]
                return self.sess.run(self.target_Q1_output, feed_dict={
                    self.state_input['critic']: state,
                    self.action_input['critic']:action,
                    self.action_input['critic']: action,
                })[0]
            else:
                state_batch = state
                action_batch = action
                return self.sess.run(self.target_Q1_output, feed_dict={
                    self.state_input['critic']: state_batch,
                    self.action_input['actor']: action_batch,
                    self.action_input['critic']: action_batch,
                })
        elif name=='Q2':
            if state.ndim < 2:  # no batch
                state = state[np.newaxis, :]  # add new axis
                action = action[np.newaxis, :]
                return self.sess.run(self.target_Q2_output, feed_dict={
                    self.state_input['critic']: state,
                    self.action_input['critic']: action,
                    self.action_input['critic']: action,
                })[0]
            else:
                state_batch = state
                action_batch = action
                return self.sess.run(self.target_Q2_output, feed_dict={
                    self.state_input['critic']: state_batch,
                    self.action_input['actor']: action_batch,
                    self.action_input['critic']: action_batch,
                })
        else:
            if state.ndim < 2:  # no batch
                state = state[np.newaxis, :]  # add new axis
                action = action[np.newaxis, :]
                return self.sess.run(self.target_Q1_output, feed_dict={
                    self.state_input['critic']: state,
                    self.action_input['critic']: action,
                    self.action_input['critic']: action,
                })[0]
            else:
                state_batch = state
                action_batch = action
                return self.sess.run(self.target_Q1_output, feed_dict={
                    self.state_input['critic']: state_batch,
                    self.action_input['actor']: action_batch,
                    self.action_input['critic']: action_batch,
                })

    def entropy(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]  # add new axis
            return self.sess.run(self.actor_entropy, feed_dict={
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.actor_entropy, feed_dict={
                self.state_input['actor']: state_batch,
            })

    def logstd(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]  # add new axis
            return self.sess.run(self.actor_logstd, feed_dict={
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.actor_logstd, feed_dict={
                self.state_input['actor']: state_batch,
            })

    def logpi(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]  # add new axis
            return self.sess.run(self.actor_logpi, feed_dict={
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.actor_logpi, feed_dict={
                self.state_input['actor']: state_batch,
            })

    def load_network(self, dir_path):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(dir_path + '/saved_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:" + checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step, dir_path):
        print('save network...' + str(time_step))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.saver.save(self.sess, dir_path + '/saved_networks/' + 'network')  # , global_step = time_step)


    def save_actor_variable(self, time_step, dir_path):
        var_dict = {}
        for var in self.actor_trainable_vars:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            #print(var.name + '  shape: ')
            #print(np.shape(var_array))
        output = open(dir_path + '/actor_variable.obj', 'wb')
        pickle.dump(var_dict, output)
        output.close()

    def load_actor_variable(self, dir_path):
        var = {}
        pkl_file = open(dir_path + '/actor_variable.obj', 'rb')
        var_temp = pickle.load(pkl_file)
        var.update(var_temp)
        pkl_file.close()
        return var

    def transfer_actor_variable(self, var_dict):
        for var in self.actor_trainable_vars:
            for key in var_dict:
                if key in var.name: # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')

    def save_critic_variable(self, time_step, dir_path):
        var_dict = {}
        for var in self.Q1_trainable_vars:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            #print(var.name + '  shape: ')
            #print(np.shape(var_array))
        for var in self.Q2_trainable_vars:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            #print(var.name + '  shape: ')
            #print(np.shape(var_array))
        output = open(dir_path + '/critic_variable.obj', 'wb')
        pickle.dump(var_dict, output)
        output.close()

    def load_critic_variable(self, dir_path):
        var = {}
        pkl_file = open(dir_path + '/critic_variable.obj', 'rb')
        var_temp = pickle.load(pkl_file)
        var.update(var_temp)
        pkl_file.close()
        return var

    def transfer_critic_variable(self, var_dict):
        for var in self.Q1_trainable_vars:
            for key in var_dict:
                if key in var.name: # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')

        for var in self.Q2_trainable_vars:
            for key in var_dict:
                if key in var.name:  # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')
        #self.update_critic()

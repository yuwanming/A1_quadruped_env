import numpy as np
import tensorflow as tf

class Actor(object):

    def __init__(self, net, sess, config):


        self.net = net
        self.sess = sess
        self.config = config

        self.input_ph = self.net.input
        self.output_net = self.net.output
        self.var_list = []
        self.trainable_var_list = []

    def get_action(self, inputs):
        raise NotImplementedError


class DeterministicActor(Actor):

    def __init__(self, net, sess, config):
        super(DeterministicActor, self).__init__(net, sess, config)

        self.action_mean = self.output_net

        self.action = self.output_net

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        self.trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def get_action(self, inputs):

        if len(inputs.shape) < 2:
            inputs = inputs[np.newaxis,:]

        feed_dict = {self.input_ph: inputs}

        action = self.sess.run(self.action, feed_dict)

        return action


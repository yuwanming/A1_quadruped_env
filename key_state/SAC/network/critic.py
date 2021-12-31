from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys

class Critic(object):
    def __init__(self, net, sess, config):
        self.sess = sess
        self.config = config
        self.var_list = []

        self.net = net

        self.input = self.net.input
        self.output = self.net.output

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        self.trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

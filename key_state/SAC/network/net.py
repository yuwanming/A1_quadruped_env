from __future__ import print_function
import tensorflow as tf
import numpy as np


class Net(object):
    def __init__(self, sess, input_dim, output_dim, layer_dim, config, name=None, **kwargs):
        self.config = config
        self.sess = sess
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.name = name

        self.all_layer_dim = np.concatenate([[self.input_dim], layer_dim, [self.output_dim]], axis=0).astype(int)
        self.if_bias = kwargs.get('if_bias', ([True] * len(layer_dim)) + [False])
        self.activation_fns = kwargs.get('activation', (['tanh'] * len(layer_dim)) + ['None'])
        self.initialize_weight = kwargs.get('init_weight', None)
        self.initialize_bias = kwargs.get('init_bias', None)
        self.trainable = kwargs.get('trainable', True)
        self.reusable = kwargs.get('reusable', False)
        if len(self.if_bias) == 1:
            self.if_bias *= len(layer_dim) + 1
        if len(self.activation_fns) == 1:
            self.activation_fns *= len(layer_dim) + 1

        act_funcs_dict = {'tanh': tf.nn.tanh, 'relu': tf.nn.relu, 'leaky_relu':tf.nn.leaky_relu, 'elu': tf.nn.elu,'None': lambda x: x}
        self.activation_fns_call = [act_funcs_dict[_] for _ in self.activation_fns]

    def build(self):
        raise NotImplementedError


class Fcnn(Net):
    def __init__(self, sess, input_dim, output_dim, layer_dim, config, name=None, **kwargs):
        super(Fcnn, self).__init__(sess, input_dim, output_dim, layer_dim, config, name, **kwargs)

        self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name='input'))

        assert (self.input.get_shape()[-1].value == input_dim)
        assert (len(self.activation_fns_call) == len(self.layer_dim) + 1)
        assert (len(self.if_bias) == len(self.layer_dim) + 1)

        self.output = self.build(self.input, self.name)

    def build(self, input_tf, name):

        net = input_tf
        weights = []
        if np.any(self.if_bias):
            biases = []

        for i, (dim_1, dim_2) in enumerate(zip(self.all_layer_dim[:-1], self.all_layer_dim[1:])):
            if self.if_bias[i]:
                init_w = self.initialize_weight[i] if self.initialize_weight is not None else .1
                weight = tf.get_variable(initializer=tf.truncated_normal([dim_1, dim_2], stddev=init_w), trainable=self.trainable, dtype=tf.float32, name='theta_%i' % i)
                weights.append(weight)
                # zero initialization for bias
                init_b = self.initialize_bias[i]
                if init_b is None:
                    bias = tf.get_variable(initializer=np.zeros([1, dim_2]).astype(np.float32), trainable=self.trainable, dtype=tf.float32, name='bias_%i' % i)
                    biases.append(bias)
                else:
                    bias = tf.get_variable(initializer=tf.truncated_normal([dim_2], stddev=init_b), trainable=self.trainable, dtype=tf.float32, name='bias_%i' % i)
                    biases.append(bias)
                net = self.activation_fns_call[i](tf.matmul(net, weights[i]) + biases[-1])

            else:
                # print(dim_1,dim_2)
                init_w = self.initialize_weight[i] if self.initialize_weight is not None else .1
                weight = tf.get_variable(initializer=tf.truncated_normal([dim_1, dim_2], stddev=init_w), trainable=self.trainable, dtype=tf.float32, name='theta_%i' % i)
                weights.append(weight)
                net = self.activation_fns_call[i](tf.matmul(net, weights[i]))

        return net
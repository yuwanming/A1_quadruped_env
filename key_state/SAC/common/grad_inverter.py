import tensorflow as tf
import numpy as np

class grad_inverter:
    def __init__(self, action_bounds, sess): #[lower, upper]
        self.sess = sess
        self.action_size = len(action_bounds[0])
        self.action_bounds = action_bounds

        self.action_input = tf.placeholder(tf.float32, [None, self.action_size])
        self.pmax = tf.constant(action_bounds[1], dtype=tf.float32)
        self.pmin = tf.constant(action_bounds[0], dtype=tf.float32)
        self.prange = tf.constant([upper - lower for lower, upper in zip(action_bounds[0], action_bounds[1])], dtype=tf.float32)
        self.pdiff_max = tf.div(-self.action_input + self.pmax, self.prange)
        self.pdiff_min = tf.div(self.action_input - self.pmin, self.prange)
        self.zeros_act_grad_filter = tf.zeros([self.action_size])
        self.act_grad = tf.placeholder(tf.float32, [None, self.action_size])
        self.grad_inverter = tf.where(tf.greater(self.act_grad, self.zeros_act_grad_filter),
                                      tf.multiply(self.act_grad, self.pdiff_max),
                                      tf.multiply(self.act_grad, self.pdiff_min))

    def invert(self, grad, action):
        return self.sess.run(self.grad_inverter, feed_dict={self.action_input: action, self.act_grad: grad})

    def invert_op(self, grad, action):

        pmax = tf.constant(self.action_bounds[1], dtype=tf.float32)
        pmin = tf.constant(self.action_bounds[0], dtype=tf.float32)
        prange = tf.constant([upper - lower for lower, upper in zip(self.action_bounds[0], self.action_bounds[1])], dtype=tf.float32)
        pdiff_max = tf.div(-action + pmax, prange)
        pdiff_min = tf.div(action - pmin, prange)
        zeros_act_grad_filter = tf.zeros([self.action_size])
        grad_inverter_op = tf.where(tf.greater(grad, zeros_act_grad_filter),
                                      tf.multiply(grad, pdiff_max),
                                      tf.multiply(grad, pdiff_min))
        return grad_inverter_op

class clip_action:
    def __init__(self, action_bounds, sess):
        self.sess = sess
        self.action_size = len(action_bounds[0])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_size])
        self.clip_action = tf.clip_by_value(self.action_input, action_bounds[0], action_bounds[1])

    def clip(self, action):
        if action.ndim < 2:  # no batch
            action = action[np.newaxis, :]
            return self.sess.run(self.clip_action, feed_dict={self.action_input: action})[0]
        else:
            action_batch = action
            return self.sess.run(self.clip_action, feed_dict={self.action_input: action_batch})

class translate_action:
    def __init__(self, input_bounds, output_bounds, sess):
        self.sess = sess
        self.action_size = len(input_bounds[0])
        self.input_bounds = input_bounds
        self.output_bounds = output_bounds
        self.input_range = (self.input_bounds[1][:] - self.input_bounds[0][:])#upper bound - lower bound
        self.output_range = (self.output_bounds[1][:] - self.output_bounds[0][:])

    def translate(self, action):
        if action.ndim < 2:  # no batch
            action = action[np.newaxis, :]
            return self.sess.run(self.rerange_action, feed_dict={self.action_input: action})[0]
        else:
            action_batch = action

            return self.sess.run(self.rerange_action, feed_dict={self.action_input: action_batch})

    def translate_op(self, input):
        return tf.add(tf.subtract(input,self.input_bounds[0])/self.input_range*self.output_range, self.output_bounds[0])

    def reverse_translate_op(self, input):
        return tf.add(tf.subtract(input,self.output_bounds[0])/self.output_range*self.input_range, self.input_bounds[0])
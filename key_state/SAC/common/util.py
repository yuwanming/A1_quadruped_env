import numpy as np
import math
import tensorflow as tf
import pickle

def translate(value, old_range, new_range):
    value = np.array(value)
    old_range = np.array(old_range)
    new_range = np.array(new_range)

    OldRange = float(old_range[1][:] - old_range[0][:])
    NewRange = float(new_range[1][:] - new_range[0][:])
    NewValue = (value - old_range[0][:]) * NewRange / float(OldRange) + new_range[0][:]
    return NewValue

def flatten_var(var_list):
    return tf.concat([tf.reshape(var, [tf.size(var)]) for var in var_list], axis = 0)

def rotX(theta):
    R = np.array([ \
        [1.0, 0.0, 0.0], \
        [0.0, math.cos(theta), -math.sin(theta)], \
        [0.0, math.sin(theta), math.cos(theta)]])
    return R


def rotY(theta):
    R = np.array([ \
        [math.cos(theta), 0.0, math.sin(theta)], \
        [0.0, 1.0, 0.0], \
        [-math.sin(theta), 0.0, math.cos(theta)]])
    return R


def rotZ(theta):
    R = np.array([ \
        [math.cos(theta), -math.sin(theta), 0.0], \
        [math.sin(theta), math.cos(theta), 0.0], \
        [0.0, 0.0, 1.0]])
    return R

def quat_to_rot(qs):  # transform quaternion into rotation matrix
    qx = qs[0]
    qy = qs[1]
    qz = qs[2]
    qw = qs[3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = np.empty([3, 3])
    m[0, 0] = 1.0 - (yy + zz)
    m[0, 1] = xy - wz
    m[0, 2] = xz + wy
    m[1, 0] = xy + wz
    m[1, 1] = 1.0 - (xx + zz)
    m[1, 2] = yz - wx
    m[2, 0] = xz - wy
    m[2, 1] = yz + wx
    m[2, 2] = 1.0 - (xx + yy)

    return m

def euler_to_quat(roll, pitch, yaw):  # rad
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return [x, y, z, w]

def rescale(value, old_range, new_range):
    value = np.array(value)
    old_range = np.array(old_range)
    new_range = np.array(new_range)

    OldRange = old_range[1][:] - old_range[0][:]
    NewRange = new_range[1][:] - new_range[0][:]
    NewValue = (value - old_range[0][:]) * NewRange / OldRange + new_range[0][:]
    return NewValue

def clip_increment(ref_value, target_value,inc_bound):
    min_bound = ref_value-inc_bound
    max_bound = ref_value+inc_bound
    new_value = np.clip(target_value, min_bound, max_bound)
    return new_value
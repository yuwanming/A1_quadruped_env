from collections import deque, defaultdict
import numpy as np
import random
import scipy.io

class logger(object):
    def __init__(self, dir_path):
        self.filename1 = (dir_path + '/train_log' + '.mat')
        self.filename2 = (dir_path + '/run_log' + '.mat')

        self.log = defaultdict(list)
        self.episode = []
        self.step =[]
        self.reward = []

    def add_train(self, key, value):
        self.log[key].append(value)

    def add_run(self, key, value):
        self.log[key].append(value)

    def save_train(self):
        scipy.io.savemat(self.filename1,
                         mdict=self.log)
    def save_run(self):
        scipy.io.savemat(self.filename2,
                         mdict=self.log)
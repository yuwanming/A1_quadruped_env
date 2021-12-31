import pickle
import numpy as np
import math

class Configuration:
    def __init__(self):
        self.conf={}
        self.conf['env-id'] = 'HumanoidBalanceFilter-v0'
        self.conf['render-eval'] = False

        self.conf['joint-interpolation'] = True

        control_joint_num = 12
        if control_joint_num == 12:
            self.conf['controlled-joints'] = list([
                "FR_hip_motor_2_chassis_joint",
                "FR_upper_leg_2_hip_motor_joint",
                "FR_lower_leg_2_upper_leg_joint",
                "FL_hip_motor_2_chassis_joint",
                "FL_upper_leg_2_hip_motor_joint",
                "FL_lower_leg_2_upper_leg_joint",
                "RR_hip_motor_2_chassis_joint",
                "RR_upper_leg_2_hip_motor_joint",
                "RR_lower_leg_2_upper_leg_joint",
                "RL_hip_motor_2_chassis_joint",
                "RL_upper_leg_2_hip_motor_joint",
                "RL_lower_leg_2_upper_leg_joint",
            ])

        actor_joint_num = 12
        if actor_joint_num == 12:
            self.conf['actor-action-joints'] = list([
                "FR_hip_motor_2_chassis_joint",
                "FR_upper_leg_2_hip_motor_joint",
                "FR_lower_leg_2_upper_leg_joint",
                "FL_hip_motor_2_chassis_joint",
                "FL_upper_leg_2_hip_motor_joint",
                "FL_lower_leg_2_upper_leg_joint",
                "RR_hip_motor_2_chassis_joint",
                "RR_upper_leg_2_hip_motor_joint",
                "RR_lower_leg_2_upper_leg_joint",
                "RL_hip_motor_2_chassis_joint",
                "RL_upper_leg_2_hip_motor_joint",
                "RL_lower_leg_2_upper_leg_joint",
            ])

        self.joint_pos_bounds = dict([
            ("FR_hip_motor_2_chassis_joint", [-0.8028, 0.8028]),  # FR_HipX_joint
            ("FR_upper_leg_2_hip_motor_joint", [-4.1885, 1.0471]),  # FR_HipY_joint
            ("FR_lower_leg_2_upper_leg_joint", [0.9162, 2.6963]),  # FR_Knee_joint
            ("FL_hip_motor_2_chassis_joint", [-0.8028, 0.8028]),  # FL_HipX_joint
            ("FL_upper_leg_2_hip_motor_joint", [-4.1885, 1.0471]),  # FL_HipY_joint
            ("FL_lower_leg_2_upper_leg_joint", [0.9162, 2.6963]),  # FL_Knee_joint
            ("RR_hip_motor_2_chassis_joint", [-0.8028, 0.8028]),  # HR_HipX_joint
            ("RR_upper_leg_2_hip_motor_joint", [-4.1885, 1.0471]),  # HR_HipY_joint
            ("RR_lower_leg_2_upper_leg_joint", [0.9162, 2.6963]),  # HR_Knee_joint
            ("RL_hip_motor_2_chassis_joint", [-0.8028, 0.8028]),  # HL_HipX_joint
            ("RL_upper_leg_2_hip_motor_joint", [-4.1885, 1.0471]),  # HL_HipY_joint
            ("RL_lower_leg_2_upper_leg_joint", [0.9162, 2.6963]),  # HL_Knee_joint
        ])

        self.action_bounds = self.joint_pos_bounds

        self.conf['state-dim'] = 60
        self.conf['action-dim'] = len(self.conf['actor-action-joints'])
        self.conf['action-bounds'] = np.zeros((2,len(self.conf['actor-action-joints'])))
        self.conf['normalized-action-bounds'] = np.zeros((2, len(self.conf['actor-action-joints'])))
        self.conf['actor-logstd-initial'] = np.zeros((1, len(self.conf['actor-action-joints'])))
        self.conf['actor-logstd-bounds'] = np.ones((2,len(self.conf['actor-action-joints'])))
        self.conf['actor-output-bounds'] = np.ones((2,len(self.conf['actor-action-joints'])))
        self.conf['actor-output-bounds'][0][:] = -1 * np.ones((len(self.conf['actor-action-joints']),))
        self.conf['actor-output-bounds'][1][:] = 1* np.ones((len(self.conf['actor-action-joints']),))
        for i in range(len(self.conf['actor-action-joints'])):
            joint_name = self.conf['actor-action-joints'][i]
            self.conf['action-bounds'][0][i] = self.action_bounds[joint_name][0] # lower bound
            self.conf['action-bounds'][1][i] = self.action_bounds[joint_name][1] # upper bound

            std = (self.conf['actor-output-bounds'][1][i]-self.conf['actor-output-bounds'][0][i])
            self.conf['actor-logstd-initial'][0][i] = np.log(std*0.25)
            self.conf['actor-logstd-bounds'][0][i] = np.log(std*0.1)
            self.conf['actor-logstd-bounds'][1][i] = np.log(std*0.25)

        self.conf['Physics-frequency'] =1000
        self.conf['LLC-frequency'] =500
        self.conf['HLC-frequency'] = 25
        self.conf['bullet-default-PD'] = False

        self.conf['gating-layer-size'] = [128,128]
        self.conf['gating-activation-fn'] = ['relu','relu','None']
        self.conf['gating-index'] = None
        self.conf['expert-index'] = None
        self.conf['expert-num'] = 4

        self.conf['batch-size'] = 128

        self.conf['filter-torque'] = False
        self.conf['interpolation'] = False
        self.conf['filter-action'] = True

        self.conf['dsr-gait-freq'] = 1.667
        self.conf['dsr-gait-period'] = 0.6

        gait_period = self.conf['dsr-gait-period']
        t = gait_period * self.conf['HLC-frequency']
        gamma = np.round(0.5**(1/t), 3)#round to 3 decimals
        self.conf['gamma'] = gamma
        self.conf['tau'] = 0.001
        self.conf['n-step'] = 1
        self.conf['categorical-distribution'] = dict( v_min = -10, v_max=250, n_atoms = 51)

        self.conf['critic-layer-size'] = [256, 256]
        self.conf['critic-activation-fn'] = ['relu', 'relu', 'None']

        self.conf['critic-lr'] = 3e-4
        self.conf['critic-weight-decay'] =1e-6
        self.conf['critic-l2-reg'] = 3e-4

        self.conf['actor-layer-size'] = [256, 256]
        self.conf['actor-activation-fn'] = ['relu', 'relu', 'None']

        self.conf['actor-lr'] = 3e-4
        self.conf['actor-weight-decay'] = 1e-6
        self.conf['actor-l2-reg'] = 3e-4

        self.conf['squash-action'] = True

        self.conf['max-path-num'] = 20
        self.conf['max-path-step'] = 5000
        self.conf['replay-ratio'] = 1

        self.conf['loss-entropy-coeff'] = 0.0
        self.conf['loss-output-bound-coeff'] = 0.0
        self.conf['loss-output-smooth-coeff'] = 0.5
        self.conf['loss-output-diff-coeff'] = 0

        self.conf['prioritized-exp-replay'] = True
        self.conf['replay-buffer-size'] = 1000000
        self.conf['replay-start-size'] = 10000
        self.conf['record-start-size'] = self.conf['replay-start-size']*1.0

        self.conf['reward-scale'] = 0.1
        self.conf['epoch-num'] = 500
        self.conf['epoch-step-num'] = 5000000
        self.conf['total-step-num'] = 2500000000
        self.conf['max-train-time'] = 10 #second
        self.conf['max-test-time'] = 10 #second
        self.conf['test-num'] = 4
        self.conf['rollout-step-num'] = 1
        self.conf['train-step-num'] = 1
        self.conf['max-episode-num'] = 5000000
        self.conf['max-step-num'] = 2500000000

        self.conf['imitation-weight'] = 0.5
        self.conf['task-weight'] = 0.5

    def save_configuration(self,dir):
        # write python dict to a file
        output = open(dir + '/configuration.obj', 'wb')
        pickle.dump(self.conf, output)
        output.close()

    def load_configuration(self,dir):
        # write python dict to a file
        pkl_file = open(dir + '/configuration.obj', 'rb')
        conf_temp = pickle.load(pkl_file)
        self.conf.update(conf_temp)
        pkl_file.close()

    def record_configuration(self,dir):
        # write python dict to a file
        output = open(dir + '/readme.txt', 'w')
        for key in self.conf:
            output.write("{}: {}\n".format(key,self.conf[key]))

    def print_configuration(self):
        for key in self.conf:
            print(key + ': ' + str(self.conf[key]))

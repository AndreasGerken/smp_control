#from puppy_config import PuppyConfig
from robot_configs.puppy_configs.main_config import PuppyConfig

import numpy as np

import matplotlib.pyplot as plt

import argparse

def get_class():
    return PuppyConfigSinSweep

class PuppyConfigSinSweep(PuppyConfig):
    def __init__(self,args):
        PuppyConfig.__init__(self,args)

        self.set_sensors(['acc', 'gyr'])
        self.lag = 1
        self.embedding = 1

        self.learning_enabled = False
        self.use_sensors_for_model = False

    def send_output(self, algorithm_output):
        # do one period

        parser = argparse.ArgumentParser()
        parser.add_argument("-periods", "--periods", type=int, default = 0)
        args, unknown_args = parser.parse_known_args()

        phase = (self.smp_control.cnt_main / (float)(self.smp_control.numtimesteps)) * 2 * np.pi * args.periods

        algorithm_output = np.array([np.sin(phase)] * 4)
        self.smp_control.y[self.smp_control.cnt_main,:] = algorithm_output

        # velocity control
        self.motor_velocity = self.motor_position_commands - algorithm_output
        self.motor_position_commands = algorithm_output
        self.motor_position_estimate = self.motor_position_estimate * 0.3 + self.motor_position_commands * 0.7

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.msg_motors_velocity.data = algorithm_output
        self.smp_control.pub["_puppyMotor"].publish(self.msg_motors)
        self.smp_control.pub["_puppyMotorVelocity"].publish(self.msg_motors_velocity)

        return algorithm_output

    def before_exit(self):
        return
        #plt.plot(self.smp_control.y)
        #plt.show()

if __name__ == "__main__":
    PuppyConfigSinSweep(None)

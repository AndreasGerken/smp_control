#from puppy_config import PuppyConfig
from robot_configs.puppy_config import PuppyConfig

import numpy as np

def get_class():
    return PuppyConfigSinSweep

class PuppyConfigSinSweep(PuppyConfig):
    def __init__(self,args):
        PuppyConfig.__init__(self,args)

        self.set_sensors(['acc', 'gyr', 'orient'])
        self.numsen = 10
        self.lag = 1
        self.embedding = 1

        self.learning_enabled = False
        self.use_sensors_for_model = False

    def send_output(self, algorithm_output):
        algorithm_output = np.array([np.sin(self.smp_control.cnt_main / 100.)] * 4)

        # velocity control
        self.motor_velocity = self.motor_position_commands - algorithm_output
        self.motor_position_commands = algorithm_output
        self.motor_position_estimate = self.motor_position_estimate * 0.3 + self.motor_position_commands * 0.7

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.msg_motors_velocity.data = algorithm_output
        self.smp_control.pub["_homeostasis_motor"].publish(self.msg_motors)
        self.smp_control.pub["_homeostasis_motor_velocity"].publish(self.msg_motors_velocity)

if __name__ == "__main__":
    PuppyConfigSinSweep(None)

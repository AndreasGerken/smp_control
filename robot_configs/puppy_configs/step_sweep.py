#from puppy_config import PuppyConfig
from robot_configs.puppy_configs.main_config import PuppyConfig

import numpy as np

import matplotlib.pyplot as plt


def get_class():
    return PuppyConfigStepSweep


class PuppyConfigStepSweep(PuppyConfig):
    def __init__(self, args):
        PuppyConfig.__init__(self, args)
        self.classname = self.__class__.__name__
        self.set_sensors(['acc', 'gyr'])
        self.numsen = 6
        self.lag = 1
        self.embedding = 1

        self.learning_enabled = False
        self.use_sensors_for_model = False

        self.step_length = 125
        self.reset_length = 125
        self.step_size = 0.333 # 0.333 corresponds to 30 deg

    def send_output(self, algorithm_output):
        cycle_length = self.step_length + self.reset_length
        cycle_max = self.smp_control.numtimesteps / cycle_length
        sweep_angle_total = 2. - self.step_size
        step_per_cycle = sweep_angle_total / cycle_max

        cycle_number = self.smp_control.cnt_main / cycle_length
        position_in_cycle = self.smp_control.cnt_main % cycle_length

        if position_in_cycle > self.reset_length:
            position = step_per_cycle * cycle_number + self.step_size - 1.
        else:
            position = step_per_cycle * cycle_number - 1.

        algorithm_output = np.array(
            [position] * 4)

        self.smp_control.y[self.smp_control.cnt_main, :] = algorithm_output

        # velocity control
        self.motor_velocity = self.motor_position_commands - algorithm_output
        self.motor_position_commands = algorithm_output
        self.motor_position_estimate = self.motor_position_estimate * \
            0.3 + self.motor_position_commands * 0.7

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.msg_motors_velocity.data = algorithm_output
        self.smp_control.pub["_puppyMotor"].publish(self.msg_motors)
        self.smp_control.pub["_puppyMotorVelocity"].publish(
            self.msg_motors_velocity)

    def before_exit(self):
        return
        # plt.plot(self.smp_control.y)
        # plt.show()


if __name__ == "__main__":
    PuppyConfigSinSweep(None)

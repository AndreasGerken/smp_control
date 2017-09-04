#from puppy_config import PuppyConfig
from robot_configs.puppy_configs.main_config import PuppyConfig
import numpy as np

# python smp_control.py robot_configs/puppy_configs/position_hk_hind_front.py -lt 0.01 -n 10000 -eC 0.1 -eA 0.005


def get_class():
    return PuppyConfigPosition


class PuppyConfigPosition(PuppyConfig):
    def __init__(self, args):
        PuppyConfig.__init__(self, args)

        #self.set_sensors(['acc', 'gyr', 'orient', 'motor_pos'])
        #self.set_sensors(['acc', 'gyr', 'orient', 'motor_pos','euler'])
        self.nummot = 2

        self.set_sensors(['gyr', 'acc'])

        self.lag = 9
        self.embedding = 1
        self.learning_enabled = True
        self.use_sensors_for_model = False

    def send_output(self, algorithm_output):

        # repeat the output twice to control front and hind legs together
        self.motor_position_commands = np.repeat(algorithm_output, 2) * 0.7

        # position control
        self.motor_velocity = self.motor_position_estimate - self.motor_position_commands

        self.motor_position_estimate = self.motor_position_estimate * \
            0.3 + self.motor_position_commands * 0.7

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.msg_motors_velocity.data = self.motor_velocity

        self.smp_control.pub["_puppyMotor"].publish(self.msg_motors)
        self.smp_control.pub["_puppyMotorVelocity"].publish(
            self.msg_motors_velocity)


if __name__ == "__main__":
    robot = PuppyConfigPosition(None)

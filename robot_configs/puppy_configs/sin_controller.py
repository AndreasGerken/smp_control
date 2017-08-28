#from puppy_config import PuppyConfig
from robot_configs.puppy_configs.main_config import PuppyConfig
import numpy as np
import argparse

def get_class():
    return PuppyConfigPosition


class PuppyConfigPosition(PuppyConfig):
    def __init__(self, args):
        PuppyConfig.__init__(self, args)

        #self.set_sensors(['acc', 'gyr', 'orient', 'motor_pos'])
        #self.set_sensors(['acc', 'gyr', 'orient', 'motor_pos','euler'])
        #self.set_sensors(['acc', 'gyr', 'orient', 'euler'])
        self.set_sensors(['acc', 'gyr'])

        self.lag = 1
        self.embedding = 1
        self.learning_enabled = False
        self.use_sensors_for_model = False

        # stable
        #self.omega = np.pi * 2.
        #self.theta = np.pi * 1.5

        # unstable
        self.omega = np.pi * 1.3
        self.theta = np.pi * 0.71


        self.Bh = 0.
        self.Bf = 0.

        self.Ah = 1.
        self.Af = 1.

        parser = argparse.ArgumentParser(
            description='Sine controller for puppy')

        parser.add_argument('-omega', '--omega', type=float,
                            help='frequency of the swings in Hz', required=True)
        parser.add_argument('-theta', '--theta', type=float,
                            help='phase offset of the hind legs in fractions of a period [0-1]', required=True)
        args, unknown = parser.parse_known_args()

        self.omega = args.omega * 2. * np.pi
        self.theta = args.theta * 2. * np.pi

        print args.omega

    def send_output(self, algorithm_output):
        t = self.smp_control.cnt_main * self.smp_control.loop_time

        posF = self.Af * np.sin(self.omega * t) + self.Bf
        posH = self.Ah * np.sin(self.omega * t + self.theta) + self.Bh

        # repeat the output twice to control front and hind legs together
        self.motor_position_commands = np.repeat(np.array([posF, posH]), 2)
        print self.motor_position_commands

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

        return self.motor_position_commands


if __name__ == "__main__":
    robot = PuppyConfigPosition(None)

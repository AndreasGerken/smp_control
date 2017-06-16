import numpy as np

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu

def get_class():
    return PuppyConfigFindResonance

# static class methods
def add_specific_args(parser):
    return

class PuppyConfigFindResonance(PuppyConfig):
    def __init__(self,args):
        PuppyConfig.__init__

        self.set_sensors(['orient', 'motor_pos'])

        self.lag = 4
        self.embedding = 1

        self.learning_enabled = True
        self.use_sensors_for_model = True

    def send_output(self, algorithm_output):
        self.motor_velocity = self.motor_position_commands - algorithm_output
        self.motor_position_commands = algorithm_output
        self.motor_position_estimate = self.motor_position_estimate * 0.3 + self.motor_position_commands * 0.7

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.msg_motors_velocity.data = self.motor_velocity
        self.pub["_homeostasis_motor"].publish(self.msg_motors)
        self.pub["_homeostasis_motor_velocity"].publish(self.msg_motors_velocity)

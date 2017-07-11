#from puppy_config import PuppyConfig
from robot_configs.puppy_configs.main_config import PuppyConfig
import numpy as np

def get_class():
    return PuppyConfigVelocity

class PuppyConfigVelocity(PuppyConfig):
    def __init__(self,args):
        PuppyConfig.__init__(self,args)

        #self.set_sensors(['gyr', 'motor_pos'])
        self.set_sensors(['gyr'])
        self.lag = 4
        self.embedding = 1
        self.learning_enabled = True
        self.use_sensors_for_model = False
        self.classname = self.__class__.__name__

    def send_output(self, algorithm_output):
        # velocity control

        self.motor_position_commands = (self.motor_position_commands + algorithm_output * 0.05).clip(-0.1,0.1)
        self.motor_position_estimate = self.motor_position_estimate * 0.3 + self.motor_position_commands * 0.7

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.msg_motors_velocity.data = algorithm_output
        self.smp_control.pub["_homeostasis_motor"].publish(self.msg_motors)
        self.smp_control.pub["_homeostasis_motor_velocity"].publish(self.msg_motors_velocity)

if __name__ == "__main__":
    robot = PuppyConfigVelocity(None)

#from puppy_config import PuppyConfig
from robot_configs.pendulum_configs.main_config import PendulumConfig

def get_class():
    return PendulumTorqueHK

class PendulumTorqueHK(PendulumConfig):
    def __init__(self,args):
        PuppyConfig.__init__(self,args)

        self.set_sensors(['poti'])
        self.lag = 4
        self.embedding = 1
        self.learning_enabled = True
        self.use_sensors_for_model = True

    def send_output(self, algorithm_output):
        # velocity control
        self.motor_position_commands = (self.motor_position_commands + algorithm_output).clip(-1,1)
        self.motor_position_estimate = self.motor_position_estimate * 0.3 + self.motor_position_commands * 0.7

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.msg_motors_velocity.data = algorithm_output
        self.smp_control.pub["_homeostasis_motor"].publish(self.msg_motors)
        self.smp_control.pub["_homeostasis_motor_velocity"].publish(self.msg_motors_velocity)

if __name__ == "__main__":
    robot = PuppyConfigVelocity(None)

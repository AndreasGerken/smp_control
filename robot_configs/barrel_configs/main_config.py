from std_msgs.msg import Float64MultiArray

from robot_configs.general_robot_config import RobotConfig

def get_class():
    return BarrelConfig

class BarrelConfig(RobotConfig):
    def __init__(self, args):
        RobotConfig.__init__(self)

        self.pub_names = {
            "/motors": [Float64MultiArray]
        }
        self.sub_names = {
            "/sensors": [Float64MultiArray, self.cb_sensors],
        }

        self.sensor_dimensions = {'rot':2}

        self.set_sensors(['rot'])

        self.nummot = 2
        self.lag = 1
        self.embedding = 1
        self.output_gain = 1

        self.learning_enabled = True
        self.use_sensors_for_model = False

        # Initialize Ros messages
        self.msg_motors = Float64MultiArray()


    def cb_sensors(self, msg):
        """ROS Sensor callback"""
        self.sensor_vec = msg.data

    def get_input(self):
        return self.sensor_vec

    def send_output(self, algorithm_output):
        self.msg_motors.data = algorithm_output
        self.smp_control.pub["_motors"].publish(self.msg_motors)

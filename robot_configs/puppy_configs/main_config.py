import numpy as np
import tf

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
from robot_configs.general_robot_config import RobotConfig

def get_class():
    return PuppyConfig

class PuppyConfig(RobotConfig):
    def __init__(self, args):
        RobotConfig.__init__(self)

        self.pub_names = {
            "/puppyMotor": [Float32MultiArray],
            "/puppyMotorVelocity": [Float32MultiArray],
        }
        self.sub_names = {
            "/imu/data": [Imu, self.cb_imu],
        }

        self.sensor_dimensions = {
            "acc": 3, "gyr": 3, "orient": 4, "euler": 3,  "motor_pos": 4, "motor_vel": 4}

        #self.use_sensors = ['orient', 'gyr', 'motor_pos']
        self.use_sensors = None
        self.numsen = 0
        self.nummot = 4
        self.lag = 4
        self.embedding = 1
        self.output_gain = 32000.

        self.learning_enabled = True
        self.use_sensors_for_model = True

        #self.control_mode = control_modes[args.control_mode]

        # Initialize Ros messages
        self.msg_motors = Float32MultiArray()
        self.msg_motors_velocity = Float32MultiArray()

        self.motor_velocity = np.zeros((self.nummot))
        self.motor_position_commands = np.zeros_like(self.motor_velocity)
        self.motor_position_estimate = np.zeros_like(self.motor_velocity)
        self.sensor_vec = np.zeros((self.numsen))

    def cb_imu(self, msg):
        """ROS IMU callback"""
        sensor_tupel = ()

        # go through the use_sensors array and attach the values to the sensor
        # tupel. The order of use_sensors is maintained in the sensor_tupel
        for sensor in self.use_sensors:
            if sensor == 'acc':
                imu_vec_acc = (msg.linear_acceleration.x,
                               msg.linear_acceleration.y, msg.linear_acceleration.z)
                sensor_tupel += (imu_vec_acc, )

            elif sensor == 'gyr' :
                imu_vec_gyr = (msg.angular_velocity.x,
                               msg.angular_velocity.y, msg.angular_velocity.z)
                sensor_tupel += (imu_vec_gyr, )

            elif sensor == 'orient' :
                imu_vec_orient = (msg.orientation.x, msg.orientation.y,
                                  msg.orientation.z, msg.orientation.w)
                sensor_tupel += (imu_vec_orient, )

            elif sensor =='euler' :
                quaternion = (
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w)
                euler = tf.transformations.euler_from_quaternion(quaternion)
                sensor_tupel += (euler, )

            elif sensor == 'motor_pos' :
                sensor_tupel += (self.motor_position_estimate,)

            elif sensor == 'motor_vel' :
                sensor_tupel += (self.motor_velocity,)

        # bring them together
        if(len(sensor_tupel) == 0):
            self.sensor_vec = None
        else:
            self.sensor_vec = np.hstack(sensor_tupel)

    def get_input(self):
        return self.sensor_vec

    # def send_output(self, algorithm_output):
    # this function should be implemented by child classes

import numpy as np
import tf

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu

def get_class():
    return PuppyConfig

# static class methods
def add_specific_args(parser):
    return

class PuppyConfig():
    def __init__(self,args):
        self.pub_names = {
            "/homeostasis_motor": [Float32MultiArray],
            "/homeostasis_motor_velocity": [Float32MultiArray],
            }
        self.sub_names = {
            "/imu/data": [Imu, self.cb_imu],
            }

        self.sensor_dimensions = {"acc": 3, "gyr": 3, "orient": 4, "euler":3,  "motor_pos": 4, "motor_vel" : 4}

        #self.use_sensors = ['orient', 'gyr', 'motor_pos']
        self.use_sensors = None
        self.numsen = 0
        self.set_sensors(['orient', 'motor_pos'])
        #self.use_sensors = ['orient', 'motor_pos', 'gyr', 'acc']

        #self.numsen = np.sum([sensor_dimensions[sensor] for sensor in self.use_sensors])
        self.nummot = 4
        self.lag = 4
        self.embedding = 1
        self.output_gain = 32000

        self.learning_enabled = True
        self.use_sensors_for_model = True

        #self.control_mode = control_modes[args.control_mode]

        # Initialize Ros messages
        self.msg_motors     = Float32MultiArray()
        self.msg_motors_velocity     = Float32MultiArray()

        self.motor_velocity = np.zeros((self.nummot))
        self.motor_position_commands = np.zeros_like(self.motor_velocity)
        self.motor_position_estimate = np.zeros_like(self.motor_velocity)
        self.imu_vec = np.zeros((self.numsen))

        # gets set by smp_control
        self.pub = None
        self.smp_control = None

    def set_sensors(self, use_sensors):
        self.use_sensors = use_sensors
        self.numsen = np.sum([self.sensor_dimensions[sensor] for sensor in self.use_sensors])
        self.imu_vec = np.zeros((self.numsen))
        print self.numsen

    def cb_imu(self, msg):
        """ROS IMU callback"""
        sensor_tupel = ()

        if 'acc' in self.use_sensors:
            imu_vec_acc = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
            sensor_tupel += (imu_vec_acc, )

        if 'gyr' in self.use_sensors:
            imu_vec_gyr = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
            sensor_tupel += (imu_vec_gyr, )

        if 'orient' in self.use_sensors:
            imu_vec_orient = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
            sensor_tupel += (imu_vec_orient, )
        if 'euler' in self.use_sensors:
            quaternion = (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            sensor_tupel += (euler, )

        if 'motor_pos' in self.use_sensors:
            sensor_tupel += (self.motor_position_estimate,)

        if 'motor_vel' in self.use_sensors:
            sensor_tupel += (self.motor_velocity,)

        # bring them together
        if(len(sensor_tupel) == 0):
            self.imu_vec = None
        else:
            self.imu_vec = np.hstack(sensor_tupel)

    def get_input(self):
        return self.imu_vec

    def send_output(self, algorithm_output):
        self.motor_velocity = self.motor_position_commands - algorithm_output
        self.motor_position_commands = algorithm_output
        self.motor_position_estimate = self.motor_position_estimate * 0.3 + self.motor_position_commands * 0.7

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.msg_motors_velocity.data = self.motor_velocity
        self.smp_control.pub["_homeostasis_motor"].publish(self.msg_motors)
        self.smp_control.pub["_homeostasis_motor_velocity"].publish(self.msg_motors_velocity)

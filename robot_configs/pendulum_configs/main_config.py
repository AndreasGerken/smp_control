import numpy as np

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from robot_configs.general_robot_config import RobotConfig


def get_class():
    return PendulumConfig


class PendulumConfig(RobotConfig):
    def __init__(self, args):
        RobotConfig.__init__(self)

        self.pub_names = {
            "/pendulumMotor": [Float32MultiArray],
        }
        self.sub_names = {
            "/imu/data": [Imu, self.cb_imu],
            "/poti": [Float32, self.cb_poti],
            "/current": [Float32, self.cb_current]
        }

        self.sensor_dimensions = {"acc": 3, "gyr": 3, "orient": 4, "motor_torque": 1,
                                  "motor_torque_abs": 1, "poti": 1, "poti_d": 1, "poti_integral": 1, "current": 1}

        self.smoothings = [0., 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]
        self.smoothings = [0,]

        #self.smoothings = [0.]
        self.smoothing_length = len(self.smoothings)

        #self.use_sensors = ['orient', 'gyr', 'motor_pos']
        self.use_sensors = None
        self.numsen = 0

        self.set_sensors(['poti', 'poti_d', 'poti_integral'])
        self.set_sensors(['poti'])
        #self.use_sensors = ['orient', 'motor_pos', 'gyr', 'acc']

        #self.numsen = np.sum([sensor_dimensions[sensor] for sensor in self.use_sensors])
        self.nummot = 1
        self.lag = 1
        self.embedding = 6
        self.output_gain = 150

        self.learning_enabled = True
        self.use_sensors_for_model = True

        #self.control_mode = control_modes[args.control_mode]

        # Initialize Ros messages
        self.msg_motors = Float32MultiArray()

        self.motor_torque_command = np.zeros((self.nummot))
        self.sensor_vec = np.zeros((self.numsen))
        self.imu_msg = None
        self.poti_old = None
        self.poti_integral = 0
        self.poti_avg = 0.5
        self.poti = None
        self.current_msg = None

    def cb_imu(self, msg):
        self.imu_msg = msg

    def cb_poti(self, msg):
        self.poti_old = self.poti
        self.poti = msg.data
        self.poti_avg = self.poti_avg * 0.9 + self.poti * 0.1
        self.poti_integral += (self.poti - self.poti_avg)

    def cb_current(self, msg):
        self.current_msg = msg

    def get_input(self):
        """ROS IMU callback"""
        sensor_tupel = ()

        if self.smp_control.cnt_main < 5:
            return

        if 'acc' in self.use_sensors:
            sensor_vec_acc = (self.imu_msg.linear_acceleration.x,
                           self.imu_msg.linear_acceleration.y, self.imu_msg.linear_acceleration.z)
            sensor_tupel += (sensor_vec_acc)

        if 'gyr' in self.use_sensors:
            sensor_vec_gyr = (self.imu_msg.angular_velocity.x,
                           self.imu_msg.angular_velocity.y, self.imu_msg.angular_velocity.z)
            sensor_tupel += (sensor_vec_gyr)

        if 'orient' in self.use_sensors:
            sensor_vec_orient = (self.imu_msg.orientation.x, self.imu_msg.orientation.y,
                              self.imu_msg.orientation.z, self.imu_msg.orientation.w)
            sensor_tupel += (sensor_vec_orient)

        if 'motor_torque' in self.use_sensors:
            sensor_tupel += (self.motor_torque_command,)

        if 'motor_torque_abs' in self.use_sensors:
            sensor_tupel += (np.abs(self.motor_torque_command),)

        if 'poti' in self.use_sensors:
            sensor_tupel += (np.array(self.poti), )

        if 'poti_d' in self.use_sensors:
            sensor_tupel += (np.array(self.poti_old - self.poti), )

        if 'poti_integral' in self.use_sensors:
            sensor_tupel += (np.array(self.poti_integral), )

        if 'current' in self.use_sensors:
            sensor_tupel += (np.array(self.current_msg.data), )

        # check if some data was gathered
        if(len(sensor_tupel) == 0):
            return None

        old_sensor_vec = self.sensor_vec
        real_numsen = self.numsen / self.smoothing_length

        # bring them data in an array
        new_sensor_vec = np.hstack(sensor_tupel)
        for i in range(self.smoothing_length):
            s = real_numsen * i
            e = real_numsen * (i + 1)

            # if it's the first timestep with sensor readings use them as initial values of all smoothing steps
            if self.smp_control.cnt_main == 5:
                self.sensor_vec[s:e] = new_sensor_vec
            else:
                self.sensor_vec[s:e] = old_sensor_vec[s:e] * \
                    self.smoothings[i] + new_sensor_vec * \
                    (1. - self.smoothings[i])
        # print " ".join(["{0:0.2f}".format(i) for i in self.sensor_vec])

        return self.sensor_vec

    def send_output(self, algorithm_output):
        #self.motor_torque_command = np.array([np.sign(algorithm_output[0]) * np.abs(algorithm_output[1])])
        #self.motor_torque_command =  np.array([1.])
        self.motor_torque_command = algorithm_output

        # write the commands to the message and publish them
        self.msg_motors.data = self.motor_torque_command * self.output_gain
        self.smp_control.pub["_pendulumMotor"].publish(self.msg_motors)

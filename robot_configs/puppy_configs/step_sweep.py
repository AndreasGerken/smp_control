#from puppy_config import PuppyConfig
from robot_configs.puppy_configs.main_config import PuppyConfig

import numpy as np

import matplotlib.pyplot as plt


def get_class():
    return PuppyConfigStepSweep


class PuppyConfigStepSweep(PuppyConfig):
    # run this with 100hz variant of puppy
    # python smp_control.py robot_configs/puppy_configs/step_sweep.py -n 60000 --pickle_name "step_sweep.pickle" -lt 0.01

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
        self.step_size = 0.333  # 0.333 corresponds to 30 deg
        self.repeat_step = 15
        self.initialized = False

    def send_output(self, algorithm_output):
        if not self.initialized:
            self.initialized = True
            print "initializing robot"

            self.smp_control.pickler.add_once_variables(
                ['robot.step_length', 'robot.reset_length', 'robot.step_size', 'robot.repeat_step'])

            cycle_length = self.step_length + self.reset_length
            cycle_max = self.smp_control.numtimesteps / cycle_length
            steps_max = cycle_max / self.repeat_step

            if self.smp_control.numtimesteps % cycle_length != 0 or cycle_max % self.repeat_step != 0:
                print "numtimesteps does not fit the cycle length or the repeat steps. To fix it it should be %d" % (steps_max * self.repeat_step * cycle_length)

            sweep_angle_total = 2. - self.step_size
            angle_per_step = sweep_angle_total / (steps_max -1)

            # write the whole sequence
            for i in range(self.smp_control.numtimesteps):
                cycle_number = i / cycle_length
                step_number = cycle_number / self.repeat_step

                position_in_cycle = i % cycle_length

                if position_in_cycle > self.reset_length:
                    # high position

                    angular_command = angle_per_step * step_number + self.step_size - 1.
                else:
                    # low position
                    angular_command = angle_per_step * step_number - 1.

                self.smp_control.y[i, :] = np.array([angular_command] * 4)

            # display the motor data
            # plt.plot(self.smp_control.y[:,0])
            # plt.show()

        self.motor_position_commands = self.smp_control.y[self.smp_control.cnt_main]
        self.msg_motors.data = self.motor_position_commands * self.output_gain
        self.smp_control.pub["_puppyMotor"].publish(self.msg_motors)


if __name__ == "__main__":
    PuppyConfigStepSweep(None)

from abc import ABCMeta, abstractmethod
import numpy as np

class RobotConfig():
    __metaclass__ = ABCMeta


    requiredProperties = ['use_sensors','sensor_dimensions','classname','learning_enabled','pub_names','lag','embedding','numsen','nummot','sensor_vec']

    def __init__(self):
        self.classname = self.__class__.__name__

        # gets set by smp_control
        self.pub = None
        self.smp_control = None

    def check_properties(self):
        """ This method checks the robot configuration if it has all required properties to work properly with smp_control"""
        for _property in RobotConfig.requiredProperties:
            assert _property in self.__dict__, str(_property) + " was not in dict"
        print "Robot configuration has all required properties"

    def set_sensors(self, use_sensors):
        self.use_sensors = use_sensors
        self.numsen = np.sum([self.sensor_dimensions[sensor]
                              for sensor in self.use_sensors])
        self.sensor_vec = np.zeros((self.numsen))
        print "sensor modalities used: [" + ",".join(self.use_sensors) + "]"


    @abstractmethod
    def get_input(self):
        """ This method is used to pass sensor data from the robot to the
        algorithm. The sensor data can be stored before and just passed with
        this function.

        Return: A numpy array with the length of self.numsen"""
        pass

    @abstractmethod
    def send_output(self, algorithm_output):
        """ The algorithm calls this message every timestep to pass the output
        to the robot. This function should pass the algorithm_output to the
        actuators.

        Arg:
            (np.array) algorithm_output: An array with the size of self.nummot

        Return:
            None
        """
        pass

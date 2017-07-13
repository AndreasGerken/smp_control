from abc import ABCMeta, abstractmethod

class RobotConfig():
    __metaclass__ = ABCMeta

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

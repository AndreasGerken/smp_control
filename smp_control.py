#!/usr/bin/env python

# general imports
import argparse
import numpy as np
import signal
import imp
import sys
import time
import os
import warnings

# ros imports
import rospy
from std_msgs.msg import Float32, Float32MultiArray

# smp imports
from smp_thread import smp_thread_ros
from pickler import Pickler

################################################################################
# helper funcs


def dtanh(x):
    """ this is the derivative of the tanh function """
    return 1 - np.tanh(x)**2


def idtanh(x):
    """ this is the inverse of the tanh function, with a protection against
    zero division zero """
    return 1. / (dtanh(x) + 0.00001)

################################################################################


class SMP_control(smp_thread_ros):
    """ This class provides the main functionality of running the algorithms
    homeostasis and homeokinesis on a ros enabled robot.
    """

    _modes = {'hs': 0, 'hk': 1}

    def __init__(self, args, robot_config):
        # save arguments
        self.mode = SMP_control._modes[args.mode]
        self.numtimesteps = args.numtimesteps
        self.loop_time = args.loop_time
        self.verbose = args.verbose
        self.robot = robot_config

        self.creativity = args.creativity
        self.epsC = args.epsC
        self.epsA = args.epsA

        pub_names = {
            '/lpzros/xsi': [Float32MultiArray, ],
            '/lpzros/EE': [Float32, ],
        }
        sub_names = {}

        pub_names.update(self.robot.pub_names)
        sub_names.update(self.robot.sub_names)

        # Robot specific
        self.lag = self.robot.lag
        self.embedding = self.robot.embedding

        smp_thread_ros.__init__(
            self, loop_time=self.loop_time, pubs=pub_names, subs=sub_names)

        self.robot.smp_control = self

        # initialize variables
        self.numsen = self.robot.numsen
        self.nummot = self.robot.nummot
        self.numsen_embedding = self.numsen * self.embedding
        self.nummot_embedding = self.nummot * self.embedding

        self.cnt_main = 0

        # This array saves all sensor data
        self.x = np.zeros((self.numtimesteps, self.numsen))
        self.x_pred = np.zeros((self.numsen, 1))

        # Shows the prediction from just one motor + the bias
        self.x_pred_coefficients = np.zeros((self.nummot_embedding + 1, self.numsen))

        # This array saves all motor data
        self.y = np.zeros((self.numtimesteps, self.nummot))

        # learning variables

        # Model
        if self.robot.use_sensors_for_model:
            self.A = np.random.uniform(-1e-4, 1e-4,
                                       (self.numsen, self.nummot_embedding + self.numsen_embedding))
        else:
            self.A = np.random.uniform(-1e-4, 1e-4,
                                       (self.numsen, self.nummot_embedding))
        self.b = np.zeros((self.numsen, 1))

        self.EE = 0.

        # Controller
        self.C = np.random.uniform(-1e-4, 1e-4,
                                   (self.nummot, self.numsen_embedding))
        self.h = np.zeros((self.nummot, 1))

        self.g = np.tanh  # sigmoidal activation function
        self.g_ = dtanh  # derivative of sigmoidal activation function

        self.L = np.zeros((self.numsen_embedding, self.nummot))
        self.v_avg = np.zeros((self.numsen_embedding, 1))
        self.xsi = np.zeros((self.numsen, 1))

        self.xsi_avg = 0
        self.xsi_avg_smooth = 0.01

        self.pickler = Pickler(self, self.numtimesteps)
        self.pickler.add_once_variables(
            ['loop_time', 'numtimesteps', 'cnt_main', 'x', 'y', 'epsC', 'epsA', 'creativity', 'nummot', 'numsen', 'lag', 'embedding', 'pickle_name'])
        self.pickler.add_frequent_variables(['A', 'b', 'C', 'h', 'x_pred', 'x_pred_coefficients', 'xsi', 'EE'])

        self.pickle_name = args.pickle_name
        self.pickle_name_newest = 'newest.pickle'
        self.pickle_folder = 'pickles'

        self.msg_xsi = Float32MultiArray()

    def run(self):
        """ Main loop of the algorithms, runs until the maximum timesteps are
        reached or the loop is canceled.
        """
        self.pickler.add_once_variables(
            ['robot.use_sensors', 'robot.sensor_dimensions', 'robot.classname'])

        # initialize motors and wait
        print('initializing motors')
        time.sleep(1)
        self.check_and_send_output()
        time.sleep(5)
        print('starting')
        while self.isrunning:
            # check if finished
            if self.cnt_main == self.numtimesteps:
                self.exit_loop()
                return

            self.get_and_check_input()

            if self.robot.learning_enabled:
                self.compute_new_output()

                self.learning_step()

            self.check_and_send_output()

            self.pickler.save_frequent_variables_to_buffer(self.cnt_main)

            self.cnt_main += 1

            self.rate.sleep()

    def get_and_check_input(self):
        """ Gathers the input from the robot and checks if the dimensionality
        is correct, then it saves the new input to the x matrix.
        """
        inputs = self.robot.get_input()

        # check input dimensionality

        if(inputs is None or len(inputs) != self.numsen):
            if inputs is None or self.cnt_main < 3:
                return
            else:
                raise Exception("numsen doesn't match up with the real input data dimensionality numsen: " +
                                str(self.numsen) + ', len: ' + str(len(inputs)))

        if self.verbose:
            print 'Inputs:\t', inputs

        # save the input (casting from (numsen,1) to (numsen))
        self.x[self.cnt_main, :] = inputs

    def compute_new_output(self):
        """ Computes a new output from the sensor readings and the controller
        variables. This output is saved to the y matrix.
        """
        if self.cnt_main < self.embedding:
            return
        x_fut_emb = self.x[self.cnt_main -
                           self.embedding: self.cnt_main, :].flatten()
        Cx_fut = np.dot(self.C, x_fut_emb).reshape((self.nummot, 1))

        self.y[self.cnt_main, :] = self.g(Cx_fut + self.h)[:, 0]

        if self.verbose:
            print 'x_fut:\t', x_fut_emb
            print 'C:\t', self.C
            print 'Cx_fut\t: ', Cx_fut
            print 'new y\t', self.y[self.cnt_main, :]

    def check_and_send_output(self):
        """ Gets the output from the y matrix, checks the dimensionality and
        commands the robot class to send the output. The robot configuration can overwrite the command by returning something else.
        """
        motor_output = self.y[self.cnt_main, :]

        # check output dimensionality
        if(len(motor_output) != self.nummot):
            raise Exception("nummot doesn't match up with the real input data dimensionality nummot: " +
                            str(self.nummot) + ', len: ' + str(len(motor_output)))

        if self.verbose:
            print 'Outputs: ', motor_output

        robot_response = self.robot.send_output(motor_output)

        if robot_response is not None:
            self.y[self.cnt_main, :] = robot_response

    def learning_step(self):
        """ One learning step of the learning algorithm (homeostasis or homeokinesis)"""

        sensor_input = self.x[self.cnt_main, :]

        if self.cnt_main <= self.lag + self.embedding:
            return

        # local variables
        # results in lagged (numsen_embedding, 1) vector
        x_lag_emb = np.atleast_2d(
            self.x[self.cnt_main - self.lag - self.embedding: self.cnt_main - self.lag, :].flatten()).T

        # results in (numsen,1) vector
        x_fut = np.atleast_2d(self.x[self.cnt_main, :].flatten()).T

        # results in lagged (nummot_embedding,1) vector
        y_lag_emb = np.atleast_2d(
            self.y[self.cnt_main - self.lag - self.embedding: self.cnt_main - self.lag, :].flatten()).T

        # attach the sensor data to the model input
        if self.robot.use_sensors_for_model:
            y_lag_emb = np.vstack((y_lag_emb, x_lag_emb))

        z = np.dot(self.C, x_lag_emb + self.v_avg * self.creativity) + self.h

        g_prime = dtanh(z)  # derivative of g
        g_prime_inv = idtanh(z)  # inverse derivative of g

        # cut of embedding
        g_prime_inv = g_prime_inv[:self.nummot, :]

        # forward prediction error xsi
        # clipping prevents overflow in unstable episodes
        self.x_pred = np.dot(self.A, y_lag_emb) + self.b

        # check from which motor the prediction comes
        for i in range(self.nummot_embedding + 1):
            y_lag_emb_cutout = np.zeros_like(y_lag_emb)

            if i != self.nummot_embedding:
                y_lag_emb_cutout[i] = y_lag_emb[i]
                bias = 0
            else:
                bias = 1

            self.x_pred_coefficients[i] = (np.dot(self.A, y_lag_emb_cutout) + bias * self.b)[0]

        self.xsi = np.clip(
            x_fut - self.x_pred, -1e+38, 1e+38)
        self.xsi_avg = np.sum(np.abs(self.xsi)) * self.xsi_avg_smooth + \
            (1 - self.xsi_avg_smooth) * self.xsi_avg

        self.msg_xsi.data = self.xsi.flatten().tolist()
        self.pub['_lpzros_xsi'].publish(self.msg_xsi)

        if(self.verbose):
            print "\n"
            print "x %s\t" % str(x_lag_emb)
            print "y %s\t" % str(y_lag_emb)
            print "A %s\t" % str(self.A)
            print "x_fut %s\t" %str(x_fut)
            print 'g_prime\t', g_prime
            print 'g_prime_inv\t', g_prime_inv
            print "Xsi Average %f\t" % self.xsi_avg
            print "\n"

        """
        forward model learning
        """
        # cooling of the modelmatrix
        self.A += self.epsA * \
            np.dot(self.xsi, y_lag_emb.T) + (self.A * -0.001)
        self.b += self.epsA * self.xsi + (self.b * -0.001)

        """
        controller learning
        """

        if self.mode == 0:  # homestastic learning
            eta = np.dot(self.A.T, self.xsi)

            dC = np.dot(eta * g_prime, x_lag_emb.T) * self.epsC
            dh = eta * g_prime * self.epsC

            if self.verbose:
                print dC, dh
                print 'eta', eta.shape, eta

        elif self.mode == 1:  # TLE / homekinesis
            # TODO: why is this different to eta in homeostasis?
            eta = np.dot(np.linalg.pinv(self.A), self.xsi)

            # cut of embedding and sensors attached
            eta = eta[:self.nummot, :]

            # after M inverse
            # zeta = eta * g_prime_inv
            zeta = np.clip(eta * g_prime_inv, -1., 1.)

            lambda_ = np.eye(self.nummot) * \
                np.random.uniform(-0.01, 0.01, self.nummot)
            mue = np.dot(np.linalg.pinv(
                np.dot(self.C, self.C.T) + lambda_), zeta)

            # after C inverse
            v = np.dot(self.C.T, mue)
            v = np.clip(np.dot(self.C.T, mue), -1., 1.)

            # moving average
            self.v_avg += (v - self.v_avg) * 0.1

            self.EE = .1 / (np.square(np.linalg.norm(v)) + 0.001)

            # Includes cooling of the control matrix
            dC = (np.dot(mue, v.T) + (np.dot((mue *
                                              y_lag_emb[:self.nummot] * zeta), -2 * x_lag_emb.T))) * self.EE * self.epsC + (self.C * -0.0003)
            dh = mue * y_lag_emb[:self.nummot] * zeta * -2 * \
                self.EE * self.epsC + (self.h * -0.0001)

            # publishing for homeokinesis
            self.pub['_lpzros_EE'].publish(self.EE)

            if self.verbose:
                print 'v', v
                print 'v_avg', self.v_avg
                print 'eta', eta
                print 'zeta', zeta

        self.C += np.clip(dC, -.1, .1)
        self.h += np.clip(dh, -.1, .1)

        if self.verbose:
            print 'C:\n', self.C
            print 'A:\n', self.A

    def exit_loop(self):
        """ Ends the loop and saves the data to a pickle file """
        self.isrunning = False
        self._save_pickle()

        if hasattr(self.robot.__class__, 'before_exit') and callable(getattr(self.robot.__class__, 'before_exit')):
            self.robot.before_exit()

        # generates problem with batch mode
        rospy.signal_shutdown('ending')
        print('ending')

    def _save_pickle(self):
        if self.cnt_main != self.numtimesteps:
            user_input = raw_input(
                "The run was not complete, do you still want to save it? (Y/N)")
            if not user_input.upper() is "Y":
                return

        # save as newest
        self.pickler.save_pickle(
            self.pickle_folder + "/" + self.pickle_name_newest, verbose=False)

        # if a pickle name was given, use it
        if self.pickle_name != "":
                self.pickler.save_pickle(
                    self.pickle_folder + "/" + self.pickle_name)
                return


        # find a new name, indicating eC, eA, c and an running id
        id = 0
        while True:
            self.pickle_name = "recording_eC%.2f_eA%.2f_c%.2f_id%d.pickle" % (
                self.epsC, self.epsA, self.creativity, id)


            if not os.path.exists(self.pickle_folder + "/" + self.pickle_name):
                self.pickler.save_pickle(
                    self.pickle_folder + "/" + self.pickle_name)
                return

            # increment id and try again
            id += 1
            if id >= 10000:
                raise Exception(
                    "While searching for an id to save the pickle 10000 was reached. Did you really do so many?")


def dynamic_importer(name):
    """
    Dynamically imports modules / classes
    """
    try:
        fp, pathname, description = imp.find_module(name)
        print 'module was found'
    except ImportError:
        print 'unable to locate module: ' + name
        return (None, None)

    try:
        package = imp.load_module(name, fp, pathname, description)
        print 'package was found'
    except Exception, e:
        print 'unable to load module:\n' + str(e)
        return (None, None)

    try:
        myclass = package.get_class()
        print 'class was found'
        #myclass = imp.load_module("%s.%s" % (name, class_name), fp, pathname, description)
        print myclass
    except Exception, e:
        print 'unable to get class:\n' + str(e)
        return (None, None)

    return package, myclass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Implementation of the self exploration algorithms homeostasis and homeokinesis')
    parser.add_argument('file')
    parser.add_argument('-m', '--mode', type=str,
                        help='Select the learning algorithm, hs for Homeokinesis and hk for Homeostasis, default: hk', default='hk')
    parser.add_argument('-n', '--numtimesteps', type=int,
                        help='Episode length in timesteps, default: 1000', default=1000)
    parser.add_argument('-lt', '--loop_time', type=float,
                        help='Delay between the timesteps in the loop, default: 0.05', default=0.05)
    parser.add_argument('-eC', '--epsC', type=float,
                        help='learning rate for controller', default=0.1)
    parser.add_argument('-eA', '--epsA', type=float,
                        help='learning rate for model', default=0.01)
    parser.add_argument('-c', '--creativity', type=float,
                        help='creativity', default=0.5)
    parser.add_argument('-v', '--verbose', type=bool,
                        help='Print debug information, default: False', default=False)
    parser.add_argument("-pickle", "--pickle_name", type=str, default="")
    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        warnings.warn("Unknown arguments were parsed in smp_control. They will be parsed by the robot configuration again:")
        print "unknown_args: %s" % str(unknown_args)

    # import the robot class
    class_name = args.file.split('.py')[0]
    robot_file, robot_class = dynamic_importer(class_name)
    robot = robot_class(args)

    print "config class %s loaded" % (robot.classname)

    # check if robot has all required properties
    robot.check_properties()

    smp_control = SMP_control(args, robot)

    def handler(signum, frame):
        print 'Signal handler called with signal', signum
        smp_control.exit_loop()
        sys.exit(0)

    # install interrupt handler
    signal.signal(signal.SIGINT, handler)

    smp_control.start()

    # prevent main from exiting
    while smp_control.isrunning:
        time.sleep(1)
